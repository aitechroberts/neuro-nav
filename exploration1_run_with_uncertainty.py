#!/usr/bin/env python3
"""
EXPLORATION 1: Run Mapping with Uncertainty Visualization
==========================================================

This script runs the full ConceptGraphs mapping pipeline with real-time
uncertainty visualization overlaid.

Usage:
    python exploration1_run_with_uncertainty.py

Configuration:
    Uses the same config as rerun_simple_test but adds uncertainty viz

Author: Jesse (CMU 11851 - Talking to Robots)
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch
from collections import Counter
import colorsys

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Standard library imports
import copy
import uuid
import pickle
import gzip

# Third-party imports
import cv2
import scipy.ndimage as ndi
from PIL import Image
from tqdm import trange
from open3d.io import read_pinhole_camera_parameters
import hydra
from omegaconf import DictConfig
import open_clip
from ultralytics import YOLO, SAM
import supervision as sv

# Rerun
import rerun as rr

# Local imports
from conceptgraph.utils.optional_rerun_wrapper import (
    OptionalReRun, 
    orr_log_annotated_image, 
    orr_log_camera, 
    orr_log_depth_image, 
    orr_log_edges, 
    orr_log_objs_pcd_and_bbox, 
    orr_log_rgb_image, 
    orr_log_vlm_image
)
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.geometry import rotation_matrix_to_quaternion
from conceptgraph.utils.logging_metrics import DenoisingTracker, MappingTracker
from conceptgraph.utils.vlm import consolidate_captions, get_obj_rel_from_image_gpt4v, get_openai_client
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.general_utils import (
    ObjectClasses, 
    find_existing_image_path, 
    get_det_out_path, 
    get_exp_out_path, 
    get_vlm_annotated_image_path, 
    handle_rerun_saving, 
    load_saved_detections, 
    load_saved_hydra_json_config, 
    make_vlm_edges_and_captions, 
    measure_time, 
    save_detection_results,
    save_edge_json, 
    save_hydra_config,
    save_obj_json, 
    save_objects_for_frame, 
    save_pointcloud, 
    should_exit_early, 
    vis_render_image
)
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import (
    OnlineObjectRenderer, 
    save_video_from_frames, 
    vis_result_fast_on_depth, 
    vis_result_for_vlm, 
    vis_result_fast, 
    save_video_detections
)
from conceptgraph.slam.slam_classes import MapEdgeMapping, MapObjectList
from conceptgraph.slam.utils import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    denoise_objects,
    merge_objects, 
    detections_to_obj_pcd_and_bbox,
    prepare_objects_save_vis,
    process_cfg,
    process_edges,
    process_pcd,
    processing_needed,
    resize_gobs
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
    merge_obj_matches
)
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.utils.general_utils import get_vis_out_path, cfg_to_dict, check_run_detections

# Import our uncertainty visualization functions
from exploration1_uncertainty_viz import (
    compute_object_confidence,
    get_confidence_color,
    get_top_k_class_hypotheses
)

# Disable torch gradient computation
torch.set_grad_enabled(False)


def log_uncertainty_visualization(objects, obj_classes, min_confidence=0.0):
    """
    Log uncertainty visualization to Rerun.
    This is called every frame to update the uncertainty view.
    """
    # Clear previous uncertainty viz
    rr.log("world/uncertainty_viz", rr.Clear(recursive=False))
    
    for obj_idx, obj in enumerate(objects):
        # Skip background or insufficient detections
        if obj.get('is_background', False):
            continue
        if obj['num_detections'] < 1:
            continue
        
        # Compute confidence
        confidence_score, confidence_category, conf_details = compute_object_confidence(obj)
        
        # Filter by minimum confidence
        if confidence_score < min_confidence:
            continue
        
        # Get confidence-based color
        conf_color = get_confidence_color(confidence_score)
        
        # Get point cloud and bbox
        positions = np.asarray(obj['pcd'].points)
        bbox = obj['bbox']
        
        # Object label
        obj_label = f"{obj['curr_obj_num']}_{obj['class_name']}"
        obj_label = obj_label.replace(" ", "_")
        
        # Get alternative hypotheses
        top_classes = get_top_k_class_hypotheses(obj, obj_classes, k=3)
        alt_hypotheses_str = " | ".join([f"{name}({prob:.2f})" for name, prob in top_classes])
        
        # Build label with confidence info
        full_label = (
            f"{obj_label}\n"
            f"Conf: {confidence_score:.2f} ({confidence_category})\n"
            f"Dets: {obj['num_detections']}\n"
            f"Hypotheses: {alt_hypotheses_str}"
        )
        
        # === Log to Rerun ===
        entity_base = f"world/uncertainty_viz/{obj_label}"
        
        # 1. Point cloud with confidence color
        rr.log(
            f"{entity_base}/pcd",
            rr.Points3D(
                positions,
                colors=[conf_color],
            )
        )
        
        # 2. Bounding box with confidence color and label
        centers = [bbox.get_center()]
        # Handle both AxisAlignedBoundingBox and OrientedBoundingBox
        if hasattr(bbox, 'extent'):
            half_sizes = [bbox.extent / 2]
        else:
            extent = bbox.get_max_bound() - bbox.get_min_bound()
            half_sizes = [extent / 2]
        
        rr.log(
            f"{entity_base}/bbox",
            rr.Boxes3D(
                centers=centers,
                half_sizes=half_sizes,
                colors=[conf_color],
                labels=[full_label],
            )
        )
        
        # 3. Log detailed confidence metrics as metadata
        # Note: Time series plots removed due to Rerun API version compatibility
        # The confidence info is still visible in the labels above


@hydra.main(version_base=None, config_path="conceptgraph/hydra_configs/", config_name="rerun_simple_test")
def main(cfg : DictConfig):
    """
    Main function - runs mapping with uncertainty visualization
    """
    print("\n" + "🤖 " * 40)
    print("EXPLORATION 1: SEMANTIC UNCERTAINTY THEATER")
    print("Real-time Uncertainty Visualization During Mapping")
    print("🤖 " * 40 + "\n")
    
    tracker = MappingTracker()
    
    # Initialize Rerun
    rr.init("uncertainty_visualization", spawn=True)
    
    owandb = OptionalWandB()
    owandb.set_use_wandb(cfg.use_wandb)
    owandb.init(project="concept-graphs", 
                config=cfg_to_dict(cfg))
    
    cfg = process_cfg(cfg)

    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )

    objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)

    # output folder for this mapping experiment
    exp_out_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, "exploration1_uncertainty")

    # output folder of the detections experiment to use
    det_exp_path = get_exp_out_path(cfg.dataset_root, cfg.scene_id, cfg.detections_exp_suffix, make_dir=False)

    # Load class information
    detections_exp_cfg = cfg_to_dict(cfg)
    obj_classes = ObjectClasses(
        classes_file_path=detections_exp_cfg['classes_file'], 
        bg_classes=detections_exp_cfg['bg_classes'], 
        skip_bg=detections_exp_cfg['skip_bg']
    )

    # Check if we need to do detections
    run_detections = check_run_detections(cfg.force_detection, det_exp_path)
    det_exp_pkl_path = get_det_out_path(det_exp_path)
    det_exp_vis_path = get_vis_out_path(det_exp_path)
    
    prev_adjusted_pose = None

    if run_detections:
        print("\n".join(["Running detections..."] * 5))
        det_exp_path.mkdir(parents=True, exist_ok=True)

        ## Initialize the detection models
        detection_model = measure_time(YOLO)('yolov8l-world.pt')
        sam_predictor = SAM('mobile_sam.pt')
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", "laion2b_s32b_b79k"
        )
        clip_model = clip_model.to(cfg.device)
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")

        detection_model.set_classes(obj_classes.get_classes_arr())
    else:
        print("\n".join(["NOT Running detections..."] * 5))

    openai_client = None
    if cfg.make_edges:
        openai_client = get_openai_client()

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)

    exit_early_flag = False
    counter = 0
    
    # Add legend for uncertainty colors
    print("\n" + "="*80)
    print("UNCERTAINTY COLOR LEGEND:")
    print("  🔴 RED    (0.0-0.5): Uncertain - Robot is guessing")
    print("  🟡 YELLOW (0.5-0.8): Moderate - Robot is fairly sure")
    print("  🟢 GREEN  (0.8-1.0): Confident - Robot is very confident")
    print("="*80 + "\n")
    
    for frame_idx in trange(len(dataset), desc="Processing frames"):
        tracker.curr_frame_idx = frame_idx
        counter += 1
        rr.set_time_sequence("frame", frame_idx)

        # Check early exit
        if not exit_early_flag and should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True

        if exit_early_flag and frame_idx < len(dataset) - 1:
            continue

        # Read current frame data
        color_path = Path(dataset.color_paths[frame_idx])
        image_original_pil = Image.open(color_path)
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]

        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy()
        image_rgb = (color_np).astype(np.uint8)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        # Load or compute detections
        raw_gobs = None
        gobs = None
        detections_path = det_exp_pkl_path / (color_path.stem + ".pkl.gz")
        
        if run_detections:
            # [Detection code - same as original]
            image = cv2.imread(str(color_path))
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = detection_model.predict(color_path, conf=0.1, verbose=False)
            confidences = results[0].boxes.conf.cpu().numpy()
            detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" 
                                      for class_idx, class_id in enumerate(detection_class_ids)]
            xyxy_tensor = results[0].boxes.xyxy
            xyxy_np = xyxy_tensor.cpu().numpy()

            if xyxy_tensor.numel() != 0:
                sam_out = sam_predictor.predict(color_path, bboxes=xyxy_tensor, verbose=False)
                masks_tensor = sam_out[0].masks.data
                masks_np = masks_tensor.detach().cpu().numpy()
                if masks_np.dtype != np.bool_:
                    masks_np = masks_np > 0.5

                n_boxes = xyxy_np.shape[0]
                n_masks = masks_np.shape[0]
                if n_masks == 0 or n_boxes == 0:
                    continue

                n = min(n_boxes, n_masks)
                if n != n_boxes or n != n_masks:
                    xyxy_np = xyxy_np[:n]
                    confidences = confidences[:n]
                    detection_class_ids = detection_class_ids[:n]
                    masks_np = masks_np[:n]
            else:
                xyxy_np = np.empty((0, 4), dtype=np.float32)
                confidences = np.empty((0,), dtype=np.float32)
                detection_class_ids = np.empty((0,), dtype=np.int32)
                masks_np = np.empty((0, *color_tensor.shape[:2]), dtype=np.bool_)

            curr_det = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=detection_class_ids,
                mask=masks_np,
            )
            
            labels, edges, edge_image, captions = make_vlm_edges_and_captions(
                image, curr_det, obj_classes, detection_class_labels, 
                det_exp_vis_path, color_path, cfg.make_edges, openai_client)

            image_crops, image_feats, text_feats = compute_clip_features_batched(
                image_rgb, curr_det, clip_model, clip_preprocess, 
                clip_tokenizer, obj_classes.get_classes_arr(), cfg.device)

            tracker.increment_total_detections(len(curr_det.xyxy))

            results = {
                "xyxy": curr_det.xyxy,
                "confidence": curr_det.confidence,
                "class_id": curr_det.class_id,
                "mask": curr_det.mask,
                "classes": obj_classes.get_classes_arr(),
                "image_crops": image_crops,
                "image_feats": image_feats,
                "text_feats": text_feats,
                "detection_class_labels": detection_class_labels,
                "labels": labels,
                "edges": edges,
                "captions": captions,
            }

            raw_gobs = results

            if cfg.save_detections:
                vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg")
                annotated_image, labels = vis_result_fast(image, curr_det, obj_classes.get_classes_arr())
                cv2.imwrite(str(vis_save_path), annotated_image)
                save_detection_results(det_exp_pkl_path / vis_save_path.stem, results)
        else:
            # Load saved detections
            if os.path.exists(det_exp_pkl_path / color_path.stem):
                raw_gobs = load_saved_detections(det_exp_pkl_path / color_path.stem)
            elif os.path.exists(det_exp_pkl_path / f"{int(color_path.stem):06}"):
                raw_gobs = load_saved_detections(det_exp_pkl_path / f"{int(color_path.stem):06}")
            else:
                raise FileNotFoundError(f"No detections found for frame {frame_idx}")

        # Get pose
        unt_pose = dataset.poses[frame_idx]
        unt_pose = unt_pose.cpu().numpy()
        adjusted_pose = unt_pose
        
        # Log camera and images to Rerun
        prev_adjusted_pose = orr_log_camera(intrinsics, adjusted_pose, prev_adjusted_pose, 
                                            cfg.image_width, cfg.image_height, frame_idx)
        
        # Use standard rerun logging (rr) instead of orr for direct control
        rr.log("world/camera/rgb", rr.ImageEncoded(path=str(color_path)))
        rr.log("world/camera/depth", rr.DepthImage(depth_tensor.numpy(), meter=0.9999999))

        # Resize and filter observations
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        filtered_gobs = filter_gobs(resized_gobs, image_rgb, 
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )

        gobs = filtered_gobs

        if len(gobs['mask']) == 0:
            continue

        gobs['mask'] = mask_subtract_contained(gobs['xyxy'], gobs['mask'])

        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth_array,
            masks=gobs['mask'],
            cam_K=intrinsics.cpu().numpy()[:3, :3],
            image_rgb=image_rgb,
            trans_pose=adjusted_pose,
            min_points_threshold=cfg.min_points_threshold,
            spatial_sim_type=cfg.spatial_sim_type,
            obj_pcd_max_points=cfg.obj_pcd_max_points,
            device=cfg.device,
        )

        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                )
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=cfg['spatial_sim_type'], 
                    pcd=obj["pcd"],
                )

        detection_list = make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx
        )

        if len(detection_list) == 0:
            continue

        if len(objects) == 0:
            objects.extend(detection_list)
            tracker.increment_total_objects(len(detection_list))
            owandb.log({
                "total_objects_so_far": tracker.get_total_objects(),
                "objects_this_frame": len(detection_list),
            })
            # Log initial uncertainty visualization
            log_uncertainty_visualization(objects, obj_classes, min_confidence=0.0)
            continue 

        # Compute similarities and merge
        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=cfg['spatial_sim_type'], 
            detection_list=detection_list, 
            objects=objects,
            downsample_voxel_size=cfg['downsample_voxel_size']
        )

        visual_sim = compute_visual_similarities(detection_list, objects)

        agg_sim = aggregate_similarities(
            match_method=cfg['match_method'], 
            phys_bias=cfg['phys_bias'], 
            spatial_sim=spatial_sim, 
            visual_sim=visual_sim
        )

        match_indices = match_detections_to_objects(
            agg_sim=agg_sim, 
            detection_threshold=cfg['sim_threshold']
        )

        objects = merge_obj_matches(
            detection_list=detection_list, 
            objects=objects, 
            match_indices=match_indices,
            downsample_voxel_size=cfg['downsample_voxel_size'], 
            dbscan_remove_noise=cfg['dbscan_remove_noise'], 
            dbscan_eps=cfg['dbscan_eps'], 
            dbscan_min_points=cfg['dbscan_min_points'], 
            spatial_sim_type=cfg['spatial_sim_type'], 
            device=cfg['device']
        )
        
        # Fix class names
        for idx, obj in enumerate(objects):
            temp_class_name = obj["class_name"]
            curr_obj_class_id_counter = Counter(obj['class_id'])
            most_common_class_id = curr_obj_class_id_counter.most_common(1)[0][0]
            most_common_class_name = obj_classes.get_classes_arr()[most_common_class_id]
            if temp_class_name != most_common_class_name:
                obj["class_name"] = most_common_class_name

        map_edges = process_edges(match_indices, gobs, len(objects), objects, map_edges, frame_idx)
        is_final_frame = frame_idx == len(dataset) - 1
        
        if is_final_frame:
            print("Final frame detected. Performing final post-processing...")

        # Post-processing
        if processing_needed(cfg["denoise_interval"], cfg["run_denoise_final_frame"], 
                           frame_idx, is_final_frame):
            objects = measure_time(denoise_objects)(
                downsample_voxel_size=cfg['downsample_voxel_size'], 
                dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                dbscan_eps=cfg['dbscan_eps'], 
                dbscan_min_points=cfg['dbscan_min_points'], 
                spatial_sim_type=cfg['spatial_sim_type'], 
                device=cfg['device'], 
                objects=objects
            )

        if processing_needed(cfg["filter_interval"], cfg["run_filter_final_frame"], 
                           frame_idx, is_final_frame):
            objects = filter_objects(
                obj_min_points=cfg['obj_min_points'], 
                obj_min_detections=cfg['obj_min_detections'], 
                objects=objects,
                map_edges=map_edges
            )

        # === LOG UNCERTAINTY VISUALIZATION ===
        # This is the key addition - log uncertainty viz every frame
        log_uncertainty_visualization(objects, obj_classes, min_confidence=0.0)
        
        # Also log standard visualization
        orr_log_objs_pcd_and_bbox(objects, obj_classes)
        orr_log_edges(objects, map_edges, obj_classes)

    # End of loop
    print("\n" + "="*80)
    print("PROCESSING COMPLETE!")
    print("="*80)
    
    # Print final statistics
    print("\nFinal Confidence Statistics:")
    uncertain_count = 0
    moderate_count = 0
    confident_count = 0
    
    for obj in objects:
        if obj.get('is_background', False) or obj['num_detections'] < 1:
            continue
        confidence_score, confidence_category, _ = compute_object_confidence(obj)
        if confidence_category == "uncertain":
            uncertain_count += 1
        elif confidence_category == "moderate":
            moderate_count += 1
        else:
            confident_count += 1
    
    print(f"  🔴 Uncertain objects:  {uncertain_count}")
    print(f"  🟡 Moderate objects:   {moderate_count}")
    print(f"  🟢 Confident objects:  {confident_count}")
    print(f"  📊 Total objects:      {len(objects)}")
    print()
    
    # Save results
    if cfg.save_pcd:
        save_pointcloud(
            exp_suffix="exploration1_uncertainty",
            exp_out_path=exp_out_path,
            cfg=cfg,
            objects=objects,
            obj_classes=obj_classes,
            latest_pcd_filepath=cfg.latest_pcd_filepath,
            create_symlink=True,
            edges=map_edges
        )

    owandb.finish()
    
    print("✓ Visualization complete. Check the Rerun viewer!")
    print("  Look for 'world/uncertainty_viz' in the timeline view")
    print()


if __name__ == "__main__":
    main()

