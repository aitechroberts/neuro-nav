'''
Hybrid mapping script for GPU batch processing:
- Keeps VLM edges/captions and robust detection/mask handling from rerun_realtime_mapping.py
- Removes all ReRun session/logging
- Adds JSON edges/objects saving and passes edges to pointcloud saving
- Uses the dedicated Hydra config `batch_vlm_mapping` for non-interactive runs.
'''

# Standard library imports
import os
import gzip
import pickle
from pathlib import Path
from collections import Counter

# Third-party imports
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import trange
# from open3d.io import read_pinhole_camera_parameters # REMOVED for headless batch
import hydra
from omegaconf import DictConfig
import open_clip
from ultralytics import YOLO, SAM
import supervision as sv

# Local application/library specific imports
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.logging_metrics import MappingTracker
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.vlm import consolidate_captions, get_openai_client
from conceptgraph.utils.ious import mask_subtract_contained
# from conceptgraph.utils.vis import (
#     OnlineObjectRenderer, # REMOVED for headless batch
#     save_video_from_frames,
#     vis_result_fast_on_depth,
#     vis_result_fast,
#     save_video_detections,
# )
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.utils.general_utils import (
    ObjectClasses,
    get_det_out_path,
    # get_vis_out_path, # REMOVED for headless batch
    handle_rerun_saving,  # safe to keep import; not used
    load_saved_detections,
    load_saved_hydra_json_config,
    make_vlm_edges_and_captions,
    measure_time,
    save_detection_results,
    save_hydra_config,
    save_obj_json,
    save_pointcloud,
    should_exit_early,
    cfg_to_dict,
    check_run_detections,
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
    process_cfg,
    processing_needed,
    resize_gobs,
    process_edges,
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
    merge_obj_matches,
)

# Disable torch gradient computation
torch.set_grad_enabled(False)


def _resolve_output_base(cfg: DictConfig) -> Path:
    """
    Allow local smoke tests to direct experiment outputs to a separate
    bind-mounted directory (e.g., /mnt/local-output) while still reading
    datasets from the usual dataset_root. Precedence:
      1. Hydra config value `output_root` (if provided)
      2. Environment variable OUTPUT_ROOT
      3. Fall back to cfg.dataset_root
    """
    output_override = None
    if "output_root" in cfg:
        output_override = cfg.get("output_root")
    if not output_override:
        output_override = os.getenv("OUTPUT_ROOT")
    if output_override:
        return Path(output_override)
    return Path(cfg.dataset_root)


def _build_exp_path(base_root: Path, scene_id: str, exp_suffix: str, create: bool = True) -> Path:
    path = base_root / scene_id / "exps" / exp_suffix
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path

@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="batch_vlm_mapping")
def main(cfg: DictConfig):
    tracker = MappingTracker()

    owandb = OptionalWandB()
    owandb.set_use_wandb(cfg.use_wandb)
    owandb.init(
        project="concept-graphs",
        config=cfg_to_dict(cfg),
    )
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

    # REMOVED: Optional visualization (disabled by default for batch)
    # if cfg.vis_render:
    #     view_param = read_pinhole_camera_parameters(cfg.render_camera_path)
    #     obj_renderer = OnlineObjectRenderer(
    #         view_param=view_param,
    #         base_objects=None,
    #         gray_map=False,
    #     )
    #     frames = []

    # Output folders (optionally redirected via OUTPUT_ROOT / output_root)
    output_base = _resolve_output_base(cfg)
    exp_out_path = _build_exp_path(output_base, cfg.scene_id, cfg.exp_suffix, create=True)
    det_exp_path = _build_exp_path(output_base, cfg.scene_id, cfg.detections_exp_suffix, create=False)

    # Classes must match detection experiment
    detections_exp_cfg = cfg_to_dict(cfg)
    obj_classes = ObjectClasses(
        classes_file_path=detections_exp_cfg["classes_file"],
        bg_classes=detections_exp_cfg["bg_classes"],
        skip_bg=detections_exp_cfg["skip_bg"],
    )

    # Detection mode
    run_detections = check_run_detections(cfg.force_detection, det_exp_path)
    det_exp_pkl_path = get_det_out_path(det_exp_path)
    # det_exp_vis_path = get_vis_out_path(det_exp_path) # REMOVED for headless batch

    # Define a temporary or dummy path for VLM visual prompt generation if needed
    det_exp_vis_path = det_exp_path / "vlm_temp"
    if cfg.make_edges:
        det_exp_vis_path.mkdir(parents=True, exist_ok=True)

    # Initialize detectors and CLIP (GPU-aware)
    if run_detections:
        print("\n".join(["Running detections..."] * 3))
        det_exp_path.mkdir(parents=True, exist_ok=True)

        # Check if CKPT_DIR env var is set (e.g. /mnt/checkpoints) to use local weights
        ckpt_dir = os.environ.get("CKPT_DIR", "")
        
        # 1. Initialize YOLO with local weights check
        yolo_weights = "yolov8l-worldv2.pt"
        if ckpt_dir and (Path(ckpt_dir) / yolo_weights).exists():
            yolo_weights = str(Path(ckpt_dir) / yolo_weights)
            print(f"Using local YOLO weights: {yolo_weights}")
        
        detection_model = measure_time(YOLO)(yolo_weights)

        # 2. Initialize SAM with local weights check
        sam_weights = "sam2.1_b.pt"
        if ckpt_dir and (Path(ckpt_dir) / sam_weights).exists():
            sam_weights = str(Path(ckpt_dir) / sam_weights)
            print(f"Using local SAM weights: {sam_weights}")
            
        sam_predictor = SAM(sam_weights)

        # 3. Initialize CLIP (MobileCLIP2-S3) with HF cache check
        model_name = "MobileCLIP2-S3"
        pretrained_tag = "dfndr2b"
        
        if ckpt_dir and os.path.exists(ckpt_dir):
             # Set HF cache to a subdirectory in checkpoints to keep it organized
             hf_cache_dir = os.path.join(ckpt_dir, "huggingface")
             os.makedirs(hf_cache_dir, exist_ok=True)
             os.environ["HF_HOME"] = hf_cache_dir
             print(f"Set HF_HOME to {hf_cache_dir}")

        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained_tag,
            cache_dir=os.environ.get("HF_HOME") # Explicitly pass cache_dir if set
        )
        clip_model = clip_model.to(cfg.device)
        clip_tokenizer = open_clip.get_tokenizer(model_name)
        detection_model.set_classes(obj_classes.get_classes_arr())
    else:
        print("\n".join(["NOT Running detections..."] * 3))
        clip_model = None
        clip_preprocess = None
        clip_tokenizer = None

    # Only initialize OpenAI when edges/captions are enabled
    openai_client = None
    if cfg.make_edges:
        openai_client = get_openai_client()

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(detections_exp_cfg, exp_out_path, is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    exit_early_flag = False
    counter = 0

    for frame_idx in trange(len(dataset)):
        tracker.curr_frame_idx = frame_idx
        counter += 1

        # Early-exit support
        if not exit_early_flag and should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True
        if exit_early_flag and frame_idx < len(dataset) - 1:
            continue

        # Load frame
        color_path = Path(dataset.color_paths[frame_idx])
        image_original_pil = Image.open(color_path)
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]

        # Tensors to numpy
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy()
        image_rgb = color_np.astype(np.uint8)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"

        # Load or compute detections
        raw_gobs = None
        gobs = None
        # detections_path = det_exp_pkl_path / (color_path.stem + ".pkl.gz") # Unused variable

        if run_detections:
            # OpenCV image
            image = cv2.imread(str(color_path))
            image_rgb_bgr_fixed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            results = detection_model.predict(color_path, conf=0.1, verbose=False)
            confidences = results[0].boxes.conf.cpu().numpy()
            detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            detection_class_labels = [
                f"{obj_classes.get_classes_arr()[class_id]} {class_idx}"
                for class_idx, class_id in enumerate(detection_class_ids)
            ]
            xyxy_tensor = results[0].boxes.xyxy
            xyxy_np = xyxy_tensor.cpu().numpy()

            # Robust SAM handling with alignment and dtype fixes
            if xyxy_tensor.numel() != 0:
                sam_out = sam_predictor.predict(color_path, bboxes=xyxy_tensor, verbose=False)
                masks_tensor = sam_out[0].masks.data  # (M, H, W)
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
                H, W = image_rgb.shape[:2]
                masks_np = np.empty((0, H, W), dtype=np.bool_)

            curr_det = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=detection_class_ids,
                mask=masks_np,
            )

            # VLM edges/captions (optional) and CLIP features
            labels, edges, edge_image, captions = make_vlm_edges_and_captions(
                image, curr_det, obj_classes, detection_class_labels, det_exp_vis_path, color_path, cfg.make_edges, openai_client
            )

            image_crops, image_feats, text_feats = compute_clip_features_batched(
                image_rgb_bgr_fixed, curr_det, clip_model, clip_preprocess, clip_tokenizer, obj_classes.get_classes_arr(), cfg.device
            )

            tracker.increment_total_detections(len(curr_det.xyxy))

            raw_gobs = {
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

            if cfg.save_detections:
                # REMOVED: vis_save_path = (det_exp_vis_path / color_path.name).with_suffix(".jpg")
                # REMOVED: annotated_image, _ = vis_result_fast(image, curr_det, obj_classes.get_classes_arr())
                # REMOVED: cv2.imwrite(str(vis_save_path), annotated_image)

                # REMOVED: depth_image_rgb = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX)
                # REMOVED: depth_image_rgb = depth_image_rgb.astype(np.uint8)
                # REMOVED: depth_image_rgb = cv2.cvtColor(depth_image_rgb, cv2.COLOR_GRAY2BGR)
                # REMOVED: annotated_depth_image, _ = vis_result_fast_on_depth(depth_image_rgb, curr_det, obj_classes.get_classes_arr())
                # REMOVED: cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth.jpg"), annotated_depth_image)
                # REMOVED: cv2.imwrite(str(vis_save_path).replace(".jpg", "_depth_only.jpg"), depth_image_rgb)

                save_detection_results(det_exp_pkl_path / color_path.stem, raw_gobs)
        else:
            # Load saved detections (support current and old formats)
            if os.path.exists(det_exp_pkl_path / color_path.stem):
                raw_gobs = load_saved_detections(det_exp_pkl_path / color_path.stem)
            elif os.path.exists(det_exp_pkl_path / f"{int(color_path.stem):06}"):
                raw_gobs = load_saved_detections(det_exp_pkl_path / f"{int(color_path.stem):06}")
            else:
                raise FileNotFoundError(
                    f"No detections found for frame {frame_idx} at paths "
                    f"{det_exp_pkl_path / color_path.stem} or "
                    f"{det_exp_pkl_path / f'{int(color_path.stem):06}'}."
                )

        # Pose (no transformation)
        unt_pose = dataset.poses[frame_idx]
        unt_pose = unt_pose.cpu().numpy()
        adjusted_pose = unt_pose

        # Resize/filter observations
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        filtered_gobs = filter_gobs(
            resized_gobs,
            image_rgb,
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )
        gobs = filtered_gobs
        if len(gobs["mask"]) == 0:
            continue

        # Separate nested masks (e.g., pillows on couches)
        gobs["mask"] = mask_subtract_contained(gobs["xyxy"], gobs["mask"])

        # Build per-detection point clouds and bboxes
        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth_array,
            masks=gobs["mask"],
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
                    spatial_sim_type=cfg["spatial_sim_type"],
                    pcd=obj["pcd"],
                )

        detection_list = make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx
        )
        if len(detection_list) == 0:
            continue

        # If no objects yet, bootstrap the map
        if len(objects) == 0:
            objects.extend(detection_list)
            tracker.increment_total_objects(len(detection_list))
            owandb.log(
                {"total_objects_so_far": tracker.get_total_objects(), "objects_this_frame": len(detection_list)}
            )
            continue

        # Similarities, matching, merging
        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=cfg["spatial_sim_type"],
            detection_list=detection_list,
            objects=objects,
            downsample_voxel_size=cfg["downsample_voxel_size"],
        )
        visual_sim = compute_visual_similarities(detection_list, objects)
        agg_sim = aggregate_similarities(
            match_method=cfg["match_method"],
            phys_bias=cfg["phys_bias"],
            spatial_sim=spatial_sim,
            visual_sim=visual_sim,
        )
        match_indices = match_detections_to_objects(
            agg_sim=agg_sim, detection_threshold=cfg["sim_threshold"]
        )
        objects = merge_obj_matches(
            detection_list=detection_list,
            objects=objects,
            match_indices=match_indices,
            downsample_voxel_size=cfg["downsample_voxel_size"],
            dbscan_remove_noise=cfg["dbscan_remove_noise"],
            dbscan_eps=cfg["dbscan_eps"],
            dbscan_min_points=cfg["dbscan_min_points"],
            spatial_sim_type=cfg["spatial_sim_type"],
            device=cfg["device"],
        )

        # Fix object class names to most common detected class
        for obj in objects:
            curr_obj_class_id_counter = Counter(obj["class_id"])
            most_common_class_id = curr_obj_class_id_counter.most_common(1)[0][0]
            most_common_class_name = obj_classes.get_classes_arr()[most_common_class_id]
            if obj["class_name"] != most_common_class_name:
                obj["class_name"] = most_common_class_name

        # Edge processing and cleanup
        map_edges = process_edges(match_indices, gobs, len(objects), objects, map_edges, frame_idx)
        is_final_frame = frame_idx == len(dataset) - 1

        edges_to_delete = []
        for curr_map_edge in map_edges.edges_by_index.values():
            curr_obj1_idx = curr_map_edge.obj1_idx
            curr_obj2_idx = curr_map_edge.obj2_idx
            curr_first_detected = curr_map_edge.first_detected
            curr_num_det = curr_map_edge.num_detections
            if (frame_idx - curr_first_detected > 5) and curr_num_det < 2:
                edges_to_delete.append((curr_obj1_idx, curr_obj2_idx))
        for edge in edges_to_delete:
            map_edges.delete_edge(edge[0], edge[1])

        # Post-processing passes
        if processing_needed(cfg["denoise_interval"], cfg["run_denoise_final_frame"], frame_idx, is_final_frame):
            objects = measure_time(denoise_objects)(
                downsample_voxel_size=cfg["downsample_voxel_size"],
                dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"],
                dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"],
                device=cfg["device"],
                objects=objects,
            )

        if processing_needed(cfg["filter_interval"], cfg["run_filter_final_frame"], frame_idx, is_final_frame):
            objects = filter_objects(
                obj_min_points=cfg["obj_min_points"],
                obj_min_detections=cfg["obj_min_detections"],
                objects=objects,
                map_edges=map_edges,
            )

        if processing_needed(cfg["merge_interval"], cfg["run_merge_final_frame"], frame_idx, is_final_frame):
            if cfg["make_edges"]:
                objects, map_edges = measure_time(merge_objects)(
                    merge_overlap_thresh=cfg["merge_overlap_thresh"],
                    merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                    merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                    objects=objects,
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                    spatial_sim_type=cfg["spatial_sim_type"],
                    device=cfg["device"],
                    do_edges=True,
                    map_edges=map_edges,
                )
            else:
                objects = measure_time(merge_objects)(
                    merge_overlap_thresh=cfg["merge_overlap_thresh"],
                    merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                    merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                    objects=objects,
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                    spatial_sim_type=cfg["spatial_sim_type"],
                    device=cfg["device"],
                    do_edges=False,
                    map_edges=None,
                )

        # Save per-frame objects if needed
        if cfg.save_objects_all_frames:
            from conceptgraph.utils.general_utils import save_objects_for_frame
            save_objects_for_frame(
                obj_all_frames_out_path,
                frame_idx,
                objects,
                cfg.obj_min_detections,
                adjusted_pose,
                color_path,
            )

        # # Optional render (usually off in batch)
        # if cfg.vis_render:
        #     filtered_objects = [
        #         obj for obj in objects if obj["num_detections"] >= cfg.obj_min_detections and not obj["is_background"]
        #     ]
        #     objects_vis = MapObjectList([o for o in filtered_objects])
        #     if cfg.class_agnostic:
        #         objects_vis.color_by_instance()
        #     else:
        #         objects_vis.color_by_most_common_classes(obj_classes)
        #     rendered_image, vis = obj_renderer.step(
        #         image=image_original_pil,
        #         gt_pose=adjusted_pose,
        #         new_objects=objects_vis,
        #         paint_new_objects=False,
        #         return_vis_handle=cfg.debug_render,
        #     )
        #     if rendered_image is not None:
        #         rendered_image = (rendered_image * 255).astype(np.uint8)
        #         frame_info_text = f"Frame: {frame_idx}, Objects: {len(objects)}, Path: {str(color_path)}"
        #         font = cv2.FONT_HERSHEY_SIMPLEX
        #         font_scale = 0.5
        #         color = (255, 0, 0)
        #         thickness = 1
        #         line_type = cv2.LINE_AA
        #         position = (10, rendered_image.shape[0] - 10)
        #         cv2.putText(rendered_image, frame_info_text, position, font, font_scale, color, thickness, line_type)
        #         frames.append(rendered_image)
        #     if is_final_frame:
        #         frames_np = np.stack(frames)
        #         video_save_path = exp_out_path / (f"s_mapping_{cfg.exp_suffix}.mp4")
        #         save_video_from_frames(frames_np, video_save_path, fps=10)
        #         print("Save video to %s" % video_save_path)

        # Periodic PCD saving (configurable)
        if cfg.periodically_save_pcd and (counter % cfg.periodically_save_pcd_interval == 0):
            save_pointcloud(
                exp_suffix=cfg.exp_suffix,
                exp_out_path=exp_out_path,
                cfg=cfg,
                objects=objects,
                obj_classes=obj_classes,
                latest_pcd_filepath=cfg.latest_pcd_filepath,
                create_symlink=True,
            )

        # Logging
        owandb.log(
            {
                "frame_idx": frame_idx,
                "counter": counter,
                "exit_early_flag": exit_early_flag,
                "is_final_frame": is_final_frame,
            }
        )
        tracker.increment_total_objects(len(objects))
        tracker.increment_total_detections(len(detection_list))
        owandb.log(
            {
                "total_objects": tracker.get_total_objects(),
                "objects_this_frame": len(objects),
                "total_detections": tracker.get_total_detections(),
                "detections_this_frame": len(detection_list),
                "frame_idx": frame_idx,
                "counter": counter,
                "exit_early_flag": exit_early_flag,
                "is_final_frame": is_final_frame,
            }
        )

    # Consolidate captions only when edges/captions are enabled
    if cfg.make_edges:
        for obj in objects:
            obj_captions = obj["captions"][:20]
            consolidated_caption = consolidate_captions(openai_client, obj_captions)
            obj["consolidated_caption"] = consolidated_caption

    # Final saves
    if cfg.save_pcd:
        save_pointcloud(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            cfg=cfg,
            objects=objects,
            obj_classes=obj_classes,
            latest_pcd_filepath=cfg.latest_pcd_filepath,
            create_symlink=True,
            edges=map_edges,
        )

    if cfg.get("save_semantic_snapshot", False):
        save_pointcloud(
            exp_suffix=cfg.exp_suffix,
            exp_out_path=exp_out_path,
            cfg=cfg,
            objects=objects,
            obj_classes=obj_classes,
            latest_pcd_filepath=None,
            create_symlink=False,
            edges=map_edges,
            include_geometry=False,
            artifact_prefix="semantic",
        )

    if cfg.save_json:
        save_obj_json(exp_suffix=cfg.exp_suffix, exp_out_path=exp_out_path, objects=objects)
        from conceptgraph.utils.general_utils import save_edge_json
        save_edge_json(exp_suffix=cfg.exp_suffix, exp_out_path=exp_out_path, objects=objects, edges=map_edges)

    if cfg.save_objects_all_frames:
        save_meta_path = obj_all_frames_out_path / "meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump(
                {
                    "cfg": cfg,
                    "class_names": obj_classes.get_classes_arr(),
                    "class_colors": obj_classes.get_class_color_dict_by_index(),
                },
                f,
            )
    # REMOVED: for headless batch
    # if run_detections and cfg.save_video:
    #     save_video_detections(det_exp_path)

    owandb.finish()

if __name__ == "__main__":
    main()