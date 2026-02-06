'''
Batch VLM mapping script using Google Gemma 3 for fully local inference.
Replaces PaliGemma with Gemma 3 (4B-IT) for improved instruction following.
'''

# Standard library imports
import os
import gzip
import pickle
from pathlib import Path
from collections import Counter
from typing import List, Optional, Tuple

# Third-party imports
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import trange
import hydra
from omegaconf import DictConfig
import open_clip
from ultralytics import YOLO, SAM
import supervision as sv

# Local application/library specific imports
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.logging_metrics import MappingTracker
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.utils.general_utils import (
    ObjectClasses,
    get_det_out_path,
    load_saved_detections,
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

# =============================================================================
# Gemma 3 Imports
# =============================================================================
from conceptgraph.utils.vlm_gemma import (
    Gemma3Client,
    get_gemma3_client,
    consolidate_captions,
)

# Disable torch gradient computation
torch.set_grad_enabled(False)


# =============================================================================
# VLM Edge/Caption Generation using Gemma 3
# =============================================================================

def make_vlm_edges_and_captions_gemma(
    image: np.ndarray,
    detections: sv.Detections,
    obj_classes,
    detection_class_labels: List[str],
    vlm_client: Gemma3Client,
    cfg: DictConfig,
    make_edges: bool = True,
):
    """
    Annotates the frame with bounding boxes and IDs, then calls Gemma 3 
    to generate captions and (optionally) edges.
    """
    # 1. Setup Annotators
    # Visual Prompting (Set-of-Mark)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=5,
        text_position=sv.Position.CENTER, 
        color=sv.ColorPalette.DEFAULT
    )

    # 2. Create Numeric Labels
    numeric_labels = [str(i) for i in range(len(detections.xyxy))]

    # 3. Annotate the Image
    annotated_frame = image.copy()
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=numeric_labels)

    # 4. Convert to PIL (RGB)
    pil_annotated_image = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

    # 5. Generate Captions
    captions = vlm_client.caption_objects_with_labels(
        image=pil_annotated_image,
        labels=detection_class_labels,
        caption_system_prompt=str(cfg.caption),
        captions_with_labels_template=str(cfg.captions_with_labels),
    )

    # 6. Generate Relations (Edges)
    edges: List[Tuple[str, str, str]] = []
    if make_edges:
        edges = vlm_client.infer_relations_with_labels(
            image=pil_annotated_image,
            labels=detection_class_labels,
            relation_system_prompt=str(cfg.relation),
            relations_with_labels_template=str(cfg.relations_with_labels),
        )

    return detection_class_labels, edges, annotated_frame, captions


# =============================================================================
# Path Resolution Helpers
# =============================================================================

def _resolve_output_base(cfg: DictConfig) -> Path:
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


# =============================================================================
# Main Entry Point
# =============================================================================

@hydra.main(version_base=None, config_path="../hydra_configs/", config_name="batch_vlm_mapping_gemma")
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

    output_base = _resolve_output_base(cfg)
    exp_out_path = _build_exp_path(output_base, cfg.scene_id, cfg.exp_suffix, create=True)
    det_exp_path = _build_exp_path(output_base, cfg.scene_id, cfg.detections_exp_suffix, create=False)

    detections_exp_cfg = cfg_to_dict(cfg)
    obj_classes = ObjectClasses(
        classes_file_path=detections_exp_cfg["classes_file"],
        bg_classes=detections_exp_cfg["bg_classes"],
        skip_bg=detections_exp_cfg["skip_bg"],
    )

    run_detections = check_run_detections(cfg.force_detection, det_exp_path)
    det_exp_pkl_path = get_det_out_path(det_exp_path)

    if run_detections:
        print("\n".join(["Running detections with Gemma 3..."] * 3))
        det_exp_path.mkdir(parents=True, exist_ok=True)

        ckpt_dir = os.environ.get("CKPT_DIR", "")
        
        yolo_weights = "yolov8l-worldv2.pt"
        if ckpt_dir and (Path(ckpt_dir) / yolo_weights).exists():
            yolo_weights = str(Path(ckpt_dir) / yolo_weights)
            print(f"Using local YOLO weights: {yolo_weights}")
        detection_model = measure_time(YOLO)(yolo_weights)

        sam_weights = "sam2.1_b.pt"
        if ckpt_dir and (Path(ckpt_dir) / sam_weights).exists():
            sam_weights = str(Path(ckpt_dir) / sam_weights)
            print(f"Using local SAM weights: {sam_weights}")
        sam_predictor = SAM(sam_weights)

    model_name = "hf-hub:timm/PE-Core-B-16"
    if ckpt_dir and os.path.exists(ckpt_dir):
        hf_cache_dir = os.path.join(ckpt_dir, "huggingface")
        os.makedirs(hf_cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = hf_cache_dir
        cache_dir = hf_cache_dir
    else:
        cache_dir = os.environ.get("HF_HOME")

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        model_name,
        cache_dir=cache_dir,
    )
    clip_model = clip_model.to(cfg.device)
    clip_tokenizer = open_clip.get_tokenizer(model_name)

    if run_detections:
        detection_model.set_classes(obj_classes.get_classes_arr())

    # =========================================================================
    # GEMMA 3 CLIENT INITIALIZATION
    # =========================================================================
    gemma_client = None
    if cfg.make_edges:
        gemma_model_name = cfg.get("gemma_model", "google/gemma-3-4b-it")
        prompts = cfg.get("prompts", None)
        
        gemma_client = get_gemma3_client(
            model_name=gemma_model_name,
            device=cfg.device,
            prompts=prompts,
            force_new=True,
        )
    # =========================================================================

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

        if not exit_early_flag and should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True
        if exit_early_flag and frame_idx < len(dataset) - 1:
            continue

        color_path = Path(dataset.color_paths[frame_idx])
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]

        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy()
        image_rgb = color_np.astype(np.uint8)

        raw_gobs = None
        gobs = None

        if run_detections:
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

            if xyxy_tensor.numel() != 0:
                sam_out = sam_predictor.predict(color_path, bboxes=xyxy_tensor, verbose=False)
                masks_tensor = sam_out[0].masks.data
                masks_np = masks_tensor.detach().cpu().numpy()
                if masks_np.dtype != np.bool_:
                    masks_np = masks_np > 0.5

                n_boxes = xyxy_np.shape[0]
                n_masks = masks_np.shape[0]
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

            # =========================================================================
            # GEMMA VLM CALLS
            # =========================================================================
            if gemma_client is not None:
                labels, edges, edge_image, captions = make_vlm_edges_and_captions_gemma(
                    image,
                    curr_det,
                    obj_classes,
                    detection_class_labels,
                    gemma_client,
                    cfg,
                    cfg.make_edges,
                )
            else:
                labels = detection_class_labels
                edges = []
                edge_image = None
                captions = [""] * len(curr_det.xyxy)
            # =========================================================================

            image_crops, image_feats, text_feats = compute_clip_features_batched(
                image_rgb_bgr_fixed, curr_det, clip_model, clip_preprocess, 
                clip_tokenizer, obj_classes.get_classes_arr(), cfg.device
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
                save_detection_results(det_exp_pkl_path / color_path.stem, raw_gobs)
        else:
            if os.path.exists(det_exp_pkl_path / color_path.stem):
                raw_gobs = load_saved_detections(det_exp_pkl_path / color_path.stem)
            elif os.path.exists(det_exp_pkl_path / f"{int(color_path.stem):06}"):
                raw_gobs = load_saved_detections(det_exp_pkl_path / f"{int(color_path.stem):06}")
            else:
                raise FileNotFoundError(
                    f"No detections found for frame {frame_idx} at paths {det_exp_pkl_path / color_path.stem}"
                )

        unt_pose = dataset.poses[frame_idx]
        unt_pose = unt_pose.cpu().numpy()
        adjusted_pose = unt_pose

        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        filtered_gobs = filter_gobs(
            resized_gobs, image_rgb, skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )
        gobs = filtered_gobs
        if len(gobs["mask"]) == 0:
            continue

        gobs["mask"] = mask_subtract_contained(gobs["xyxy"], gobs["mask"])

        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth_array, masks=gobs["mask"], cam_K=intrinsics.cpu().numpy()[:3, :3],
            image_rgb=image_rgb, trans_pose=adjusted_pose,
            min_points_threshold=cfg.min_points_threshold, spatial_sim_type=cfg.spatial_sim_type,
            obj_pcd_max_points=cfg.obj_pcd_max_points, device=cfg.device,
        )

        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"], downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"], dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                )
                obj["bbox"] = get_bounding_box(spatial_sim_type=cfg["spatial_sim_type"], pcd=obj["pcd"])

        detection_list = make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx
        )
        if len(detection_list) == 0:
            continue

        if len(objects) == 0:
            objects.extend(detection_list)
            tracker.increment_total_objects(len(detection_list))
            owandb.log({"total_objects_so_far": tracker.get_total_objects(), "objects_this_frame": len(detection_list)})
            continue

        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=cfg["spatial_sim_type"], detection_list=detection_list,
            objects=objects, downsample_voxel_size=cfg["downsample_voxel_size"],
        )
        visual_sim = compute_visual_similarities(detection_list, objects)
        agg_sim = aggregate_similarities(
            match_method=cfg["match_method"], phys_bias=cfg["phys_bias"],
            spatial_sim=spatial_sim, visual_sim=visual_sim,
        )
        match_indices = match_detections_to_objects(agg_sim=agg_sim, detection_threshold=cfg["sim_threshold"])
        objects = merge_obj_matches(
            detection_list=detection_list, objects=objects, match_indices=match_indices,
            downsample_voxel_size=cfg["downsample_voxel_size"], dbscan_remove_noise=cfg["dbscan_remove_noise"],
            dbscan_eps=cfg["dbscan_eps"], dbscan_min_points=cfg["dbscan_min_points"],
            spatial_sim_type=cfg["spatial_sim_type"], device=cfg["device"],
        )

        for obj in objects:
            curr_obj_class_id_counter = Counter(obj["class_id"])
            most_common_class_id = curr_obj_class_id_counter.most_common(1)[0][0]
            most_common_class_name = obj_classes.get_classes_arr()[most_common_class_id]
            if obj["class_name"] != most_common_class_name:
                obj["class_name"] = most_common_class_name

        map_edges = process_edges(match_indices, gobs, len(objects), objects, map_edges, frame_idx)
        is_final_frame = frame_idx == len(dataset) - 1

        edges_to_delete = []
        for curr_map_edge in map_edges.edges_by_index.values():
            if (frame_idx - curr_map_edge.first_detected > 5) and curr_map_edge.num_detections < 2:
                edges_to_delete.append((curr_map_edge.obj1_idx, curr_map_edge.obj2_idx))
        for edge in edges_to_delete:
            map_edges.delete_edge(edge[0], edge[1])

        if processing_needed(cfg["denoise_interval"], cfg["run_denoise_final_frame"], frame_idx, is_final_frame):
            objects = measure_time(denoise_objects)(
                downsample_voxel_size=cfg["downsample_voxel_size"], dbscan_remove_noise=cfg["dbscan_remove_noise"],
                dbscan_eps=cfg["dbscan_eps"], dbscan_min_points=cfg["dbscan_min_points"],
                spatial_sim_type=cfg["spatial_sim_type"], device=cfg["device"], objects=objects,
            )

        if processing_needed(cfg["filter_interval"], cfg["run_filter_final_frame"], frame_idx, is_final_frame):
            objects = filter_objects(
                obj_min_points=cfg["obj_min_points"], obj_min_detections=cfg["obj_min_detections"],
                objects=objects, map_edges=map_edges,
            )

        if processing_needed(cfg["merge_interval"], cfg["run_merge_final_frame"], frame_idx, is_final_frame):
            do_edges = cfg.get("make_edges", False)
            if do_edges:
                objects, map_edges = measure_time(merge_objects)(
                    merge_overlap_thresh=cfg["merge_overlap_thresh"], merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                    merge_text_sim_thresh=cfg["merge_text_sim_thresh"], objects=objects,
                    downsample_voxel_size=cfg["downsample_voxel_size"], dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"], dbscan_min_points=cfg["dbscan_min_points"],
                    spatial_sim_type=cfg["spatial_sim_type"], device=cfg["device"], do_edges=True, map_edges=map_edges,
                )
            else:
                objects = measure_time(merge_objects)(
                    merge_overlap_thresh=cfg["merge_overlap_thresh"], merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                    merge_text_sim_thresh=cfg["merge_text_sim_thresh"], objects=objects,
                    downsample_voxel_size=cfg["downsample_voxel_size"], dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"], dbscan_min_points=cfg["dbscan_min_points"],
                    spatial_sim_type=cfg["spatial_sim_type"], device=cfg["device"], do_edges=False, map_edges=None,
                )

        if cfg.save_objects_all_frames:
            from conceptgraph.utils.general_utils import save_objects_for_frame
            save_objects_for_frame(obj_all_frames_out_path, frame_idx, objects, cfg.obj_min_detections, adjusted_pose, color_path)

        if cfg.periodically_save_pcd and (counter % cfg.periodically_save_pcd_interval == 0):
            save_pointcloud(cfg.exp_suffix, exp_out_path, cfg, objects, obj_classes, cfg.latest_pcd_filepath, create_symlink=True)

        owandb.log({"frame_idx": frame_idx, "counter": counter, "total_objects": tracker.get_total_objects(), "objects_this_frame": len(objects)})

    # =========================================================================
    # CAPTION CONSOLIDATION
    # =========================================================================
    if cfg.make_edges and gemma_client is not None:
        print("[Gemma 3] Consolidating captions...")
        for obj in objects:
            obj_captions = obj.get("captions", [])[:20]
            if obj_captions:
                consolidated_caption = consolidate_captions(gemma_client, obj_captions)
                obj["consolidated_caption"] = consolidated_caption
            else:
                obj["consolidated_caption"] = obj.get("class_name", "unknown object")

    if cfg.save_pcd:
        save_pointcloud(cfg.exp_suffix, exp_out_path, cfg, objects, obj_classes, cfg.latest_pcd_filepath, create_symlink=True, edges=map_edges)

    if cfg.save_json:
        save_obj_json(cfg.exp_suffix, exp_out_path, objects)
        from conceptgraph.utils.general_utils import save_edge_json
        save_edge_json(cfg.exp_suffix, exp_out_path, objects, map_edges)

    if cfg.save_objects_all_frames:
        save_meta_path = obj_all_frames_out_path / "meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump({
                "cfg": cfg, "class_names": obj_classes.get_classes_arr(),
                "class_colors": obj_classes.get_class_color_dict_by_index(),
            }, f)

    if gemma_client is not None:
        gemma_client.cleanup()

    owandb.finish()

if __name__ == "__main__":
    main()