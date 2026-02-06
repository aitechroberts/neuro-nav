"""
Build a scene graph using InternVL2-2B Vision-Language Model
This replaces the YOLO+CLIP+LLaVA+GPT-4 pipeline with a modern VLM.

Key changes from build_scenegraph_cfslam.py:
- InternVL2-2B for captioning (replaces LLaVA)
- InternVL2-2B for caption refinement (replaces GPT-4)
- InternVL2-2B for relationship extraction (replaces GPT-4)
- Single unified VLM for all language+vision tasks
"""

import gc
import gzip
import json
import os
import pickle as pkl
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Literal, Union
from textwrap import wrap

from conceptgraph.utils.general_utils import prjson

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import rich
import torch
import tyro
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from tqdm import tqdm, trange
from transformers import logging as hf_logging

torch.autograd.set_grad_enabled(False)
hf_logging.set_verbosity_error()


@dataclass
class ProgramArgs:
    mode: Literal[
        "extract-node-captions",
        "refine-node-captions",
        "build-scenegraph",
        "generate-scenegraph-json",
        "annotate-scenegraph",
    ]

    # Path to cache directory
    cachedir: str = "saved/room0"
    
    prompts_path: str = "prompts/gpt_prompts.json"

    # Path to map file
    mapfile: str = "saved/room0/map/scene_map_cfslam.pkl.gz"

    # Device to use
    device: str = "cuda:0"

    # Voxel size for downsampling
    downsample_voxel_size: float = 0.025

    # Maximum number of detections to consider, per object
    max_detections_per_object: int = 10

    # Suppress objects with less than this number of observations
    min_views_per_object: int = 2

    # List of objects to annotate (default: all objects)
    annot_inds: Union[List[int], None] = None

    # Masking option
    masking_option: Literal["blackout", "red_outline", "none"] = "none"
    
    # VLM model name
    internvl_model: str = "OpenGVLab/InternVL2-2B"


def load_scene_map(args, scene_map):
    """
    Loads a scene map from a gzip-compressed pickle file.
    """
    with gzip.open(Path(args.mapfile), "rb") as f:
        loaded_data = pkl.load(f)
        
        if isinstance(loaded_data, dict) and "objects" in loaded_data:
            scene_map.load_serializable(loaded_data["objects"])
        elif isinstance(loaded_data, list) or isinstance(loaded_data, dict):
            scene_map.load_serializable(loaded_data)
        else:
            raise ValueError("Unexpected data format in map file.")
        print(f"Loaded {len(scene_map)} objects")


def crop_image_pil(image: Image, x1: int, y1: int, x2: int, y2: int, padding: int = 0) -> Image:
    """Crop the image with some padding"""
    image_width, image_height = image.size
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image_width, x2 + padding)
    y2 = min(image_height, y2 + padding)
    image_crop = image.crop((x1, y1, x2, y2))
    return image_crop


def draw_red_outline(image, mask):
    """Draw a red outline around the object in an image"""
    image_np = np.array(image)
    red_outline = [255, 0, 0]
    contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_np, contours, -1, red_outline, 3)
    kernel = np.ones((5, 5), np.uint8)
    image_np = cv2.dilate(image_np, kernel, iterations=1)
    image_pil = Image.fromarray(image_np)
    return image_pil


def crop_image_and_mask(image: Image, mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, padding: int = 0):
    """Crop the image and mask with some padding"""
    image = np.array(image)
    if image.shape[:2] != mask.shape:
        raise ValueError("Initial shape mismatch: Image shape {} != Mask shape {}".format(image.shape, mask.shape))

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.shape[1], x2 + padding)
    y2 = min(image.shape[0], y2 + padding)
    x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)

    image_crop = image[y1:y2, x1:x2]
    mask_crop = mask[y1:y2, x1:x2]

    if image_crop.shape[:2] != mask_crop.shape:
        print("Cropped shape mismatch: Image crop shape {} != Mask crop shape {}".format(image_crop.shape, mask_crop.shape))
        return None, None
    
    image_crop = Image.fromarray(image_crop)
    return image_crop, mask_crop


def blackout_nonmasked_area(image_pil, mask):
    """Blackout the non-masked area of an image"""
    image_np = np.array(image_pil)
    black_image = np.zeros_like(image_np)
    black_image[mask] = image_np[mask]
    black_image = Image.fromarray(black_image)
    return black_image


def plot_images_with_captions(images, captions, confidences, low_confidences, masks, savedir, idx_obj):
    """Debug helper function that plots images with captions and masks"""
    n = min(9, len(images))
    nrows = int(np.ceil(n / 3))
    ncols = 3 if n > 1 else 1
    fig, axarr = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), squeeze=False)

    for i in range(n):
        row, col = divmod(i, 3)
        ax = axarr[row][col]
        ax.imshow(images[i])

        img_array = np.array(images[i])
        if img_array.shape[:2] != masks[i].shape:
            ax.text(0.5, 0.5, "Plotting error: Shape mismatch between image and mask", ha='center', va='center')
        else:
            green_mask = np.zeros((*masks[i].shape, 3), dtype=np.uint8)
            green_mask[masks[i]] = [0, 255, 0]
            ax.imshow(green_mask, alpha=0.15)

        title_text = f"Caption: {captions[i]}\nConfidence: {confidences[i]:.2f}"
        if low_confidences[i]:
            title_text += "\nLow Confidence"
        
        wrapped_title = '\n'.join(wrap(title_text, 30))
        ax.set_title(wrapped_title, fontsize=12)
        ax.axis('off')

    for i in range(n, nrows * ncols):
        row, col = divmod(i, 3)
        axarr[row][col].axis('off')
    
    plt.tight_layout()
    plt.savefig(savedir / f"{idx_obj}.png")
    plt.close()


def extract_node_captions(args):
    """
    Extract captions using InternVL2-2B (replaces LLaVA).
    """
    from conceptgraph.vlm.internvl2_model import InternVL2Model
    from conceptgraph.slam.slam_classes import MapObjectList

    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)

    # Initialize InternVL2
    console = rich.console.Console()
    internvl = InternVL2Model(
        model_name=args.internvl_model,
        device=args.device,
    )
    print("InternVL2 initialized...")

    # Directories to save features and captions
    savedir_feat = Path(args.cachedir) / "cfslam_feat_internvl"
    savedir_feat.mkdir(exist_ok=True, parents=True)
    savedir_captions = Path(args.cachedir) / "cfslam_captions_internvl"
    savedir_captions.mkdir(exist_ok=True, parents=True)
    savedir_debug = Path(args.cachedir) / "cfslam_captions_internvl_debug"
    savedir_debug.mkdir(exist_ok=True, parents=True)

    caption_dict_list = []

    for idx_obj, obj in tqdm(enumerate(scene_map), total=len(scene_map)):
        conf = obj["conf"]
        conf = np.array(conf)
        idx_most_conf = np.argsort(conf)[::-1]

        features = []
        captions = []
        low_confidences = []
        
        image_list = []
        caption_list = []
        confidences_list = []
        low_confidences_list = []
        mask_list = []
        
        if len(idx_most_conf) < 2:
            continue 
        idx_most_conf = idx_most_conf[:args.max_detections_per_object]

        for idx_det in tqdm(idx_most_conf, leave=False):
            image = Image.open(obj["color_path"][idx_det]).convert("RGB")
            xyxy = obj["xyxy"][idx_det]
            mask = obj["mask"][idx_det]
            
            padding = 10
            x1, y1, x2, y2 = xyxy
            image_crop, mask_crop = crop_image_and_mask(image, mask, x1, y1, x2, y2, padding=padding)
            
            if image_crop is None or mask_crop is None:
                continue
            
            # Apply masking option
            if args.masking_option == "blackout":
                image_crop_modified = blackout_nonmasked_area(image_crop, mask_crop)
            elif args.masking_option == "red_outline":
                image_crop_modified = draw_red_outline(image_crop, mask_crop)
            else:
                image_crop_modified = image_crop

            _w, _h = image_crop.size
            if _w * _h < 70 * 70:
                print("small object. Skipping captioning...")
                low_confidences.append(True)
                continue
            else:
                low_confidences.append(False)

            # Generate caption using InternVL2
            try:
                caption = internvl.caption_image(image_crop_modified)
                console.print(f"[bold green]InternVL2:[/bold green] {caption}")
                captions.append(caption)
                
                # Note: InternVL2 embeddings can be added here if needed
                # For now, we'll skip feature extraction since InternVL2 handles everything
                
            except Exception as e:
                print(f"Error captioning image: {e}")
                low_confidences[-1] = True
                continue
        
            # For debug visualization
            conf_value = conf[idx_det]
            image_list.append(image_crop)
            caption_list.append(caption)
            confidences_list.append(conf_value)
            low_confidences_list.append(low_confidences[-1])
            mask_list.append(mask_crop)

        caption_dict_list.append({
            "id": idx_obj,
            "captions": captions,
            "low_confidences": low_confidences,
        })

        # Features not needed with InternVL2 (it handles everything internally)
        
        # Debug visualization
        if len(image_list) > 0:
            plot_images_with_captions(
                image_list, caption_list, confidences_list,
                low_confidences_list, mask_list, savedir_debug, idx_obj
            )

    # Save captions
    with open(Path(args.cachedir) / "cfslam_internvl_captions.json", "w", encoding="utf-8") as f:
        json.dump(caption_dict_list, f, indent=4, sort_keys=False)
    
    # Clean up
    del internvl
    torch.cuda.empty_cache()


def save_json_to_file(json_str, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(json_str, f, indent=4, sort_keys=False)


def refine_node_captions(args):
    """
    Refine captions using InternVL2-2B (replaces GPT-4).
    """
    from conceptgraph.slam.slam_classes import MapObjectList
    from conceptgraph.vlm.internvl2_model import InternVL2Model

    # Load the captions
    caption_file = Path(args.cachedir) / "cfslam_internvl_captions.json"
    with open(caption_file, "r") as f:
        captions = json.load(f)

    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)
    
    # Initialize InternVL2
    internvl = InternVL2Model(
        model_name=args.internvl_model,
        device=args.device,
    )
    print("InternVL2 initialized...")

    responses_savedir = Path(args.cachedir) / "cfslam_internvl_responses"
    responses_savedir.mkdir(exist_ok=True, parents=True)

    responses = []
    unsuccessful_responses = 0

    # Loop over every object
    for _i in trange(len(captions)):
        if len(captions[_i]['captions']) == 0:
            continue
        
        _caption = captions[_i]
        
        # Get a representative image
        obj = scene_map[_i]
        conf = np.array(obj["conf"])
        idx_most_conf = np.argsort(conf)[::-1][0]
        image = Image.open(obj["color_path"][idx_most_conf]).convert("RGB")
        xyxy = obj["xyxy"][idx_most_conf]
        x1, y1, x2, y2 = xyxy
        image_crop = crop_image_pil(image, x1, y1, x2, y2, padding=10)
        
        # Refine captions using InternVL2
        try:
            result = internvl.refine_caption(_caption["captions"])
            
            print(f"\nObject {_i}:")
            print(f"  Raw captions: {_caption['captions'][:2]}")
            print(f"  Refined: {result['object_tag']} - {result['summary']}")
            
            _dict = {
                "id": _caption["id"],
                "captions": _caption["captions"],
                "response": result
            }
            
            responses.append(json.dumps(_dict))
            save_json_to_file(_dict, responses_savedir / f"{_caption['id']}.json")
            
        except Exception as e:
            print(f"Error refining caption for object {_i}: {e}")
            unsuccessful_responses += 1
            _dict = {
                "id": _caption["id"],
                "captions": _caption["captions"],
                "response": {
                    'summary': _caption["captions"][0] if _caption["captions"] else "unknown",
                    'object_tag': 'invalid',
                    'possible_tags': ['invalid']
                }
            }
            responses.append(json.dumps(_dict))
            save_json_to_file(_dict, responses_savedir / f"{_caption['id']}.json")

    print(f"\nUnsuccessful responses: {unsuccessful_responses}")

    # Save all responses
    with open(Path(args.cachedir) / "cfslam_internvl_responses.pkl", "wb") as f:
        pkl.dump(responses, f)
    
    # Clean up
    del internvl
    torch.cuda.empty_cache()


def build_scenegraph(args):
    """Build scene graph with VLM-refined captions"""
    from conceptgraph.slam.slam_classes import MapObjectList
    from conceptgraph.slam.utils import compute_overlap_matrix_general

    # Load the scene map
    scene_map = MapObjectList()
    load_scene_map(args, scene_map)

    response_dir = Path(args.cachedir) / "cfslam_internvl_responses"
    responses = []
    object_tags = []
    also_indices_to_remove = []
    
    for idx in range(len(scene_map)):
        if not (response_dir / f"{idx}.json").exists():
            also_indices_to_remove.append(idx)
            continue
        with open(response_dir / f"{idx}.json", "r") as f:
            _d = json.load(f)
            try:
                if isinstance(_d["response"], str):
                    _d["response"] = json.loads(_d["response"])
            except (json.JSONDecodeError, KeyError):
                _d["response"] = {
                    'summary': f'InternVL2 response failed',
                    'possible_tags': ['invalid'],
                    'object_tag': 'invalid'
                }
            responses.append(_d)
            object_tags.append(_d["response"]["object_tag"])

    # Remove invalid segments
    indices_to_remove = [i for i in range(len(responses)) if object_tags[i].lower() in ["fail", "invalid"]]
    indices_to_remove = set(indices_to_remove)
    
    for obj_idx in range(len(scene_map)):
        conf = scene_map[obj_idx]["conf"]
        if len(conf) < args.min_views_per_object:
            indices_to_remove.add(obj_idx)
    
    # Combine sets properly (also_indices_to_remove is a list)
    indices_to_remove = list(indices_to_remove.union(also_indices_to_remove))
    segment_ids_to_retain = [i for i in range(len(scene_map)) if i not in indices_to_remove]
    
    with open(Path(args.cachedir) / "cfslam_scenegraph_invalid_indices.pkl", "wb") as f:
        pkl.dump(indices_to_remove, f)
    print(f"Removed {len(indices_to_remove)} segments")
    
    responses = [resp for resp in responses if resp['id'] in segment_ids_to_retain]
    object_tags = [resp['response']['object_tag'] for resp in responses]

    pruned_scene_map = []
    pruned_object_tags = []
    for _idx, segmentidx in enumerate(segment_ids_to_retain):
        pruned_scene_map.append(scene_map[segmentidx])
        pruned_object_tags.append(object_tags[_idx])
    
    scene_map = MapObjectList(pruned_scene_map)
    object_tags = pruned_object_tags
    del pruned_scene_map
    gc.collect()
    num_segments = len(scene_map)

    for i in range(num_segments):
        scene_map[i]["caption_dict"] = responses[i]

    # Save pruned scene map
    if not (Path(args.cachedir) / "map").exists():
        (Path(args.cachedir) / "map").mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "wb") as f:
        pkl.dump(scene_map.to_serializable(), f)

    print("Computing bounding box overlaps...")
    bbox_overlaps = compute_overlap_matrix_general(scene_map, downsample_voxel_size=args.downsample_voxel_size)

    # Construct weighted adjacency matrix
    weights = []
    rows = []
    cols = []
    for i in range(num_segments):
        for j in range(i + 1, num_segments):
            if i == j:
                continue
            if bbox_overlaps[i, j] > 0.01:
                weights.append(bbox_overlaps[i, j])
                rows.append(i)
                cols.append(j)
                weights.append(bbox_overlaps[i, j])
                rows.append(j)
                cols.append(i)

    adjacency_matrix = csr_matrix((weights, (rows, cols)), shape=(num_segments, num_segments))
    mst = minimum_spanning_tree(adjacency_matrix)
    _, labels = connected_components(mst)

    components = []
    if len(labels) != 0:
        for label in range(labels.max() + 1):
            indices = np.where(labels == label)[0]
            components.append(indices.tolist())

    with open(Path(args.cachedir) / "cfslam_scenegraph_components.pkl", "wb") as f:
        pkl.dump(components, f)

    # Extract relationships using InternVL2
    minimum_spanning_trees = []
    relations = []
    
    if len(labels) != 0:
        from conceptgraph.vlm.internvl2_model import InternVL2Model
        internvl = InternVL2Model(model_name=args.internvl_model, device=args.device)
        
        for label in range(labels.max() + 1):
            component_indices = np.where(labels == label)[0]
            subgraph = adjacency_matrix[component_indices][:, component_indices]
            _mst = minimum_spanning_tree(subgraph)
            minimum_spanning_trees.append(_mst)

        if not (Path(args.cachedir) / "cfslam_object_relations.json").exists():
            for componentidx, component in enumerate(components):
                if len(component) <= 1:
                    continue
                for u, v in zip(
                    minimum_spanning_trees[componentidx].nonzero()[0],
                    minimum_spanning_trees[componentidx].nonzero()[1]
                ):
                    segmentidx1 = component[u]
                    segmentidx2 = component[v]
                    _bbox1 = scene_map[segmentidx1]["bbox"]
                    _bbox2 = scene_map[segmentidx2]["bbox"]

                    object1_info = {
                        "id": segmentidx1,
                        "bbox_extent": np.round(_bbox1.extent, 1).tolist(),
                        "bbox_center": np.round(_bbox1.center, 1).tolist(),
                        "object_tag": object_tags[segmentidx1],
                    }
                    object2_info = {
                        "id": segmentidx2,
                        "bbox_extent": np.round(_bbox2.extent, 1).tolist(),
                        "bbox_center": np.round(_bbox2.center, 1).tolist(),
                        "object_tag": object_tags[segmentidx2],
                    }

                    print(f"{object1_info['object_tag']} <-> {object2_info['object_tag']}")

                    try:
                        result = internvl.extract_object_relationships(
                            obj1_info=object1_info,
                            obj2_info=object2_info,
                        )
                        
                        output_dict = {
                            "object1": object1_info,
                            "object2": object2_info,
                            "object_relation": result["object_relation"],
                            "reason": result["reason"]
                        }
                        relations.append(output_dict)
                        
                    except Exception as e:
                        print(f"Error extracting relationship: {e}")
                        output_dict = {
                            "object1": object1_info,
                            "object2": object2_info,
                            "object_relation": "FAIL",
                            "reason": "FAIL"
                        }
                        relations.append(output_dict)

            with open(Path(args.cachedir) / "cfslam_object_relations.json", "w") as f:
                json.dump(relations, f, indent=4)
        else:
            relations = json.load(open(Path(args.cachedir) / "cfslam_object_relations.json", "r"))
        
        del internvl
        torch.cuda.empty_cache()

    scenegraph_edges = []
    _idx = 0
    for componentidx, component in enumerate(components):
        if len(component) <= 1:
            continue
        for u, v in zip(
            minimum_spanning_trees[componentidx].nonzero()[0],
            minimum_spanning_trees[componentidx].nonzero()[1]
        ):
            segmentidx1 = component[u]
            segmentidx2 = component[v]
            if relations[_idx]["object_relation"] != "none of these":
                scenegraph_edges.append((segmentidx1, segmentidx2, relations[_idx]["object_relation"]))
            _idx += 1
    
    print(f"Created 3D scenegraph with {num_segments} nodes and {len(scenegraph_edges)} edges")

    with open(Path(args.cachedir) / "cfslam_scenegraph_edges.pkl", "wb") as f:
        pkl.dump(scenegraph_edges, f)


def generate_scenegraph_json(args):
    """Generate JSON summary of scene graph"""
    from conceptgraph.slam.slam_classes import MapObjectList

    scene_desc = []
    print("Generating scene graph JSON file...")

    scene_map = MapObjectList()
    with gzip.open(Path(args.cachedir) / "map" / "scene_map_cfslam_pruned.pkl.gz", "rb") as f:
        scene_map.load_serializable(pkl.load(f))
    print(f"Loaded scene map with {len(scene_map)} objects")

    for i, segment in enumerate(scene_map):
        _d = {
            "id": segment["caption_dict"]["id"],
            "bbox_extent": np.round(segment['bbox'].extent, 1).tolist(),
            "bbox_center": np.round(segment['bbox'].center, 1).tolist(),
            "possible_tags": segment["caption_dict"]["response"]["possible_tags"],
            "object_tag": segment["caption_dict"]["response"]["object_tag"],
            "caption": segment["caption_dict"]["response"]["summary"],
        }
        scene_desc.append(_d)
    
    with open(Path(args.cachedir) / "scene_graph.json", "w") as f:
        json.dump(scene_desc, f, indent=4)


def main():
    args = tyro.cli(ProgramArgs)
    print(f"args.masking_option: {args.masking_option}")
    print(f"InternVL model: {args.internvl_model}")

    if args.mode == "extract-node-captions":
        extract_node_captions(args)
    elif args.mode == "refine-node-captions":
        refine_node_captions(args)
    elif args.mode == "build-scenegraph":
        build_scenegraph(args)
    elif args.mode == "generate-scenegraph-json":
        generate_scenegraph_json(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

