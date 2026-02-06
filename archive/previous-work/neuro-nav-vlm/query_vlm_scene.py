#!/usr/bin/env python3
"""
Query VLM Scene - Example script for querying the scene graph using Qwen2-VL

This demonstrates how to use the VLM-based scene graph for robot navigation tasks.
"""

import json
import sys
from pathlib import Path
from PIL import Image
import argparse

# Add conceptgraph to path
sys.path.insert(0, str(Path(__file__).parent))

from conceptgraph.vlm.qwen2vl_model import Qwen2VLModel


def load_scene_graph(scene_graph_path):
    """Load the scene graph JSON file"""
    with open(scene_graph_path, 'r') as f:
        scene_graph = json.load(f)
    return scene_graph


def build_scene_context(scene_graph, max_objects=20, include_full_captions=True):
    """Build a textual context from the scene graph"""
    context_parts = [f"The scene contains {len(scene_graph)} objects:"]
    
    for i, obj in enumerate(scene_graph[:max_objects]):
        obj_tag = obj.get('object_tag', 'unknown')
        caption = obj.get('caption', '')
        bbox_center = obj.get('bbox_center', [0, 0, 0])
        
        # Extract first sentence for a concise description
        if caption:
            # Get first 1-2 sentences (up to first period + space)
            first_sentence = caption.split('. ')[0] + '.'
            # If still too long, truncate to ~150 chars
            if len(first_sentence) > 200:
                first_sentence = first_sentence[:197] + '...'
            
            context_parts.append(
                f"- Object {obj['id']} at ({bbox_center[0]:.1f}, {bbox_center[1]:.1f}, {bbox_center[2]:.1f}): {first_sentence}"
            )
            
            # Optionally include full caption for more detail
            if include_full_captions and len(caption) > len(first_sentence):
                # Add a few more sentences if available
                sentences = caption.split('. ')[:3]  # First 3 sentences
                extended_desc = '. '.join(sentences)
                if not extended_desc.endswith('.'):
                    extended_desc += '.'
                context_parts.append(f"  Full description: {extended_desc}")
        else:
            context_parts.append(
                f"- Object {obj['id']}: {obj_tag} at position ({bbox_center[0]:.1f}, {bbox_center[1]:.1f}, {bbox_center[2]:.1f})"
            )
    
    if len(scene_graph) > max_objects:
        context_parts.append(f"... and {len(scene_graph) - max_objects} more objects")
    
    return "\n".join(context_parts)


def query_scene_interactive(qwen_model, scene_graph, image_path=None):
    """Interactive query session"""
    print("\n" + "="*70)
    print("VLM Scene Query Interface")
    print("="*70)
    
    # Build context
    context = build_scene_context(scene_graph)
    print(f"\n{context}\n")
    
    # Load image if provided
    image = None
    if image_path and Path(image_path).exists():
        image = Image.open(image_path).convert('RGB')
        print(f"Loaded image: {image_path}\n")
    else:
        print("No image provided - using text-only mode\n")
    
    print("Examples:")
    print("  - Where is the chair?")
    print("  - What objects are on the table?")
    print("  - Find the laptop")
    print("  - What is in the center of the room?")
    print("  - Describe the scene")
    print("\nType 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            query = input("Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print("\nThinking...")
            answer = qwen_model.query_scene(image, query, context=context)
            print(f"\n{answer}\n")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def query_scene_single(qwen_model, scene_graph, query, image_path=None):
    """Single query mode"""
    context = build_scene_context(scene_graph)
    
    image = None
    if image_path and Path(image_path).exists():
        image = Image.open(image_path).convert('RGB')
    
    answer = qwen_model.query_scene(image, query, context=context)
    return answer


def main():
    parser = argparse.ArgumentParser(description="Query VLM-based scene graph")
    parser.add_argument(
        '--scene-graph',
        type=str,
        help='Path to scene_graph.json file'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to scene image (optional)'
    )
    parser.add_argument(
        '--query',
        type=str,
        help='Single query to execute (non-interactive mode)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen2-VL-2B-Instruct',
        help='Qwen2-VL model to use'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device to run on'
    )
    
    args = parser.parse_args()
    
    # Find scene graph if not provided
    if not args.scene_graph:
        # Try to find latest scene graph
        possible_paths = list(Path('data/outputs').rglob('scene_graph.json'))
        if possible_paths:
            args.scene_graph = str(sorted(possible_paths)[-1])
            print(f"Using scene graph: {args.scene_graph}")
        else:
            print("Error: No scene graph found. Please provide --scene-graph")
            print("\nRun the VLM pipeline first:")
            print("  ./run_vlm_pipeline.sh")
            return 1
    
    # Load scene graph
    print(f"Loading scene graph from: {args.scene_graph}")
    scene_graph = load_scene_graph(args.scene_graph)
    print(f"Loaded {len(scene_graph)} objects")
    
    # Initialize Qwen2-VL
    print(f"\nInitializing Qwen2-VL model: {args.model}")
    qwen = Qwen2VLModel(model_name=args.model, device=args.device)
    print("Model loaded successfully!")
    
    # Query mode
    if args.query:
        # Single query mode
        print(f"\nQuery: {args.query}")
        answer = query_scene_single(qwen, scene_graph, args.query, args.image)
        print(f"\nAnswer:\n{answer}\n")
    else:
        # Interactive mode
        query_scene_interactive(qwen, scene_graph, args.image)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

