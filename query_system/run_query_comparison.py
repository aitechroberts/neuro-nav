#!/usr/bin/env python3
"""
Query System Comparison Runner

Compares all 4 query setups on the same scene with the same queries.

Usage:
    python query_system/run_query_comparison.py
    python query_system/run_query_comparison.py --scene path/to/scene.pkl.gz
    python query_system/run_query_comparison.py --query "Find me something to sit on"
    python query_system/run_query_comparison.py --interactive
"""

import argparse
import gzip
import pickle
import sys
import os
from pathlib import Path
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from query_system.setup1_gpt4_query import GPT4QueryEngine
from query_system.setup2_local_vlm_query import LocalVLMQueryEngine
from query_system.setup3_clip_query import CLIPQueryEngine
from query_system.setup4_yolo_query import YOLOQueryEngine
from query_system.setup5_gpt4_vision import GPT4VisionQueryEngine


def load_scene(scene_path: str) -> List[Dict]:
    """
    Load objects from a saved ConceptGraphs scene.

    Args:
        scene_path: Path to .pkl.gz file

    Returns:
        List of object dictionaries
    """
    print(f"\n📦 Loading scene from: {scene_path}")

    # Resolve symlinks to get actual file path
    scene_path = str(Path(scene_path).resolve())
    print(f"   Resolved to: {scene_path}")

    if scene_path.endswith('.pkl.gz'):
        with gzip.open(scene_path, 'rb') as f:
            data = pickle.load(f)
    elif scene_path.endswith('.pkl'):
        with open(scene_path, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError(f"Unsupported file format: {scene_path}")

    # Handle both formats: dict with 'objects' key or direct list
    if isinstance(data, dict) and 'objects' in data:
        objects = data['objects']
    elif isinstance(data, list):
        objects = data
    else:
        raise ValueError(f"Unexpected data format: {type(data)}")

    print(f"   ✓ Loaded {len(objects)} objects")

    # Print scene summary
    class_names = [obj.get('class_name', 'unknown') for obj in objects]
    from collections import Counter
    class_counts = Counter(class_names)

    print(f"\n📊 Scene Summary:")
    for class_name, count in class_counts.most_common(10):
        print(f"   {class_name}: {count}")

    return objects


def initialize_engines(objects: List[Dict], setups: List[str]) -> Dict:
    """
    Initialize requested query engines.

    Args:
        objects: List of scene objects
        setups: List of setup names to initialize (e.g., ['1', '3', '4'])

    Returns:
        Dictionary mapping setup name to engine instance
    """
    print(f"\n🤖 Initializing query engines...")

    engines = {}

    if '1' in setups:
        print(f"\n--- Setup 1: GPT-4V ---")
        engines['setup1'] = GPT4QueryEngine(objects)

    if '2' in setups:
        print(f"\n--- Setup 2: LLaVA (Local VLM) ---")
        engines['setup2'] = LocalVLMQueryEngine(objects)

    if '3' in setups:
        print(f"\n--- Setup 3: CLIP ---")
        engines['setup3'] = CLIPQueryEngine(objects)

    if '4' in setups:
        print(f"\n--- Setup 4: YOLO Labels ---")
        engines['setup4'] = YOLOQueryEngine(objects)

    if '5' in setups:
        print(f"\n--- Setup 5: GPT-4V with Vision ---")
        engines['setup5'] = GPT4VisionQueryEngine(objects)

    print(f"\n✅ Initialized {len(engines)} engines")
    return engines


def run_query(engines: Dict, query: str):
    """
    Run a query on all engines and display results.

    Args:
        engines: Dictionary of initialized engines
        query: Natural language query string
    """
    print(f"\n" + "=" * 80)
    print(f"🔍 Query: \"{query}\"")
    print("=" * 80)

    results = {}

    for name, engine in engines.items():
        print(f"\n--- {engine.setup_name} ---")
        try:
            result = engine.answer(query)
            results[name] = result

            # Display result
            print(f"✓ Match: {result.class_name} (ID: {result.object_id})")
            print(f"  Confidence: {result.confidence:.3f}")
            if result.reasoning:
                print(f"  Reasoning: {result.reasoning}")

            # Show top-3 matches if available
            if result.matched_objects and len(result.matched_objects) > 1:
                print(f"  Top-3 matches:")
                for i, match in enumerate(result.matched_objects[:3], 1):
                    if isinstance(match, dict):
                        label = match.get('label') or match.get('class_name', 'unknown')
                        sim = match.get('similarity', 0.0)
                        print(f"    {i}. {label} ({sim:.3f})")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary comparison
    print(f"\n" + "=" * 80)
    print(f"📊 COMPARISON SUMMARY")
    print("=" * 80)

    # Check agreement
    matched_objects = [r.class_name for r in results.values()]
    if len(set(matched_objects)) == 1:
        print(f"✓ All setups AGREE: {matched_objects[0]}")
    else:
        print(f"⚠ Setups DISAGREE:")
        for name, result in results.items():
            print(f"  {name}: {result.class_name} ({result.confidence:.3f})")

    # Confidence ranking
    print(f"\nConfidence Ranking:")
    sorted_results = sorted(results.items(), key=lambda x: x[1].confidence, reverse=True)
    for i, (name, result) in enumerate(sorted_results, 1):
        print(f"  {i}. {name}: {result.confidence:.3f} - {result.class_name}")


def interactive_mode(engines: Dict):
    """
    Interactive query mode - keep asking for queries.

    Args:
        engines: Dictionary of initialized engines
    """
    print(f"\n" + "=" * 80)
    print(f"🎮 INTERACTIVE MODE")
    print("=" * 80)
    print(f"Enter queries to test all setups. Type 'quit' or 'exit' to stop.")
    print(f"\nExample queries:")
    print(f"  - Find me something to sit on")
    print(f"  - Where is the chair?")
    print(f"  - Find me something red")
    print(f"  - What can I use to type?")
    print("=" * 80)

    while True:
        try:
            query = input("\n🔍 Query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            if not query:
                continue

            run_query(engines, query)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")


def batch_mode(engines: Dict, queries: List[str]):
    """
    Run a batch of predefined queries.

    Args:
        engines: Dictionary of initialized engines
        queries: List of query strings
    """
    print(f"\n" + "=" * 80)
    print(f"📋 BATCH MODE - Running {len(queries)} queries")
    print("=" * 80)

    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}/{len(queries)}]")
        run_query(engines, query)


def main():
    parser = argparse.ArgumentParser(
        description="Compare query systems on ConceptGraphs scenes"
    )

    parser.add_argument(
        '--scene',
        type=str,
        default='latest_pcd_save',
        help='Path to scene .pkl.gz file (default: latest_pcd_save symlink)'
    )

    parser.add_argument(
        '--query',
        type=str,
        help='Single query to run (if not specified, enters interactive mode)'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Force interactive mode'
    )

    parser.add_argument(
        '--setups',
        type=str,
        default='1,2,3,4',
        help='Comma-separated list of setups to run (1-5, default: 1,2,3,4)'
    )

    parser.add_argument(
        '--batch',
        action='store_true',
        help='Run predefined batch of test queries'
    )

    args = parser.parse_args()

    # Load scene
    try:
        objects = load_scene(args.scene)
    except Exception as e:
        print(f"\n✗ Failed to load scene: {e}")
        print(f"\nTip: Run an exploration first to generate a scene file:")
        print(f"  ./run_exploration1.sh")
        print(f"  ./run_exploration2.sh")
        return 1

    # Parse setups
    setups = [s.strip() for s in args.setups.split(',')]

    # Initialize engines
    try:
        engines = initialize_engines(objects, setups)
    except Exception as e:
        print(f"\n✗ Failed to initialize engines: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run queries
    if args.batch:
        # Predefined test queries
        test_queries = [
            "Find me a chair",
            "Find me something to sit on",
            "Where is the table?",
            "Find me something red",
            "What can I use to type?",
        ]
        batch_mode(engines, test_queries)

    elif args.query:
        # Single query mode
        run_query(engines, args.query)

    else:
        # Interactive mode (default)
        interactive_mode(engines)

    return 0


if __name__ == '__main__':
    sys.exit(main())
