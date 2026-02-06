"""
Data loading utilities for neuro-nav distillation training
"""

import os
import sys
import json
import logging
from typing import List, Dict, Optional, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from conceptgraph.dataset.datasets_common import get_dataset, load_dataset_config
except ImportError:
    get_dataset = None
    load_dataset_config = None
    logging.warning("Could not import dataset utilities. Some features may not work.")

logger = logging.getLogger(__name__)


class NeuroNavDistillationDataset(Dataset):
    """
    Dataset for loading neuro-nav scenes for distillation training.
    Loads images and associated object captions from neuro-nav outputs.
    """
    
    def __init__(
        self,
        data_root: str,
        scene_ids: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        task: str = "caption_refinement",
        max_samples: Optional[int] = None,
    ):
        """
        Initialize distillation dataset.
        
        Args:
            data_root: Root directory containing neuro-nav data
            scene_ids: List of scene IDs to use (e.g., ['room0', 'room1'])
            output_dir: Directory containing neuro-nav outputs (scene graphs, captions)
            task: Task type ('caption_refinement', 'relationships', 'querying')
            max_samples: Maximum number of samples to load
        """
        self.data_root = data_root
        self.task = task
        self.max_samples = max_samples
        
        # Find output directories
        if output_dir is None:
            # Look for outputs in common locations
            possible_output_dirs = [
                os.path.join(data_root, 'outputs'),
                os.path.join(os.path.dirname(data_root), 'outputs'),
                os.path.join(data_root, '..', 'outputs'),
            ]
            output_dir = None
            for possible_dir in possible_output_dirs:
                if os.path.exists(possible_dir):
                    output_dir = possible_dir
                    break
        
        if output_dir is None:
            logger.warning("Could not find output directory. Will try to load from data_root directly.")
            output_dir = data_root
        
        self.output_dir = output_dir
        
        # Load scene data
        self.samples = self._load_scene_data(scene_ids)
        
        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
        
        logger.info(f"Loaded {len(self.samples)} samples for task: {task}")
    
    def _load_scene_data(self, scene_ids: Optional[List[str]]) -> List[Dict]:
        """Load scene data from neuro-nav outputs."""
        samples = []
        
        # Try to load from JSON outputs (scene graphs)
        if scene_ids is None:
            # Auto-discover scenes
            scene_dirs = []
            if os.path.exists(self.output_dir):
                for item in os.listdir(self.output_dir):
                    item_path = os.path.join(self.output_dir, item)
                    if os.path.isdir(item_path):
                        # Check for timestamp directories
                        for subitem in os.listdir(item_path):
                            subitem_path = os.path.join(item_path, subitem)
                            if os.path.isdir(subitem_path):
                                scene_dirs.append(subitem_path)
        else:
            # Use specified scene IDs
            scene_dirs = []
            for scene_id in scene_ids:
                # Look for scene in outputs
                for root, dirs, files in os.walk(self.output_dir):
                    if scene_id in root:
                        scene_dirs.append(root)
        
        # Load samples from scene graphs
        for scene_dir in scene_dirs:
            scene_graph_path = os.path.join(scene_dir, 'scene_graph.json')
            if os.path.exists(scene_graph_path):
                try:
                    with open(scene_graph_path, 'r') as f:
                        scene_graph = json.load(f)
                    
                    # Extract object data
                    if 'objects' in scene_graph:
                        for obj in scene_graph['objects']:
                            # Get image path if available
                            image_path = None
                            if 'image_path' in obj:
                                image_path = obj['image_path']
                            elif 'representative_image' in obj:
                                image_path = obj['representative_image']
                            
                            # Get captions
                            captions = []
                            if 'captions' in obj:
                                captions = obj['captions']
                            elif 'caption' in obj:
                                captions = [obj['caption']]
                            elif 'description' in obj:
                                captions = [obj['description']]
                            
                            if image_path and captions:
                                # Resolve image path
                                if not os.path.isabs(image_path):
                                    image_path = os.path.join(scene_dir, image_path)
                                
                                if os.path.exists(image_path):
                                    samples.append({
                                        'image_path': image_path,
                                        'captions': captions,
                                        'object_id': obj.get('id', len(samples)),
                                        'scene_dir': scene_dir,
                                    })
                except Exception as e:
                    logger.warning(f"Error loading scene graph from {scene_graph_path}: {e}")
        
        # Also try loading from raw Replica data if available
        replica_dir = os.path.join(self.data_root, 'Replica')
        if os.path.exists(replica_dir) and len(samples) == 0:
            logger.info("Loading from Replica dataset directly...")
            samples = self._load_from_replica(replica_dir, scene_ids)
        
        return samples
    
    def _load_from_replica(self, replica_dir: str, scene_ids: Optional[List[str]]) -> List[Dict]:
        """Load samples directly from Replica dataset."""
        samples = []
        
        if scene_ids is None:
            # Get all scene directories
            scene_dirs = [d for d in os.listdir(replica_dir) 
                         if os.path.isdir(os.path.join(replica_dir, d))]
        else:
            scene_dirs = scene_ids
        
        for scene_id in scene_dirs:
            scene_path = os.path.join(replica_dir, scene_id)
            results_path = os.path.join(scene_path, 'results')
            
            if os.path.exists(results_path):
                # Get image files
                image_files = sorted([f for f in os.listdir(results_path) 
                                    if f.startswith('frame') and f.endswith('.jpg')])
                
                # Create samples (with placeholder captions - would need to run detection first)
                for img_file in image_files[:10]:  # Limit to first 10 per scene
                    samples.append({
                        'image_path': os.path.join(results_path, img_file),
                        'captions': [],  # Would need to run detection to get captions
                        'object_id': len(samples),
                        'scene_dir': scene_path,
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample['image_path']).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {sample['image_path']}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='black')
        
        return {
            'image': image,
            'captions': sample['captions'],
            'image_path': sample['image_path'],
            'object_id': sample['object_id'],
        }


def create_distillation_dataloader(
    data_root: str,
    batch_size: int = 4,
    scene_ids: Optional[List[str]] = None,
    task: str = "caption_refinement",
    num_workers: int = 2,
    max_samples: Optional[int] = None,
) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for distillation training.
    
    Args:
        data_root: Root directory for neuro-nav data
        batch_size: Batch size for training
        scene_ids: List of scene IDs to use
        task: Task type
        num_workers: Number of data loading workers
        max_samples: Maximum number of samples
        
    Returns:
        DataLoader instance
    """
    dataset = NeuroNavDistillationDataset(
        data_root=data_root,
        scene_ids=scene_ids,
        task=task,
        max_samples=max_samples,
    )
    
    def collate_fn(batch):
        """Custom collate function for batching."""
        images = [item['image'] for item in batch]
        captions = [item['captions'] for item in batch]
        return {
            'images': images,
            'captions': captions,
            'image_paths': [item['image_path'] for item in batch],
        }
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

