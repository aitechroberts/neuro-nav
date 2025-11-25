"""
Base Query Engine Interface

Defines the abstract interface that all query engines must implement.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class QueryResult:
    """
    Result of a query containing the matched object and metadata.

    Attributes:
        object_id: UUID or index of the matched object
        object_index: Integer index in the object list
        confidence: Confidence score [0, 1]
        reasoning: Optional explanation of why this object was chosen
        class_name: The object's class label
        matched_objects: List of top-K matches (for comparison)
    """
    object_id: Any
    object_index: int
    confidence: float
    reasoning: Optional[str] = None
    class_name: str = "unknown"
    matched_objects: Optional[List[Dict]] = None

    def __str__(self):
        result = f"Object: {self.class_name} (ID: {self.object_id})\n"
        result += f"Confidence: {self.confidence:.3f}\n"
        if self.reasoning:
            result += f"Reasoning: {self.reasoning}\n"
        return result


class BaseQueryEngine(ABC):
    """
    Abstract base class for all query engines.

    Each setup (1-4) implements this interface with different underlying models.
    """

    def __init__(self, objects: List[Dict], config: Optional[Dict] = None):
        """
        Initialize the query engine.

        Args:
            objects: List of object dictionaries from ConceptGraphs mapping
                     Each object has keys: 'id', 'class_name', 'clip_ft', 'bbox', etc.
            config: Optional configuration dictionary
        """
        self.objects = objects
        self.config = config or {}
        self.setup_name = "BaseEngine"

    @abstractmethod
    def answer(self, query: str) -> QueryResult:
        """
        Answer a natural language query about the scene.

        Args:
            query: Natural language query string
                   Examples: "Find me something to sit on"
                            "Where is the chair?"
                            "Find me something red"

        Returns:
            QueryResult containing the best matching object
        """
        pass

    def get_object_by_index(self, idx: int) -> Dict:
        """Helper to get object by index."""
        return self.objects[idx]

    def get_object_by_id(self, obj_id: Any) -> Optional[Dict]:
        """Helper to get object by ID."""
        for obj in self.objects:
            if obj.get('id') == obj_id or obj.get('curr_obj_num') == obj_id:
                return obj
        return None

    def get_all_class_names(self) -> List[str]:
        """Get all unique class names in the scene."""
        return list(set(obj.get('class_name', 'unknown') for obj in self.objects))

    def compute_object_confidence(self, obj: Dict) -> float:
        """
        Compute intrinsic confidence of an object based on detections.

        This is the same confidence metric from Exploration 1.
        """
        # Detection confidence (average YOLO confidence)
        conf_scores = obj.get('conf', [])
        avg_detection_conf = np.mean(conf_scores) if conf_scores else 0.5

        # Temporal confidence (more detections = more certain)
        num_detections = obj.get('num_detections', 1)
        temporal_conf = min(1.0, num_detections / 10.0)

        # Class consistency (how often does YOLO agree?)
        class_ids = obj.get('class_id', [])
        if class_ids:
            from collections import Counter
            counter = Counter(class_ids)
            most_common_count = counter.most_common(1)[0][1]
            class_consistency = most_common_count / len(class_ids)
        else:
            class_consistency = 1.0

        # Weighted average
        confidence = (
            0.4 * avg_detection_conf +
            0.3 * temporal_conf +
            0.3 * class_consistency
        )

        return confidence

    def __str__(self):
        return f"{self.setup_name} ({len(self.objects)} objects in scene)"
