from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


def get_ultralytics_detections(
    frame: np.ndarray,
    model: YOLO,
    model_params: Dict[str, Any],
    class_confidence: List[Tuple[List[int], float]] = None,
    bgr: bool = False,
) -> sv.Detections:
    """
    Runs object detection on a single frame and filters out low-confidence detections
    for specified class groups.

    Args:
        frame (np.ndarray): Input image/frame as a NumPy array (e.g., BGR or RGB).
        model (YOLO): An Ultralytics YOLO model or compatible callable model.
        model_params (Dict[str, Any]): Parameters to pass into the model's forward call (e.g., conf, iou, imgsz).
        class_confidence (List[Tuple[List[int], float]], optional):
            List of (class_ids, threshold) pairs. Detections belonging to any class in class_ids
            below the given threshold will be filtered out.
        bgr (bool): Whether the input frame is in BGR format. If True, converts to RGB for inference.

    Returns:
        sv.Detections: Filtered detections compatible with the Supervision library.
    """
    # Create a copy to avoid modifying the original frame
    frame_copy = frame.copy()

    # Convert BGR to RGB if needed
    if bgr:
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)

    results = model(frame_copy, **model_params)[0]
    detections = sv.Detections.from_ultralytics(results)

    if class_confidence:
        for class_ids, threshold in class_confidence:
            is_in_classes = np.isin(detections.class_id, class_ids)
            is_below_threshold = detections.confidence < threshold
            detections = detections[~(is_in_classes & is_below_threshold)]

    return detections
