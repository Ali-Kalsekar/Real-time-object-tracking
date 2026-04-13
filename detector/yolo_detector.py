from __future__ import annotations

from typing import Any

import cv2
import torch
from ultralytics import YOLO


class YOLODetector:
    """YOLOv8-based person detector optimized for real-time inference."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        img_size: int = 640,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.img_size = img_size
        self.model: YOLO | None = None
        self.load_model()

    def load_model(self) -> YOLO:
        """Load the YOLO model and move it to the requested device."""
        if self.model is not None:
            return self.model

        try:
            model = YOLO(self.model_path)
            try:
                model.to(self.device)
            except Exception:
                pass
            self.model = model
            return model
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model from {self.model_path}") from exc

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        """Detect persons in a frame and return bounding boxes with scores."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return []

        if self.model is None:
            self.load_model()

        if self.model is None:
            return []

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                classes=[0],
                imgsz=self.img_size,
                device=self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu",
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError("YOLO inference failed.") from exc

        detections: list[dict[str, Any]] = []
        result = results[0] if results else None
        if result is None or result.boxes is None:
            return detections

        for box in result.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            x1, y1, x2, y2 = [int(round(value)) for value in xyxy]
            confidence = float(box.conf[0].detach().cpu().item())
            class_id = int(box.cls[0].detach().cpu().item())
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                }
            )

        return detections
