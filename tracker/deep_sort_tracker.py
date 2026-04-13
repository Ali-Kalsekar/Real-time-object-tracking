from __future__ import annotations

from typing import Any

from deep_sort_realtime.deepsort_tracker import DeepSort


class DeepSortTracker:
    """DeepSORT tracker wrapper with ReID embeddings."""

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.2,
        nn_budget: int = 100,
        use_gpu: bool = False,
    ) -> None:
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.use_gpu = use_gpu
        self.tracker = self._build_tracker()

    def _build_tracker(self) -> DeepSort:
        """Instantiate DeepSORT with GPU-aware embedding when available."""
        tracker_kwargs = {
            "max_age": self.max_age,
            "n_init": self.n_init,
            "max_cosine_distance": self.max_cosine_distance,
            "nn_budget": self.nn_budget,
            "embedder": "mobilenet",
            "half": self.use_gpu,
            "bgr": True,
        }

        try:
            return DeepSort(**tracker_kwargs, embedder_gpu=self.use_gpu)
        except TypeError:
            return DeepSort(**tracker_kwargs)

    def update(self, detections: list[dict[str, Any]], frame: Any) -> list[dict[str, Any]]:
        """Update the tracker state and return active tracks."""
        if frame is None or getattr(frame, "size", 0) == 0:
            return []

        raw_detections: list[tuple[list[int], float, int]] = []
        for detection in detections:
            bbox = detection.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(value) for value in bbox]
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            raw_detections.append(
                ([x1, y1, width, height], float(detection.get("confidence", 0.0)), int(detection.get("class_id", 0)))
            )

        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        active_tracks: list[dict[str, Any]] = []

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = [int(round(value)) for value in ltrb]
            confidence = float(getattr(track, "det_conf", 0.0) or 0.0)
            class_id = int(getattr(track, "det_class", 0) or 0)
            active_tracks.append(
                {
                    "track_id": int(track.track_id),
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                }
            )

        return active_tracks
