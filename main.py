from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import yaml

from detector.yolo_detector import YOLODetector
from tracker.deep_sort_tracker import DeepSortTracker
from utils.draw import draw_tracking_info
from utils.fps import FPSCounter
from utils.logger import TrackingLogger


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: Path) -> dict[str, Any]:
    """Load application configuration from a YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    return config


def resolve_video_source(value: Any) -> Any:
    """Convert the configured video source into a value accepted by OpenCV."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return value


def resolve_path(path_value: str) -> Path:
    """Resolve a path relative to the project root unless it is already absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def resolve_device(config_value: str | None) -> str:
    """Resolve the execution device for inference."""
    if config_value:
        normalized = config_value.strip().lower()
        if normalized in {"cpu", "cuda"}:
            return normalized

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Real-Time Multi-Object Tracking System")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the real-time person detection and tracking pipeline."""
    args = parse_args()
    config = load_config(Path(args.config))

    video_source = resolve_video_source(config.get("video_source", 0))
    confidence_threshold = float(config.get("confidence_threshold", 0.5))
    max_age = int(config.get("max_age", 30))
    n_init = int(config.get("n_init", 3))
    resize_width = int(config.get("resize_width", 1280))
    img_size = int(config.get("imgsz", 640))
    model_path = str(config.get("model_path", "yolov8n.pt"))
    output_csv = resolve_path(str(config.get("output_csv", "output/tracking_log.csv")))
    window_title = str(config.get("window_title", "Real-Time Object Tracking System"))
    tracker_max_cosine_distance = float(config.get("tracker_max_cosine_distance", 0.2))
    tracker_nn_budget = int(config.get("tracker_nn_budget", 100))
    device = resolve_device(config.get("device"))

    capture = cv2.VideoCapture(video_source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {video_source}")

    detector = YOLODetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device,
        img_size=img_size,
    )
    tracker = DeepSortTracker(
        max_age=max_age,
        n_init=n_init,
        max_cosine_distance=tracker_max_cosine_distance,
        nn_budget=tracker_nn_budget,
        use_gpu=device == "cuda",
    )
    fps_counter = FPSCounter()
    logger = TrackingLogger(output_csv)

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if resize_width > 0 and frame.shape[1] > resize_width:
                scale = resize_width / float(frame.shape[1])
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (resize_width, new_height), interpolation=cv2.INTER_AREA)

            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)

            current_timestamp = fps_counter.timestamp_iso()
            log_rows: list[dict[str, Any]] = []

            for track in tracks:
                bbox = track["bbox"]
                track_id = track["track_id"]
                confidence = track.get("confidence", 0.0)
                draw_tracking_info(frame, bbox, track_id, confidence)
                x1, y1, x2, y2 = bbox
                log_rows.append(
                    {
                        "timestamp": current_timestamp,
                        "object_id": track_id,
                        "x": x1,
                        "y": y1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                    }
                )

            if log_rows:
                logger.write_rows(log_rows)

            fps = fps_counter.update()
            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_title, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        logger.close()


if __name__ == "__main__":
    main()
