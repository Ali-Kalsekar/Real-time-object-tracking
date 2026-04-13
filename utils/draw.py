from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def _track_color(track_id: int) -> tuple[int, int, int]:
    """Generate a stable color for a track ID."""
    palette = [
        (54, 162, 235),
        (255, 99, 132),
        (75, 192, 192),
        (255, 206, 86),
        (153, 102, 255),
        (255, 159, 64),
    ]
    return palette[track_id % len(palette)]


def draw_tracking_info(
    frame: np.ndarray,
    bbox: Iterable[int],
    track_id: int,
    confidence: float,
) -> None:
    """Draw the tracking box, ID, and confidence score on the frame."""
    x1, y1, x2, y2 = [int(value) for value in bbox]
    color = _track_color(track_id)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label_lines = [f"ID: {track_id}", f"Conf: {confidence:.2f}"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    line_height = 18
    padding = 6

    text_width = 0
    text_heights: list[int] = []
    for line in label_lines:
        (text_size, _) = cv2.getTextSize(line, font, font_scale, thickness)
        text_width = max(text_width, text_size[0])
        text_heights.append(text_size[1])

    box_width = text_width + padding * 2
    box_height = sum(text_heights) + padding * 2 + 4

    box_x1 = max(0, x1)
    box_y1 = max(0, y1 - box_height - 6)
    box_x2 = min(frame.shape[1] - 1, box_x1 + box_width)
    box_y2 = min(frame.shape[0] - 1, box_y1 + box_height)

    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), color, -1)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), 1)

    text_x = box_x1 + padding
    text_y = box_y1 + padding + text_heights[0]
    cv2.putText(frame, label_lines[0], (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(
        frame,
        label_lines[1],
        (text_x, text_y + line_height),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
