from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class FPSCounter:
    """Compute frames per second using wall-clock time."""

    start_time: float = field(default_factory=time.perf_counter)
    frame_count: int = 0
    last_fps: float = 0.0

    def update(self) -> float:
        """Update the frame counter and return the current FPS."""
        self.frame_count += 1
        elapsed = time.perf_counter() - self.start_time
        if elapsed > 0:
            self.last_fps = self.frame_count / elapsed
        return self.last_fps

    def timestamp_iso(self) -> str:
        """Return the current timestamp in ISO 8601 format."""
        return datetime.now().isoformat(timespec="seconds")
