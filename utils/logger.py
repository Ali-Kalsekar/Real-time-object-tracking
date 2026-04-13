from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class TrackingLogger:
    """Append tracking results to a CSV file."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.columns = ["timestamp", "object_id", "x", "y", "width", "height"]
        if not self.csv_path.exists():
            pd.DataFrame(columns=self.columns).to_csv(self.csv_path, index=False)

    def write_rows(self, rows: list[dict[str, Any]]) -> None:
        """Append rows to the tracking CSV file."""
        if not rows:
            return

        frame = pd.DataFrame(rows, columns=self.columns)
        frame.to_csv(
            self.csv_path,
            mode="a",
            header=not self.csv_path.exists() or self.csv_path.stat().st_size == 0,
            index=False,
        )

    def close(self) -> None:
        """Reserved for compatibility with future buffered logging."""
        return None
