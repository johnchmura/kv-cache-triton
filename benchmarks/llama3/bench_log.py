"""JSON logging helpers for llama3 benchmarks.

We want a single machine-readable stream (JSONL) of *all* measurement points so
plots and summaries can be regenerated post-hoc.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _json_default(obj: Any):
    if is_dataclass(obj):
        return asdict(obj)
    return str(obj)


class BenchLogger:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._rows_path = self.out_dir / "rows.jsonl"
        self._fh = open(self._rows_path, "a", encoding="utf-8")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def write_config(self, config: dict[str, Any]) -> None:
        path = self.out_dir / "run_config.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=_json_default)

    def log(self, row: dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("wall_time_s", time.time())
        row.setdefault("hostname", os.uname().nodename if hasattr(os, "uname") else None)
        self._fh.write(json.dumps(row, default=_json_default) + "\n")
        self._fh.flush()

