"""Shared helper: write pipeline progress to a JSON file that dashboard.html polls."""
import json
import os
import time
from pathlib import Path

PROGRESS_PATH = Path("C:/Users/Gaming PC/musicgen_data/progress.json")


class ProgressWriter:
    def __init__(self, stage: str, total: int, path: Path = PROGRESS_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.total = total
        self.current = 0
        self.current_item = ""
        self.extra = {}
        self.started_at = time.time()
        self._write()

    def update(self, current: int, current_item: str = "", **extra):
        self.current = current
        self.current_item = current_item
        self.extra = extra
        self._write()

    def finish(self, **extra):
        self.current = self.total
        self.extra = extra
        self.stage = f"{self.stage}_done"
        self._write()

    def _write(self):
        data = {
            "stage": self.stage,
            "current": self.current,
            "total": self.total,
            "current_item": self.current_item,
            "started_at": self.started_at,
            "updated_at": time.time(),
            "extra": self.extra,
        }
        # Best-effort: a stalled progress bar is not worth crashing a multi-hour
        # pipeline over. Windows can transiently deny os.replace() if something
        # (AV scanner, the dashboard's http.server) has the file briefly open.
        tmp = self.path.parent / f".progress_{os.getpid()}.tmp"
        try:
            tmp.write_text(json.dumps(data), encoding="utf-8")
            for attempt in range(5):
                try:
                    os.replace(tmp, self.path)
                    return
                except PermissionError:
                    time.sleep(0.05 * (attempt + 1))
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
