from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRecord:
    run_id: str
    started_at: str
    model_family: str
    target: str
    algorithm: str
    metrics: dict[str, float]
    params: dict[str, Any]
    artifact_path: str


class JsonExperimentTracker:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: ExperimentRecord) -> None:
        payload = asdict(record)
        with self.output_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload) + "\n")

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

