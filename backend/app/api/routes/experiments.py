from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from app.core.settings import get_settings

router = APIRouter()


@router.get("/experiments/history")
async def experiment_history(limit: int = 50) -> list[dict]:
    settings = get_settings()
    path = Path(settings.experiments_path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    selected = lines[-limit:]
    return [json.loads(line) for line in selected]

