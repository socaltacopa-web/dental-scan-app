from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[1]

volume_path = os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
data_path = os.getenv("DATA_DIR")

if data_path:
    DATA_DIR = Path(data_path)
elif volume_path:
    DATA_DIR = Path(volume_path)
else:
    DATA_DIR = BASE / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

SCAN_ID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")


def validate_scan_id(scan_id: str) -> None:
    if not SCAN_ID_RE.match(scan_id):
        raise ValueError("Invalid scan ID.")


def scan_dir_for(scan_id: str) -> Path:
    validate_scan_id(scan_id)
    return DATA_DIR / "scans" / scan_id


def create_scan_dir(scan_id: str) -> Path:
    d = scan_dir_for(scan_id)
    d.mkdir(parents=True, exist_ok=False)
    return d


def status_path(scan_id: str) -> Path:
    return scan_dir_for(scan_id) / "status.json"


def load_status(scan_id: str) -> dict[str, Any] | None:
    p = status_path(scan_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_status(scan_id: str, status: dict[str, Any]) -> None:
    p = status_path(scan_id)
    p.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix="status_",
        suffix=".json",
        dir=str(p.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)
        os.replace(tmp_name, p)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def safe_artifact_path(scan_id: str, artifact_path: str) -> Path:
    base = scan_dir_for(scan_id).resolve()
    candidate = (base / artifact_path).resolve()

    try:
        candidate.relative_to(base)
    except ValueError:
        raise ValueError("Invalid artifact path.")

    return candidate
