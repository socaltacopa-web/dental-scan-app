from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .jobs import process_scan
from .storage import (
    DATA_DIR,
    create_scan_dir,
    load_status,
    safe_artifact_path,
    save_status,
    scan_dir_for,
    validate_scan_id,
)

APP_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = APP_ROOT / "static"

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
AUTO_PROCESS = os.getenv("AUTO_PROCESS", "1") == "1"

app = FastAPI(
    title="Dental Scan Prototype API",
    version="0.1.0",
    description="Prototype upload and 3D reconstruction service. Not a clinical medical device.",
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "dental-scan-prototype",
        "data_dir": str(DATA_DIR),
    }


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "max_upload_mb": MAX_UPLOAD_MB,
        "auto_process": AUTO_PROCESS,
        "prototype": True,
    }


@app.post("/api/scans")
async def create_scan(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    case_name: str = Form("Untitled scan"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing upload filename.")

    scan_id = str(uuid.uuid4())
    scan_dir = create_scan_dir(scan_id)
    input_path = scan_dir / "input.dscan.json"

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    total = 0

    try:
        with input_path.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Scan exceeds {MAX_UPLOAD_MB} MB upload limit.",
                    )
                out.write(chunk)

        try:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON scan package: {exc}")

        if payload.get("format") not in {"dscan-v4", "dscan-v3"}:
            raise HTTPException(
                status_code=400,
                detail="Expected a dscan-v4 or dscan-v3 package.",
            )

        sweeps = payload.get("sweeps")
        if not isinstance(sweeps, list) or not sweeps:
            raise HTTPException(status_code=400, detail="Scan package has no sweeps.")

        safe_case = re.sub(r"[\r\n\t]+", " ", case_name).strip()[:120] or "Untitled scan"

        status = {
            "scan_id": scan_id,
            "case_name": safe_case,
            "status": "queued" if AUTO_PROCESS else "uploaded",
            "stage": "waiting",
            "message": "Scan uploaded.",
            "bytes": total,
            "artifacts": [],
            "prototype_warning": (
                "This reconstruction is experimental and must not be used for "
                "diagnosis, treatment planning, measurements, or fabrication."
            ),
        }
        save_status(scan_id, status)

        if AUTO_PROCESS:
            background_tasks.add_task(process_scan, scan_id)

        return JSONResponse(status_code=201, content=status)

    except HTTPException:
        shutil.rmtree(scan_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(scan_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/scans/{scan_id}")
def get_scan(scan_id: str):
    validate_scan_id(scan_id)
    status = load_status(scan_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Scan not found.")

    scan_dir = scan_dir_for(scan_id)
    artifacts = []

    for rel in [
        "global/merged_mouth.ply",
        "global/merged_mouth_mesh.ply",
        "global/alignment_report.json",
        "reconstruction/summary.txt",
        "processing.log",
    ]:
        p = scan_dir / rel
        if p.exists() and p.is_file():
            artifacts.append(
                {
                    "name": p.name,
                    "path": rel,
                    "url": f"/api/scans/{scan_id}/artifacts/{rel}",
                }
            )

    status["artifacts"] = artifacts
    return status


@app.post("/api/scans/{scan_id}/process")
def start_processing(scan_id: str, background_tasks: BackgroundTasks):
    validate_scan_id(scan_id)
    status = load_status(scan_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Scan not found.")

    if status.get("status") in {"processing", "completed"}:
        return status

    status.update(
        {
            "status": "queued",
            "stage": "waiting",
            "message": "Processing queued.",
        }
    )
    save_status(scan_id, status)
    background_tasks.add_task(process_scan, scan_id)
    return status


@app.get("/api/scans/{scan_id}/artifacts/{artifact_path:path}")
def get_artifact(scan_id: str, artifact_path: str):
    validate_scan_id(scan_id)
    path = safe_artifact_path(scan_id, artifact_path)

    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found.")

    return FileResponse(
        path,
        filename=path.name,
        media_type="application/octet-stream",
    )


@app.delete("/api/scans/{scan_id}")
def delete_scan(scan_id: str):
    validate_scan_id(scan_id)
    scan_dir = scan_dir_for(scan_id)

    if not scan_dir.exists():
        raise HTTPException(status_code=404, detail="Scan not found.")

    shutil.rmtree(scan_dir)
    return {"deleted": True, "scan_id": scan_id}


@app.get("/")
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/viewer")
def viewer():
    return FileResponse(STATIC_DIR / "viewer.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
