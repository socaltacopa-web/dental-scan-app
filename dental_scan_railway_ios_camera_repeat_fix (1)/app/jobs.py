from __future__ import annotations

import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .storage import load_status, save_status, scan_dir_for

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "pipeline"
PROCESS_TIMEOUT_SECONDS = int(os.getenv("PROCESS_TIMEOUT_SECONDS", "1800"))
ENABLE_MESH = os.getenv("ENABLE_MESH", "1") == "1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def update(scan_id: str, **changes):
    status = load_status(scan_id) or {"scan_id": scan_id}
    status.update(changes)
    status["updated_at"] = utc_now()
    save_status(scan_id, status)


def run_command(command: list[str], log_path: Path) -> subprocess.CompletedProcess:
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n\n$ " + " ".join(command) + "\n")
        log.flush()
        return subprocess.run(
            command,
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=PROCESS_TIMEOUT_SECONDS,
            check=False,
        )


def process_scan(scan_id: str):
    scan_dir = scan_dir_for(scan_id)
    package = scan_dir / "input.dscan.json"
    reconstruction_dir = scan_dir / "reconstruction"
    global_dir = scan_dir / "global"
    log_path = scan_dir / "processing.log"

    try:
        update(
            scan_id,
            status="processing",
            stage="reconstructing_sweeps",
            message="Building individual 3D sweep point clouds.",
            started_at=utc_now(),
        )

        reconstruct_cmd = [
            sys.executable,
            str(PIPELINE / "reconstruct_sweeps.py"),
            str(package),
            "--output",
            str(reconstruction_dir),
        ]

        first = run_command(reconstruct_cmd, log_path)
        if first.returncode != 0:
            update(
                scan_id,
                status="failed",
                stage="reconstructing_sweeps",
                message="Individual sweep reconstruction failed. Check processing.log.",
                finished_at=utc_now(),
            )
            return

        update(
            scan_id,
            status="processing",
            stage="aligning_global_model",
            message="Aligning overlapping sweeps into one mouth model.",
        )

        align_cmd = [
            sys.executable,
            str(PIPELINE / "align_global.py"),
            str(reconstruction_dir),
            "--package",
            str(package),
            "--output",
            str(global_dir),
        ]
        if ENABLE_MESH:
            align_cmd.append("--mesh")

        second = run_command(align_cmd, log_path)

        if second.returncode != 0:
            update(
                scan_id,
                status="partial",
                stage="global_alignment_failed",
                message=(
                    "Individual sweep reconstructions finished, but global alignment "
                    "did not complete. The per-sweep models and log are still available."
                ),
                finished_at=utc_now(),
            )
            return

        merged = global_dir / "merged_mouth.ply"

        update(
            scan_id,
            status="completed",
            stage="done",
            message=(
                "Reconstruction completed."
                if merged.exists()
                else "Processing completed, but no merged model was produced."
            ),
            finished_at=utc_now(),
        )

    except subprocess.TimeoutExpired:
        update(
            scan_id,
            status="failed",
            stage="timeout",
            message=f"Processing exceeded {PROCESS_TIMEOUT_SECONDS} seconds.",
            finished_at=utc_now(),
        )
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as log:
            log.write("\n\nUNHANDLED ERROR\n")
            traceback.print_exc(file=log)

        update(
            scan_id,
            status="failed",
            stage="exception",
            message=str(exc),
            finished_at=utc_now(),
        )
