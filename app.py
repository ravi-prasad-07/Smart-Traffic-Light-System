from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from runtime_config import get_runtime_status
from storage import fetch_recent_cycles, fetch_wait_history, init_database, save_cycle
from traffic_core_optimized import LANE_NAMES, analyze_lane_video, build_signal_plan


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = STATIC_DIR / "uploads"
SNAPSHOTS_DIR = STATIC_DIR / "snapshots"
DATABASE_PATH = BASE_DIR / "traffic.db"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024
app.config["JSON_SORT_KEYS"] = False

logging.basicConfig(
    level=getattr(logging, os.getenv("TRAFFIC_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger("traffic.app")

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
init_database(DATABASE_PATH)

# ---------- Government override state (in-memory) ----------
_override_state = {
    "active": False,
    "lane_id": None,
    "reason": "",
    "activated_at": None,
}


def _allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _batch_token() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"{stamp}-{uuid4().hex[:6]}"


def _public_lane(lane: dict) -> dict:
    keys = {
        "lane_id",
        "lane_name",
        "filename",
        "video_url",
        "snapshot_url",
        "vehicle_count",
        "average_count",
        "peak_count",
        "tracked_vehicle_count",
        "pedestrian_count",
        "occupancy_ratio",
        "density_level",
        "priority_score",
        "signal_order",
        "green_time",
        "yellow_time",
        "sampled_frames",
        "duration_seconds",
        "sample_step",
        "processed_fps",
        "analysis_seconds",
        "analysis_fps",
        "inference_fps",
        "inference_batches",
        "emergency_detected",
        "emergency_reason",
        "emergency_confidence",
        "emergency_type",
        "emergency_model_active",
        "is_override",
        "detector_name",
        "inference_backend",
        "inference_precision",
        "gpu_verified",
        "gpu_name",
        "gpu_utilization",
        "gpu_memory_allocated_mb",
    }
    return {key: value for key, value in lane.items() if key in keys}


@app.get("/")
def index():
    return render_template(
        "index.html",
        lane_names=LANE_NAMES,
        recent_cycles=fetch_recent_cycles(DATABASE_PATH, 6),
    )


@app.post("/analyze")
def analyze():
    lane_count = request.form.get("lane_count", type=int)
    if lane_count not in (3, 4):
        return jsonify({"error": "Please choose either 3 lanes or 4 lanes."}), 400

    wait_history = fetch_wait_history(DATABASE_PATH)
    batch_token = _batch_token()
    upload_batch_dir = UPLOADS_DIR / batch_token
    snapshot_batch_dir = SNAPSHOTS_DIR / batch_token
    upload_batch_dir.mkdir(parents=True, exist_ok=True)
    snapshot_batch_dir.mkdir(parents=True, exist_ok=True)

    lanes = []
    for lane_id in range(1, lane_count + 1):
        video = request.files.get(f"lane_{lane_id}")
        if video is None or not video.filename:
            return jsonify({"error": f"Lane {lane_id} is missing a video file."}), 400
        if not _allowed_file(video.filename):
            return jsonify({"error": f"{video.filename} is not a supported video format."}), 400

        original_name = secure_filename(video.filename) or f"lane_{lane_id}.mp4"
        extension = Path(original_name).suffix.lower() or ".mp4"
        stored_name = f"lane_{lane_id}{extension}"
        video_path = upload_batch_dir / stored_name
        snapshot_path = snapshot_batch_dir / f"lane_{lane_id}.jpg"
        video.save(video_path)

        try:
            lane = analyze_lane_video(
                video_path,
                lane_id=lane_id,
                snapshot_path=snapshot_path,
                source_name=original_name,
            )
        except Exception as exc:
            return jsonify({"error": f"Lane {lane_id} analysis failed: {exc}"}), 500
        lane["video_url"] = url_for("static", filename=(Path("uploads") / batch_token / stored_name).as_posix())
        lane["snapshot_url"] = url_for(
            "static", filename=(Path("snapshots") / batch_token / snapshot_path.name).as_posix()
        )
        lane["video_path"] = str(video_path)
        lane["snapshot_path"] = str(snapshot_path)
        lanes.append(lane)

    # Check if government override is active
    government_override = None
    if _override_state["active"] and _override_state["lane_id"]:
        government_override = {"lane_id": _override_state["lane_id"]}

    plan = build_signal_plan(lanes, wait_history=wait_history, government_override=government_override)
    cycle_id = save_cycle(DATABASE_PATH, plan)

    return jsonify(
        {
            "cycle_id": cycle_id,
            "lane_count": lane_count,
            "priority_lane": plan["priority_lane"],
            "priority_lane_name": plan["priority_lane_name"],
            "cycle_total": plan["cycle_total"],
            "decision_text": plan["decision_text"],
            "lanes": [_public_lane(lane) for lane in plan["lanes"]],
            "signal_sequence": plan["signal_sequence"],
            "pedestrian_phase": plan.get("pedestrian_phase"),
            "total_pedestrians": plan.get("total_pedestrians", 0),
            "government_override_active": plan.get("government_override_active", False),
            "override_lane_id": plan.get("override_lane_id"),
            "override_state": _override_state,
            "runtime_status": get_runtime_status(),
            "history": fetch_recent_cycles(DATABASE_PATH, 6),
        }
    )


# ---------- Government Override API ----------

@app.post("/override")
def set_override():
    """Activate government priority override for a specific lane."""
    data = request.get_json(silent=True) or {}
    lane_id = data.get("lane_id")
    reason = data.get("reason", "Government/Authority override")

    if lane_id is None:
        return jsonify({"error": "lane_id is required."}), 400

    lane_id = int(lane_id)
    if lane_id < 1 or lane_id > 4:
        return jsonify({"error": "lane_id must be between 1 and 4."}), 400

    _override_state["active"] = True
    _override_state["lane_id"] = lane_id
    _override_state["reason"] = reason
    _override_state["activated_at"] = datetime.now(timezone.utc).isoformat()

    lane_name = LANE_NAMES[lane_id - 1] if lane_id <= len(LANE_NAMES) else f"Lane {lane_id}"

    return jsonify({
        "status": "override_activated",
        "lane_id": lane_id,
        "lane_name": lane_name,
        "reason": reason,
        "message": f"Government override activated. {lane_name} will receive top priority.",
    })


@app.post("/override/clear")
def clear_override():
    """Deactivate government override and return to automatic mode."""
    prev = dict(_override_state)
    _override_state["active"] = False
    _override_state["lane_id"] = None
    _override_state["reason"] = ""
    _override_state["activated_at"] = None

    return jsonify({
        "status": "override_cleared",
        "message": "Government override deactivated. System returned to automatic mode.",
        "previous_override": prev,
    })


@app.get("/override/status")
def override_status():
    """Check current override status."""
    return jsonify(_override_state)


@app.get("/runtime/status")
def runtime_status():
    """Check current detector and GPU runtime status."""
    return jsonify(get_runtime_status())


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Upload too large. Use shorter or lower-resolution traffic videos."}), 413


if __name__ == "__main__":
    LOGGER.info("Runtime status: %s", get_runtime_status())
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n{'='*60}\n🚀 HARDWARE ACCELERATION ACTIVE: Using eGPU ({gpu_name})\n{'='*60}\n")
    else:
        print(f"\n{'='*60}\n⚠️ RUNNING ON CPU: PyTorch did not detect a CUDA-compatible GPU.\n{'='*60}\n")
    app.run(debug=True)
