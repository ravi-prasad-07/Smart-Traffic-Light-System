"""Optimized traffic analysis engine for GPU-first YOLO inference."""

from __future__ import annotations

import logging
import math
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch

from emergency_detector_v2 import EmergencyDetector
from runtime_config import configure_torch_runtime, get_runtime_config, get_runtime_status
from vehicle_tracker import VehicleTracker


RUNTIME = configure_torch_runtime()
LOGGER = logging.getLogger("traffic.core")

cv2.setUseOptimized(True)
cv2.setNumThreads(max(1, min(4, cv2.getNumberOfCPUs())))

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - surfaced at runtime
    YOLO = None
    YOLO_IMPORT_ERROR = exc
else:
    YOLO_IMPORT_ERROR = None


LANE_NAMES = ("North Lane", "East Lane", "South Lane", "West Lane")
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PEDESTRIAN_CLASS_ID = 0
ALL_DETECT_IDS = list(VEHICLE_CLASS_IDS.keys()) + [PEDESTRIAN_CLASS_ID]
CLASS_CONF_THRESHOLDS = {
    PEDESTRIAN_CLASS_ID: 0.22,
    2: 0.20,
    3: 0.20,
    5: 0.24,
    7: 0.24,
}


def _resize(frame: np.ndarray, max_width: int = 1280) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _extract_emergency_type(reason: str) -> str:
    lowered = reason.lower()
    if "ambulance" in lowered:
        return "ambulance"
    if "fire_truck" in lowered or "fire truck" in lowered:
        return "fire_truck"
    return ""


def _sample_step(fps: float, total_frames: int) -> int:
    if fps <= 0:
        return 1
    step_by_target_fps = max(1, int(round(fps / RUNTIME.sample_fps)))
    if total_frames <= 0:
        return step_by_target_fps
    step_by_max_samples = max(1, int(math.ceil(total_frames / float(RUNTIME.max_samples))))
    return max(step_by_target_fps, step_by_max_samples)


def _predict_kwargs(*, conf: float | None = None, classes: list[int] | None = None, max_det: int | None = None, batch_size: int | None = None) -> dict:
    kwargs = {
        "imgsz": RUNTIME.imgsz,
        "conf": conf if conf is not None else RUNTIME.yolo_conf,
        "iou": RUNTIME.yolo_iou,
        "device": RUNTIME.device,
        "half": RUNTIME.use_half,
        "verbose": False,
        "max_det": max_det if max_det is not None else RUNTIME.yolo_max_det,
        "batch": batch_size if batch_size is not None else RUNTIME.batch_size,
    }
    if classes is not None:
        kwargs["classes"] = classes
    return kwargs


def _verify_result_device(result) -> None:
    boxes = getattr(result, "boxes", None)
    if (
        RUNTIME.use_cuda
        and RUNTIME.strict_cuda
        and boxes is not None
        and len(boxes) > 0
        and getattr(boxes.data, "device", None) is not None
        and boxes.data.device.type != "cuda"
    ):
        raise RuntimeError("YOLO inference moved off CUDA unexpectedly.")


def _prepare_model(model) -> None:
    if hasattr(model, "to"):
        model.to(RUNTIME.device)

    if RUNTIME.backend != "pytorch" or not hasattr(model, "model"):
        return

    model.model.eval()

    if RUNTIME.torch_compile and hasattr(torch, "compile"):
        try:
            model.model = torch.compile(model.model, mode="reduce-overhead")
            LOGGER.info("torch.compile enabled for %s", RUNTIME.model_name)
        except Exception as exc:
            LOGGER.warning("torch.compile could not be enabled: %s", exc)


def _warmup_model(model) -> None:
    warmup_batch = max(1, min(RUNTIME.batch_size, 2))
    dummy = np.zeros((RUNTIME.imgsz, RUNTIME.imgsz, 3), dtype=np.uint8)
    with torch.inference_mode():
        for _ in range(RUNTIME.warmup_runs):
            model.predict(
                source=[dummy] * warmup_batch,
                **_predict_kwargs(classes=ALL_DETECT_IDS, max_det=1, batch_size=warmup_batch),
            )
    if RUNTIME.use_cuda:
        torch.cuda.synchronize(RUNTIME.device_index)


@lru_cache(maxsize=1)
def _get_model():
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics could not be imported. Install the dependencies in requirements.txt first."
        ) from YOLO_IMPORT_ERROR
    if not RUNTIME.general_model_path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {RUNTIME.general_model_path}. "
            "Add yolov8m.pt or yolov8l.pt to the project root, or set TRAFFIC_YOLO_MODEL explicitly."
        )

    model = YOLO(str(RUNTIME.general_model_path))
    _prepare_model(model)
    _warmup_model(model)
    LOGGER.info(
        "Loaded detector model=%s backend=%s device=%s precision=%s batch=%s",
        RUNTIME.general_model_path.name,
        RUNTIME.backend,
        RUNTIME.device,
        RUNTIME.precision,
        RUNTIME.batch_size,
    )
    return model


def _parse_result(result) -> tuple[list[dict], list[dict]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return [], []

    _verify_result_device(result)
    data = boxes.data.detach().cpu().numpy()

    vehicles: list[dict] = []
    pedestrians: list[dict] = []
    for row in data:
        x1, y1, x2, y2, confidence, class_id = row[:6]
        class_id = int(class_id)
        threshold = CLASS_CONF_THRESHOLDS.get(class_id, RUNTIME.yolo_conf)
        if float(confidence) < threshold:
            continue

        width = max(0, int(round(x2 - x1)))
        height = max(0, int(round(y2 - y1)))
        if width <= 0 or height <= 0:
            continue

        det = {
            "box": (int(round(x1)), int(round(y1)), width, height),
            "confidence": float(confidence),
            "class_id": class_id,
            "label": VEHICLE_CLASS_IDS.get(class_id, "person"),
            "is_pedestrian": class_id == PEDESTRIAN_CLASS_ID,
        }
        if class_id == PEDESTRIAN_CLASS_ID:
            pedestrians.append(det)
        else:
            vehicles.append(det)

    return vehicles, pedestrians


def _run_detection_batch(frames: list[np.ndarray]) -> list[tuple[list[dict], list[dict]]]:
    if not frames:
        return []

    model = _get_model()
    with torch.inference_mode():
        results = model.predict(
            source=frames,
            **_predict_kwargs(
                classes=ALL_DETECT_IDS,
                batch_size=min(len(frames), RUNTIME.batch_size),
            ),
        )
    return [_parse_result(result) for result in results]


def _density_label(vehicle_count: int, occupancy_ratio: float) -> str:
    score = vehicle_count + (occupancy_ratio * 10.0)
    if score >= 12:
        return "Very Heavy"
    if score >= 8:
        return "Heavy"
    if score >= 5:
        return "Moderate"
    if score >= 2:
        return "Light"
    return "Low"


def _annotate_frame(
    frame: np.ndarray,
    vehicle_dets: list[dict],
    pedestrian_dets: list[dict],
    lane_name: str,
    vehicle_count: int,
    pedestrian_count: int,
    occupancy_ratio: float,
    density_level: str,
    emergency_detected: bool,
    emergency_reason: str,
    emergency_confidence: float,
) -> np.ndarray:
    annotated = frame.copy()
    vehicle_color = (80, 220, 120)
    emergency_color = (60, 90, 255)
    pedestrian_color = (255, 200, 60)

    for detection in vehicle_dets:
        x, y, width, height = detection.get("stable_box", detection["box"])
        confidence = detection["confidence"]
        emergency_label = detection.get("emergency_label", "")
        confirmed = detection.get("emergency_confirmed", False)
        box_color = emergency_color if confirmed else vehicle_color
        cv2.rectangle(annotated, (x, y), (x + width, y + height), box_color, 2)
        label = emergency_label or detection["label"]
        tag = f"{label} {confidence:.0%}"
        cv2.putText(
            annotated,
            tag,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2,
            cv2.LINE_AA,
        )

    for detection in pedestrian_dets:
        x, y, width, height = detection.get("stable_box", detection["box"])
        confidence = detection["confidence"]
        cv2.rectangle(annotated, (x, y), (x + width, y + height), pedestrian_color, 2)
        cv2.putText(
            annotated,
            f"person {confidence:.0%}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            pedestrian_color,
            2,
            cv2.LINE_AA,
        )

    overlay = annotated.copy()
    cv2.rectangle(overlay, (15, 15), (440, 240), (10, 18, 28), -1)
    cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)

    lines = [
        lane_name,
        f"Detector: {RUNTIME.model_name}",
        f"Vehicles (tracked): {vehicle_count}",
        f"Pedestrians: {pedestrian_count}",
        f"Occupancy: {occupancy_ratio * 100:.1f}%",
        f"Density: {density_level}",
        f"Emergency: {'YES' if emergency_detected else 'NO'} ({emergency_confidence:.0%})",
    ]
    if emergency_reason:
        lines.append(emergency_reason[:62])

    colors = [
        (255, 255, 255),
        (110, 214, 255),
        (220, 230, 240),
        pedestrian_color,
        (220, 230, 240),
        (255, 194, 92),
        emergency_color if emergency_detected else vehicle_color,
        (220, 230, 240),
    ]
    for row_idx, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (28, 42 + (row_idx * 26)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            colors[min(row_idx, len(colors) - 1)],
            2,
            cv2.LINE_AA,
        )
    return annotated


def _iter_sampled_frames(capture: cv2.VideoCapture, sample_step: int):
    frame_index = 0
    while True:
        ok = capture.grab()
        if not ok:
            break
        if frame_index % sample_step != 0:
            frame_index += 1
            continue

        ok, frame = capture.retrieve()
        if not ok:
            break
        yield frame_index, _resize(frame)
        frame_index += 1


def _process_batch(
    batch_frames: list[np.ndarray],
    batch_indices: list[int],
    tracker: VehicleTracker,
    emergency_detector: EmergencyDetector,
    source_label: str,
    lane_id: int,
    samples: list[dict],
    perf: dict,
) -> None:
    if not batch_frames:
        return

    infer_start = time.perf_counter()
    detection_results = _run_detection_batch(batch_frames)
    infer_elapsed = time.perf_counter() - infer_start

    perf["frames_inferred"] += len(batch_frames)
    perf["batches"] += 1
    perf["inference_seconds"] += infer_elapsed

    for frame_index, frame, (vehicle_dets, pedestrian_dets) in zip(batch_indices, batch_frames, detection_results):
        tracker.update(vehicle_dets + pedestrian_dets)

        frame_area = float(frame.shape[0] * frame.shape[1]) + 1e-6
        occupancy_ratio = min(
            1.0,
            sum(det["box"][2] * det["box"][3] for det in vehicle_dets) / frame_area,
        )

        emergency_detected, emergency_reason, emergency_confidence = emergency_detector.process_frame(
            frame,
            vehicle_dets,
            source_label,
        )
        if emergency_detected:
            LOGGER.info(
                "Lane %s emergency detected frame=%s conf=%.3f reason=%s",
                lane_id,
                frame_index,
                emergency_confidence,
                emergency_reason,
            )

        samples.append(
            {
                "frame_index": frame_index,
                "frame": frame,
                "vehicle_dets": vehicle_dets,
                "pedestrian_dets": pedestrian_dets,
                "vehicle_count": len(vehicle_dets),
                "pedestrian_count": len(pedestrian_dets),
                "occupancy_ratio": occupancy_ratio,
                "emergency_detected": emergency_detected,
                "emergency_reason": emergency_reason,
                "emergency_confidence": emergency_confidence,
            }
        )


def analyze_lane_video(
    video_path: str | Path,
    lane_id: int,
    snapshot_path: str | Path,
    source_name: str | None = None,
) -> dict:
    """Analyze a single lane with batched GPU inference."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    duration_seconds = round(total_frames / fps, 2) if fps > 0 else 0.0
    sample_step = _sample_step(fps, total_frames)
    processed_fps = fps / sample_step if fps > 0 else RUNTIME.sample_fps
    source_label = source_name or Path(video_path).name

    tracker = VehicleTracker(iou_threshold=0.25, max_misses=4, min_hits=2, smoothing_alpha=0.40)
    emergency_detector = EmergencyDetector(fps=processed_fps)

    samples: list[dict] = []
    perf = {"frames_inferred": 0, "batches": 0, "inference_seconds": 0.0}
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []

    analysis_start = time.perf_counter()
    for frame_index, frame in _iter_sampled_frames(capture, sample_step):
        batch_frames.append(frame)
        batch_indices.append(frame_index)
        if len(batch_frames) >= RUNTIME.batch_size:
            _process_batch(
                batch_frames,
                batch_indices,
                tracker,
                emergency_detector,
                source_label,
                lane_id,
                samples,
                perf,
            )
            batch_frames = []
            batch_indices = []

    if batch_frames:
        _process_batch(
            batch_frames,
            batch_indices,
            tracker,
            emergency_detector,
            source_label,
            lane_id,
            samples,
            perf,
        )

    capture.release()

    if not samples:
        raise ValueError(f"No readable frames found in {video_path}.")

    analysis_seconds = max(time.perf_counter() - analysis_start, 1e-6)
    average_count = round(float(sum(sample["vehicle_count"] for sample in samples)) / len(samples), 2)
    tracked_vehicle_count = tracker.get_unique_vehicle_count()
    tracked_pedestrian_count = tracker.get_unique_pedestrian_count()

    representative = max(
        samples,
        key=lambda sample: (
            sample["emergency_detected"],
            sample["emergency_confidence"],
            sample["vehicle_count"],
            sample["occupancy_ratio"],
        ),
    )
    peak_count = int(representative["vehicle_count"])
    estimated_count = max(
        tracked_vehicle_count,
        int(round(peak_count * 0.50 + average_count * 0.30 + tracked_vehicle_count * 0.20)),
    )

    occupancy_ratio = round(max(sample["occupancy_ratio"] for sample in samples), 4)
    density_level = _density_label(estimated_count, occupancy_ratio)

    emergency_samples = [sample for sample in samples if sample["emergency_detected"]]
    emergency_detected = bool(emergency_samples)
    if emergency_samples:
        best_emergency = max(emergency_samples, key=lambda sample: sample["emergency_confidence"])
        emergency_reason = best_emergency["emergency_reason"]
        emergency_confidence = round(float(best_emergency["emergency_confidence"]), 3)
    else:
        emergency_reason = ""
        emergency_confidence = 0.0

    annotated = _annotate_frame(
        representative["frame"],
        representative["vehicle_dets"],
        representative["pedestrian_dets"],
        LANE_NAMES[lane_id - 1],
        estimated_count,
        tracked_pedestrian_count,
        occupancy_ratio,
        density_level,
        emergency_detected,
        emergency_reason,
        emergency_confidence,
    )
    cv2.imwrite(str(snapshot_path), annotated)

    runtime_status = get_runtime_status()
    inference_seconds = max(perf["inference_seconds"], 1e-6)
    return {
        "lane_id": lane_id,
        "lane_name": LANE_NAMES[lane_id - 1],
        "filename": Path(video_path).name,
        "vehicle_count": estimated_count,
        "average_count": average_count,
        "peak_count": peak_count,
        "tracked_vehicle_count": tracked_vehicle_count,
        "pedestrian_count": tracked_pedestrian_count,
        "occupancy_ratio": occupancy_ratio,
        "density_level": density_level,
        "sampled_frames": len(samples),
        "sample_step": sample_step,
        "processed_fps": round(processed_fps, 2),
        "duration_seconds": duration_seconds,
        "analysis_seconds": round(analysis_seconds, 3),
        "analysis_fps": round(len(samples) / analysis_seconds, 2),
        "inference_fps": round(perf["frames_inferred"] / inference_seconds, 2),
        "inference_batches": perf["batches"],
        "emergency_detected": emergency_detected,
        "emergency_reason": emergency_reason,
        "emergency_confidence": emergency_confidence,
        "emergency_type": _extract_emergency_type(emergency_reason),
        "detector_name": RUNTIME.model_name,
        "inference_backend": RUNTIME.backend,
        "inference_precision": RUNTIME.precision,
        "gpu_verified": bool(runtime_status.get("use_cuda")),
        "gpu_name": runtime_status.get("gpu_name"),
        "gpu_utilization": runtime_status.get("utilization_percent"),
        "gpu_memory_allocated_mb": runtime_status.get("memory_allocated_mb"),
        "emergency_model_active": emergency_detector.has_model,
    }


def build_signal_plan(
    lanes: list[dict],
    wait_history: dict[int, int],
    government_override: dict | None = None,
) -> dict:
    """Build the signal plan from lane analytics."""
    ranking: list[dict] = []
    total_pedestrians = 0

    override_lane_id = None
    if government_override and government_override.get("lane_id"):
        override_lane_id = int(government_override["lane_id"])

    for lane in lanes:
        wait_bonus = wait_history.get(lane["lane_id"], 0) * 2.4
        density_score = (
            lane["vehicle_count"] * 3.0
            + lane["average_count"] * 1.2
            + lane["occupancy_ratio"] * 80.0
        )
        emergency_bonus = 420.0 if lane["emergency_detected"] else 0.0
        override_bonus = 9999.0 if lane["lane_id"] == override_lane_id else 0.0
        priority_score = round(density_score + wait_bonus + emergency_bonus + override_bonus, 2)

        green_time = 15 + round(
            lane["vehicle_count"] * 1.5
            + lane["occupancy_ratio"] * 20.0
            + wait_bonus * 0.5
        )
        green_time = int(_clamp(green_time, 15, 60))

        if lane["lane_id"] == override_lane_id:
            green_time = 60
        elif lane["emergency_detected"]:
            green_time = int(_clamp(green_time + 15, 30, 60))

        lane["priority_score"] = priority_score
        lane["green_time"] = green_time
        lane["yellow_time"] = 3
        lane["is_override"] = lane["lane_id"] == override_lane_id
        ranking.append(lane)
        total_pedestrians += lane.get("pedestrian_count", 0)

    ranking.sort(
        key=lambda lane: (lane.get("is_override", False), lane["emergency_detected"], lane["priority_score"]),
        reverse=True,
    )
    for order, lane in enumerate(ranking, start=1):
        lane["signal_order"] = order

    cycle_total = sum(lane["green_time"] + lane["yellow_time"] for lane in ranking)
    pedestrian_phase = None
    if total_pedestrians > 0:
        pedestrian_time = int(_clamp(8 + total_pedestrians * 0.8, 8, 18))
        pedestrian_phase = {
            "phase": "pedestrian_crossing",
            "pedestrian_count": total_pedestrians,
            "crossing_time": pedestrian_time,
        }
        cycle_total += pedestrian_time

    priority_lane = ranking[0]
    if priority_lane.get("is_override"):
        decision_text = (
            f"GOVERNMENT OVERRIDE ACTIVE. {priority_lane['lane_name']} has top priority for this cycle."
        )
    elif priority_lane["emergency_detected"]:
        decision_text = (
            f"Emergency override active. {priority_lane['lane_name']} gets the first green signal "
            "with extra clearance time before density scheduling resumes."
        )
    else:
        decision_text = (
            f"Signal order is based on {RUNTIME.model_name} vehicle counts, roadway occupancy, "
            "wait-history balancing, and emergency-priority rules."
        )

    if pedestrian_phase:
        decision_text += (
            f" A pedestrian phase ({pedestrian_phase['crossing_time']}s) is appended for "
            f"{total_pedestrians} detected pedestrian(s)."
        )

    return {
        "lanes": sorted(lanes, key=lambda lane: lane["lane_id"]),
        "signal_sequence": [
            {
                "lane_id": lane["lane_id"],
                "lane_name": lane["lane_name"],
                "signal_order": lane["signal_order"],
                "green_time": lane["green_time"],
                "yellow_time": lane["yellow_time"],
                "is_override": lane.get("is_override", False),
            }
            for lane in ranking
        ],
        "priority_lane": priority_lane["lane_id"],
        "priority_lane_name": priority_lane["lane_name"],
        "cycle_total": cycle_total,
        "decision_text": decision_text,
        "pedestrian_phase": pedestrian_phase,
        "total_pedestrians": total_pedestrians,
        "government_override_active": override_lane_id is not None,
        "override_lane_id": override_lane_id,
    }
