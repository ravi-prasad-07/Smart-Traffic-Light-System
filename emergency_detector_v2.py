"""Emergency-vehicle detector with model, color, flash, and temporal fusion."""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
import torch

from runtime_config import configure_torch_runtime, get_runtime_config

try:
    from scipy.fft import rfft, rfftfreq  # type: ignore[import-untyped]
except ImportError:
    rfft = None  # type: ignore[assignment]
    rfftfreq = None  # type: ignore[assignment]


RUNTIME = configure_torch_runtime()
LOGGER = logging.getLogger("traffic.emergency")

EMERGENCY_CLASSES = {0: "ambulance", 1: "fire_truck"}
EMERGENCY_CLASS_THRESHOLDS = {"ambulance": 0.28, "fire_truck": 0.34}
FILENAME_HINTS = ("ambulance", "fire", "brigade", "rescue", "emergency")
FLASH_BUFFER_SIZE = 12
FLASH_FREQ_LOW = 0.8
FLASH_FREQ_HIGH = 4.0
MIN_VEHICLE_AREA_FRAC = 0.004
CANDIDATE_LABELS = {"car", "truck", "bus"}


@dataclass
class EmergencyTrackState:
    """Per-track temporal evidence for emergency detection."""

    key: str
    label: str = ""
    ema_confidence: float = 0.0
    best_confidence: float = 0.0
    consecutive_hits: int = 0
    confirmed: bool = False
    last_reason: str = ""
    last_seen_frame: int = 0
    flash_signal: collections.deque[float] = field(
        default_factory=lambda: collections.deque(maxlen=FLASH_BUFFER_SIZE)
    )


@lru_cache(maxsize=1)
def _load_emergency_model():
    """Load the custom emergency model if weights are available."""
    runtime = get_runtime_config()
    model_path = runtime.emergency_model_path
    if not model_path.exists():
        return None

    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except Exception:
        return None

    model = YOLO(str(model_path))
    if hasattr(model, "to"):
        model.to(runtime.device)
    if hasattr(model, "model"):
        model.model.eval()
    return model


def _aspect_ratio_ok(w: int, h: int) -> bool:
    if h <= 0:
        return False
    aspect_ratio = w / float(h)
    return 0.9 <= aspect_ratio <= 5.5


def _colour_stats(region: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    total = float(region.shape[0] * region.shape[1]) + 1e-6

    red_lo = cv2.inRange(hsv, (0, 90, 65), (14, 255, 255))
    red_hi = cv2.inRange(hsv, (165, 90, 65), (179, 255, 255))
    red = cv2.bitwise_or(red_lo, red_hi)

    white = cv2.inRange(hsv, (0, 0, 160), (179, 70, 255))
    blue = cv2.inRange(hsv, (90, 80, 70), (135, 255, 255))
    bright_red = cv2.inRange(hsv, (0, 130, 180), (14, 255, 255))
    bright_blue = cv2.inRange(hsv, (95, 130, 180), (135, 255, 255))

    return {
        "red": float(np.count_nonzero(red)) / total,
        "white": float(np.count_nonzero(white)) / total,
        "blue": float(np.count_nonzero(blue)) / total,
        "bright_red": float(np.count_nonzero(bright_red)) / total,
        "bright_blue": float(np.count_nonzero(bright_blue)) / total,
    }


def _expand_box(
    x: int,
    y: int,
    w: int,
    h: int,
    frame_shape: tuple[int, int, int],
    pad: float = 0.12,
) -> tuple[int, int, int, int]:
    pad_x = int(round(w * pad))
    pad_y = int(round(h * pad))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_shape[1], x + w + pad_x)
    y2 = min(frame_shape[0], y + h + pad_y)
    return x1, y1, x2, y2


class EmergencyDetector:
    """High-priority emergency detector with temporal confirmation."""

    def __init__(self, fps: float = 6.0):
        self._model = _load_emergency_model()
        self._fps = max(1.0, fps)
        self._frame_index = 0
        self._states: dict[str, EmergencyTrackState] = {}

    @property
    def has_model(self) -> bool:
        return self._model is not None

    def process_frame(
        self,
        frame: np.ndarray,
        general_detections: list[dict],
        filename: str = "",
    ) -> tuple[bool, str, float]:
        self._frame_index += 1
        self._prune_states()

        model_signals = self._tier_model(frame, general_detections)
        best_detection: tuple[str, str, float] | None = None

        for det in general_detections:
            if not self._is_candidate(frame, det):
                continue

            key = self._track_key(det)
            state = self._states.setdefault(key, EmergencyTrackState(key=key))
            x, y, w, h = det["box"]
            x1, y1, x2, y2 = _expand_box(x, y, w, h, frame.shape)
            region = frame[y1:y2, x1:x2]
            if region.size == 0:
                continue

            color_signal = self._tier_colour(region, det["label"], w, h)
            flash_signal = self._tier_flash(region, state)
            model_signal = model_signals.get(key, {"score": 0.0, "label": "", "reason": ""})

            fused_score, fused_label, fused_reason = self._fuse_signals(
                det,
                state,
                model_signal=model_signal,
                color_signal=color_signal,
                flash_signal=flash_signal,
            )

            state.last_seen_frame = self._frame_index
            state.last_reason = fused_reason
            state.best_confidence = max(state.best_confidence, fused_score)
            if state.ema_confidence == 0.0:
                state.ema_confidence = fused_score
            else:
                state.ema_confidence = 0.62 * state.ema_confidence + 0.38 * fused_score

            if fused_label:
                state.label = fused_label
            if fused_score >= 0.45:
                state.consecutive_hits += 1
            else:
                state.consecutive_hits = max(0, state.consecutive_hits - 1)

            just_confirmed = (
                fused_label
                and (
                    fused_score >= 0.72
                    or state.best_confidence >= 0.82
                    or (state.ema_confidence >= 0.52 and state.consecutive_hits >= 2)
                    or model_signal["score"] >= 0.65
                )
            )
            if just_confirmed and not state.confirmed:
                LOGGER.info(
                    "Emergency vehicle confirmed: label=%s track=%s conf=%.3f reason=%s",
                    state.label or fused_label,
                    key,
                    max(fused_score, state.ema_confidence),
                    fused_reason,
                )
            state.confirmed = state.confirmed or just_confirmed

            det["emergency_score"] = round(fused_score, 3)
            det["emergency_label"] = fused_label
            det["emergency_confirmed"] = state.confirmed

            if state.confirmed:
                score = max(fused_score, state.ema_confidence, state.best_confidence)
                candidate = (state.label or fused_label, fused_reason, score)
                if best_detection is None or candidate[2] > best_detection[2]:
                    best_detection = candidate

        if best_detection:
            label, reason, score = best_detection
            return True, f"{label}: {reason}", min(1.0, score)

        recent_confirmed = [
            state
            for state in self._states.values()
            if state.confirmed and (self._frame_index - state.last_seen_frame) <= 2
        ]
        if recent_confirmed:
            state = max(recent_confirmed, key=lambda item: max(item.best_confidence, item.ema_confidence))
            return True, f"{state.label}: {state.last_reason}", min(1.0, max(state.best_confidence, state.ema_confidence))

        lowered = filename.lower()
        if any(keyword in lowered for keyword in FILENAME_HINTS):
            return True, "Filename contains an emergency-vehicle hint.", 0.35

        return False, "", 0.0

    def _is_candidate(self, frame: np.ndarray, det: dict) -> bool:
        if det.get("label") not in CANDIDATE_LABELS:
            return False
        x, y, w, h = det["box"]
        if w <= 0 or h <= 0 or not _aspect_ratio_ok(w, h):
            return False
        frame_area = float(frame.shape[0] * frame.shape[1]) + 1e-6
        return ((w * h) / frame_area) >= MIN_VEHICLE_AREA_FRAC

    def _tier_model(self, frame: np.ndarray, general_detections: list[dict]) -> dict[str, dict]:
        if self._model is None:
            return {}

        crops: list[np.ndarray] = []
        keys: list[str] = []

        for det in general_detections:
            if not self._is_candidate(frame, det):
                continue
            x, y, w, h = det["box"]
            x1, y1, x2, y2 = _expand_box(x, y, w, h, frame.shape)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crops.append(crop)
            keys.append(self._track_key(det))

        if not crops:
            return {}

        with torch.inference_mode():
            results = self._model.predict(
                source=crops,
                imgsz=min(RUNTIME.imgsz, 512),
                conf=0.18,
                iou=0.45,
                device=RUNTIME.device,
                half=RUNTIME.use_half,
                batch=min(len(crops), max(1, min(RUNTIME.batch_size, 8))),
                verbose=False,
                max_det=4,
            )

        signals: dict[str, dict] = {}
        for key, result in zip(keys, results):
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            self._verify_result_device(result)

            data = boxes.data.detach().cpu().numpy()
            best_signal = {"score": 0.0, "label": "", "reason": ""}
            for row in data:
                _, _, _, _, raw_conf, raw_cls = row[:6]
                label = EMERGENCY_CLASSES.get(int(raw_cls), "")
                if not label:
                    continue
                threshold = EMERGENCY_CLASS_THRESHOLDS.get(label, 0.32)
                score = min(1.0, float(raw_conf) + 0.10)
                if float(raw_conf) < threshold:
                    score *= 0.60
                if score <= best_signal["score"]:
                    continue
                best_signal = {
                    "score": score,
                    "label": label,
                    "reason": f"Emergency model detected {label} ({float(raw_conf):.0%}).",
                }
            if best_signal["score"] > 0.0:
                signals[key] = best_signal

        return signals

    def _tier_colour(self, region: np.ndarray, label: str, w: int, h: int) -> dict:
        if label not in CANDIDATE_LABELS or not _aspect_ratio_ok(w, h):
            return {"score": 0.0, "label": "", "reason": ""}

        top_band = region[: max(1, int(region.shape[0] * 0.25)), :]
        mid_band = region[int(region.shape[0] * 0.35): int(region.shape[0] * 0.72), :]

        whole = _colour_stats(region)
        top = _colour_stats(top_band if top_band.size else region)
        mid = _colour_stats(mid_band if mid_band.size else region)
        top_light = top["bright_red"] + top["bright_blue"]

        ambulance_score = 0.0
        if whole["white"] > 0.18:
            ambulance_score += min(0.40, (whole["white"] - 0.18) * 1.8)
        if mid["red"] > 0.02:
            ambulance_score += min(0.22, (mid["red"] - 0.02) * 4.8)
        if top_light > 0.008:
            ambulance_score += min(0.25, top_light * 10.0)
        if whole["blue"] > 0.015:
            ambulance_score += min(0.08, whole["blue"] * 2.4)
        if label == "car":
            ambulance_score += 0.04

        fire_score = 0.0
        if whole["red"] > 0.12:
            fire_score += min(0.52, (whole["red"] - 0.12) * 3.0)
        if whole["white"] > 0.04:
            fire_score += min(0.10, (whole["white"] - 0.04) * 1.2)
        if top_light > 0.008:
            fire_score += min(0.18, top_light * 7.0)
        if label in {"truck", "bus"}:
            fire_score += 0.06

        if ambulance_score >= fire_score and ambulance_score >= 0.18:
            return {
                "score": min(1.0, ambulance_score),
                "label": "ambulance",
                "reason": (
                    f"Ambulance visual pattern: white={whole['white']:.0%}, "
                    f"red={mid['red']:.0%}, roof_light={top_light:.0%}."
                ),
            }
        if fire_score >= 0.20:
            return {
                "score": min(1.0, fire_score),
                "label": "fire_truck",
                "reason": (
                    f"Fire-truck visual pattern: red={whole['red']:.0%}, "
                    f"white={whole['white']:.0%}, roof_light={top_light:.0%}."
                ),
            }
        return {"score": 0.0, "label": "", "reason": ""}

    def _tier_flash(self, region: np.ndarray, state: EmergencyTrackState) -> dict:
        top_band = region[: max(1, int(region.shape[0] * 0.22)), :]
        if top_band.size == 0:
            return {"score": 0.0, "reason": ""}

        stats = _colour_stats(top_band)
        flash_signal = stats["bright_red"] + stats["bright_blue"]
        state.flash_signal.append(flash_signal)

        if len(state.flash_signal) < 4:
            return {"score": 0.0, "reason": ""}

        signal = np.asarray(state.flash_signal, dtype=np.float32)
        signal_std = float(np.std(signal))
        peak_to_peak = float(np.ptp(signal))
        if signal_std < 0.002 and peak_to_peak < 0.007:
            return {"score": 0.0, "reason": ""}

        if rfft is not None and rfftfreq is not None and len(signal) >= 6:
            centered = signal - float(np.mean(signal))
            spectrum = np.abs(rfft(centered))
            freqs = rfftfreq(len(centered), d=1.0 / self._fps)
            if len(spectrum) > 1:
                dominant_idx = int(np.argmax(spectrum[1:])) + 1
                dominant_freq = float(freqs[dominant_idx])
                dominant_power = float(spectrum[dominant_idx])
                total_power = float(np.sum(spectrum[1:])) + 1e-6
                power_ratio = dominant_power / total_power
                if FLASH_FREQ_LOW <= dominant_freq <= FLASH_FREQ_HIGH and power_ratio > 0.25:
                    return {
                        "score": min(0.75, 0.32 + power_ratio),
                        "reason": f"Beacon flash pattern detected at {dominant_freq:.1f}Hz.",
                    }

        return {
            "score": min(0.55, 0.18 + signal_std * 38.0 + peak_to_peak * 10.0),
            "reason": f"Roof-light oscillation detected (std={signal_std:.3f}, span={peak_to_peak:.3f}).",
        }

    def _fuse_signals(
        self,
        det: dict,
        state: EmergencyTrackState,
        model_signal: dict,
        color_signal: dict,
        flash_signal: dict,
    ) -> tuple[float, str, str]:
        label_votes: dict[str, float] = {}
        if model_signal["label"]:
            label_votes[model_signal["label"]] = label_votes.get(model_signal["label"], 0.0) + (0.55 + model_signal["score"] * 0.45)
        if color_signal["label"]:
            label_votes[color_signal["label"]] = label_votes.get(color_signal["label"], 0.0) + (0.30 + color_signal["score"] * 0.50)
        if flash_signal["score"] > 0.22:
            flash_label = color_signal["label"] or model_signal["label"] or state.label
            if flash_label:
                label_votes[flash_label] = label_votes.get(flash_label, 0.0) + 0.18
        if not label_votes and state.label:
            label_votes[state.label] = 0.15

        final_label = max(label_votes, key=label_votes.get) if label_votes else ""

        combined = max(
            model_signal["score"],
            color_signal["score"] * 0.94,
            flash_signal["score"] * 0.82,
        )
        if model_signal["score"] > 0.30 and color_signal["score"] > 0.22:
            combined += 0.16
        if color_signal["score"] > 0.20 and flash_signal["score"] > 0.20:
            combined += 0.12
        if state.confirmed:
            combined += 0.08
        if state.consecutive_hits >= 1 and (model_signal["score"] > 0.20 or color_signal["score"] > 0.20):
            combined += 0.05

        if final_label == "ambulance" and det.get("label") == "car":
            combined += 0.04
        if final_label == "fire_truck" and det.get("label") in {"truck", "bus"}:
            combined += 0.04
        if not final_label:
            combined *= 0.40

        reason_parts = [
            signal["reason"]
            for signal in (model_signal, color_signal, flash_signal)
            if signal.get("reason") and signal.get("score", 0.0) >= 0.18
        ]
        reason = " | ".join(reason_parts) if reason_parts else "Temporal emergency ensemble activated."
        return min(1.0, combined), final_label, reason

    def _prune_states(self) -> None:
        stale_after = max(8, int(round(self._fps * 2.5)))
        stale_keys = [
            key
            for key, state in self._states.items()
            if (self._frame_index - state.last_seen_frame) > stale_after
        ]
        for key in stale_keys:
            self._states.pop(key, None)

    @staticmethod
    def _track_key(det: dict) -> str:
        track_id = det.get("track_id")
        if track_id is not None:
            return f"track:{track_id}"
        x, y, w, h = det["box"]
        return f"grid:{(x + w // 2) // 80}:{(y + h // 2) // 80}"

    @staticmethod
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
            raise RuntimeError("Emergency model post-processing moved off CUDA unexpectedly.")
