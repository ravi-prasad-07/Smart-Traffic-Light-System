"""Lightweight vectorized IoU tracker with box smoothing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import lap  # type: ignore[import-untyped]

    _HAS_LAP = True
except ImportError:
    _HAS_LAP = False


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (x, y, w, h) boxes to (x1, y1, x2, y2)."""
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    out = boxes.astype(np.float32, copy=True)
    out[:, 2] = out[:, 0] + out[:, 2]
    out[:, 3] = out[:, 1] + out[:, 3]
    return out


def _xyxy_to_xywh(box: np.ndarray) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box.tolist()
    return (
        int(round(x1)),
        int(round(y1)),
        int(round(max(0.0, x2 - x1))),
        int(round(max(0.0, y2 - y1))),
    )


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU between two sets of boxes using vectorized NumPy."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = np.clip(bottom_right - top_left, a_min=0.0, a_max=None)

    inter = wh[..., 0] * wh[..., 1]
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))[:, None]
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))[None, :]
    union = np.clip(area_a + area_b - inter, a_min=1e-6, a_max=None)
    return inter / union


def _linear_assignment(cost_matrix: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Solve the linear assignment problem with LAP or a greedy fallback."""
    if cost_matrix.size == 0:
        return []

    if _HAS_LAP and cost_matrix.shape[0] > 0 and cost_matrix.shape[1] > 0:
        padded = cost_matrix.copy()
        padded[padded > threshold] = threshold + 1.0
        try:
            _, rows, _ = lap.lapjv(padded, extend_cost=True)
            return [
                (row_idx, col_idx)
                for row_idx, col_idx in enumerate(rows)
                if col_idx >= 0 and cost_matrix[row_idx, col_idx] <= threshold
            ]
        except Exception:
            pass

    indices = np.argwhere(cost_matrix <= threshold)
    if len(indices) == 0:
        return []

    costs = cost_matrix[indices[:, 0], indices[:, 1]]
    order = np.argsort(costs)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    matches: list[tuple[int, int]] = []

    for idx in order:
        row_idx = int(indices[idx, 0])
        col_idx = int(indices[idx, 1])
        if row_idx in used_rows or col_idx in used_cols:
            continue
        matches.append((row_idx, col_idx))
        used_rows.add(row_idx)
        used_cols.add(col_idx)

    return matches


@dataclass
class Track:
    """A tracked object with a stable ID and smoothed box."""

    track_id: int
    box_xyxy: np.ndarray
    smoothed_box_xyxy: np.ndarray
    label: str = "vehicle"
    is_pedestrian: bool = False
    confidence: float = 0.0
    hits: int = 1
    misses: int = 0
    active: bool = True

    @property
    def stable_box_xywh(self) -> tuple[int, int, int, int]:
        return _xyxy_to_xywh(self.smoothed_box_xyxy)


class VehicleTracker:
    """IoU-based tracker with lightweight temporal smoothing."""

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_misses: int = 3,
        min_hits: int = 2,
        smoothing_alpha: float = 0.35,
    ):
        self._iou_threshold = iou_threshold
        self._max_misses = max_misses
        self._min_hits = min_hits
        self._smoothing_alpha = float(np.clip(smoothing_alpha, 0.05, 1.0))
        self._next_id = 1
        self._tracks: list[Track] = []

    def update(self, detections: list[dict]) -> list[Track]:
        """Update tracks and write stable tracking metadata into detections."""
        if not detections:
            for track in self._tracks:
                track.misses += 1
                if track.misses > self._max_misses:
                    track.active = False
            return [track for track in self._tracks if track.active]

        det_boxes_xywh = np.asarray([det["box"] for det in detections], dtype=np.float32)
        det_boxes_xyxy = _xywh_to_xyxy(det_boxes_xywh)
        active_tracks = [track for track in self._tracks if track.active]

        if not active_tracks:
            for det_idx, det in enumerate(detections):
                track = self._create_track(det_boxes_xyxy[det_idx], det)
                self._write_track_metadata(det, track)
            return [track for track in self._tracks if track.active]

        track_boxes = np.asarray([track.smoothed_box_xyxy for track in active_tracks], dtype=np.float32)
        iou = _iou_matrix(track_boxes, det_boxes_xyxy)
        cost = 1.0 - iou

        matches = _linear_assignment(cost, 1.0 - self._iou_threshold)
        matched_track_indices = {track_idx for track_idx, _ in matches}
        matched_det_indices = {det_idx for _, det_idx in matches}

        for track_idx, det_idx in matches:
            track = active_tracks[track_idx]
            det = detections[det_idx]
            measured_box = det_boxes_xyxy[det_idx]
            track.box_xyxy = measured_box
            track.smoothed_box_xyxy = (
                self._smoothing_alpha * measured_box
                + (1.0 - self._smoothing_alpha) * track.smoothed_box_xyxy
            )
            track.label = det["label"]
            track.is_pedestrian = det.get("is_pedestrian", False)
            track.confidence = float(det["confidence"])
            track.hits += 1
            track.misses = 0
            self._write_track_metadata(det, track)

        for idx, track in enumerate(active_tracks):
            if idx in matched_track_indices:
                continue
            track.misses += 1
            if track.misses > self._max_misses:
                track.active = False

        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_indices:
                continue
            track = self._create_track(det_boxes_xyxy[det_idx], det)
            self._write_track_metadata(det, track)

        return [track for track in self._tracks if track.active]

    def get_unique_vehicle_count(self) -> int:
        return sum(1 for track in self._tracks if track.hits >= self._min_hits and not track.is_pedestrian)

    def get_unique_pedestrian_count(self) -> int:
        return sum(1 for track in self._tracks if track.hits >= self._min_hits and track.is_pedestrian)

    def get_all_confirmed_tracks(self) -> list[Track]:
        return [track for track in self._tracks if track.hits >= self._min_hits]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def _create_track(self, box_xyxy: np.ndarray, det: dict) -> Track:
        stable_box = box_xyxy.astype(np.float32, copy=True)
        track = Track(
            track_id=self._next_id,
            box_xyxy=box_xyxy.astype(np.float32, copy=True),
            smoothed_box_xyxy=stable_box,
            label=det["label"],
            is_pedestrian=det.get("is_pedestrian", False),
            confidence=float(det["confidence"]),
        )
        self._next_id += 1
        self._tracks.append(track)
        return track

    @staticmethod
    def _write_track_metadata(det: dict, track: Track) -> None:
        det["track_id"] = track.track_id
        det["track_hits"] = track.hits
        det["stable_box"] = track.stable_box_xywh
