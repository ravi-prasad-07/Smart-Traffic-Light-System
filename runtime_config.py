"""Shared runtime and GPU configuration for the traffic system."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch


BASE_DIR = Path(__file__).resolve().parent
YOLO_CONFIG_DIR = BASE_DIR / ".yolo"
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _normalise_path(raw_value: str | None, default_name: str) -> Path:
    candidate = Path(raw_value) if raw_value else BASE_DIR / default_name
    if not candidate.is_absolute():
        candidate = BASE_DIR / candidate
    return candidate


def _candidate_model_paths(variant: str) -> list[Path]:
    return [
        BASE_DIR / f"yolov8{variant}.engine",
        BASE_DIR / f"yolov8{variant}.torchscript",
        BASE_DIR / f"yolov8{variant}.onnx",
        BASE_DIR / f"yolov8{variant}.pt",
    ]


def _resolve_general_model_path() -> Path:
    explicit_model = os.getenv("TRAFFIC_YOLO_MODEL")
    if explicit_model:
        return _normalise_path(explicit_model, "yolov8m.pt")

    preferred_variant = os.getenv("TRAFFIC_MODEL_VARIANT", "m").strip().lower()
    valid_variants = {"m", "l", "s", "n", "x"}
    if preferred_variant not in valid_variants:
        preferred_variant = "m"

    variant_order = [preferred_variant]
    for fallback_variant in ("m", "l", "s", "n", "x"):
        if fallback_variant not in variant_order:
            variant_order.append(fallback_variant)

    for variant in variant_order:
        for candidate in _candidate_model_paths(variant):
            if candidate.exists():
                return candidate

    return BASE_DIR / f"yolov8{preferred_variant}.pt"


def _detect_backend(model_path: Path) -> str:
    suffix = model_path.suffix.lower()
    if suffix == ".engine":
        return "tensorrt"
    if suffix == ".torchscript":
        return "torchscript"
    if suffix == ".onnx":
        return "onnx"
    return "pytorch"


def _format_model_name(model_path: Path) -> str:
    stem = model_path.stem.replace("_", " ").replace("-", " ").strip()
    return stem or model_path.name


@dataclass(frozen=True)
class RuntimeConfig:
    strict_cuda: bool
    use_cuda: bool
    device: str
    device_index: int
    precision: str
    use_half: bool
    imgsz: int
    batch_size: int
    yolo_conf: float
    yolo_iou: float
    yolo_max_det: int
    sample_fps: float
    max_samples: int
    torch_compile: bool
    general_model_path: Path
    emergency_model_path: Path
    backend: str
    model_name: str
    warmup_runs: int
    log_level: str


@lru_cache(maxsize=1)
def get_runtime_config() -> RuntimeConfig:
    strict_cuda = _env_flag("TRAFFIC_FORCE_CUDA", True)
    use_cuda = torch.cuda.is_available()
    if strict_cuda and not use_cuda:
        raise RuntimeError(
            "CUDA is required for this deployment but PyTorch did not detect a CUDA GPU. "
            "Set TRAFFIC_FORCE_CUDA=0 to allow CPU fallback."
        )

    general_model_path = _resolve_general_model_path()
    emergency_model_path = _normalise_path(os.getenv("TRAFFIC_EMERGENCY_MODEL"), "emergency_yolov8s.pt")
    backend = _detect_backend(general_model_path)
    device = "cuda:0" if use_cuda else "cpu"

    config = RuntimeConfig(
        strict_cuda=strict_cuda,
        use_cuda=use_cuda,
        device=device,
        device_index=0,
        precision="fp16" if use_cuda and _env_flag("TRAFFIC_FP16", True) else "fp32",
        use_half=use_cuda and _env_flag("TRAFFIC_FP16", True),
        imgsz=_env_int("TRAFFIC_IMGSZ", 640),
        batch_size=max(1, _env_int("TRAFFIC_BATCH_SIZE", 12)),
        yolo_conf=_env_float("TRAFFIC_YOLO_CONF", 0.20),
        yolo_iou=_env_float("TRAFFIC_YOLO_IOU", 0.45),
        yolo_max_det=max(20, _env_int("TRAFFIC_MAX_DET", 200)),
        sample_fps=max(1.0, _env_float("TRAFFIC_SAMPLE_FPS", 6.0)),
        max_samples=max(24, _env_int("TRAFFIC_MAX_SAMPLES", 240)),
        torch_compile=use_cuda and _env_flag("TRAFFIC_TORCH_COMPILE", False) and backend == "pytorch",
        general_model_path=general_model_path,
        emergency_model_path=emergency_model_path,
        backend=backend,
        model_name=_format_model_name(general_model_path),
        warmup_runs=max(1, _env_int("TRAFFIC_WARMUP_RUNS", 1)),
        log_level=os.getenv("TRAFFIC_LOG_LEVEL", "INFO").upper(),
    )
    return config


def configure_torch_runtime() -> RuntimeConfig:
    config = get_runtime_config()

    logging.getLogger("traffic").setLevel(getattr(logging, config.log_level, logging.INFO))
    torch.set_float32_matmul_precision("high")

    if config.use_cuda:
        torch.cuda.set_device(config.device_index)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return config


def get_runtime_status() -> dict:
    config = get_runtime_config()
    status = {
        "strict_cuda": config.strict_cuda,
        "device": config.device,
        "use_cuda": config.use_cuda,
        "precision": config.precision,
        "backend": config.backend,
        "model_path": str(config.general_model_path),
        "model_name": config.model_name,
        "preferred_model_variant": os.getenv("TRAFFIC_MODEL_VARIANT", "m").strip().lower() or "m",
        "emergency_model_path": str(config.emergency_model_path),
        "batch_size": config.batch_size,
        "imgsz": config.imgsz,
        "sample_fps": config.sample_fps,
    }

    if not config.use_cuda:
        return status

    device_index = config.device_index
    status.update(
        {
            "gpu_name": torch.cuda.get_device_name(device_index),
            "cuda_version": torch.version.cuda,
            "capability": list(torch.cuda.get_device_capability(device_index)),
            "memory_allocated_mb": round(torch.cuda.memory_allocated(device_index) / (1024 ** 2), 2),
            "memory_reserved_mb": round(torch.cuda.memory_reserved(device_index) / (1024 ** 2), 2),
            "memory_total_mb": round(torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 2), 2),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    )

    utilization = None
    if hasattr(torch.cuda, "utilization"):
        try:
            utilization = float(torch.cuda.utilization(device_index))
        except Exception:
            utilization = None
    status["utilization_percent"] = utilization
    return status
