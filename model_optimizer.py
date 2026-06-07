"""Utilities to export YOLO models to TorchScript or TensorRT."""

from __future__ import annotations

import argparse
from pathlib import Path

from runtime_config import configure_torch_runtime, get_runtime_config


def export_model(
    model_path: str,
    export_format: str = "torchscript",
    imgsz: int | None = None,
    half: bool | None = None,
) -> str:
    """Export a YOLO model to a faster deployment format."""
    runtime = configure_torch_runtime()

    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except Exception as exc:  # pragma: no cover - dependency surfaced to user
        raise RuntimeError("Ultralytics is required to export models.") from exc

    model = YOLO(model_path)
    imgsz = imgsz or runtime.imgsz
    half = runtime.use_half if half is None else half
    device_arg = 0 if runtime.use_cuda else "cpu"

    exported_path = model.export(
        format=export_format,
        imgsz=imgsz,
        half=half,
        device=device_arg,
        simplify=False,
    )
    return str(Path(exported_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a YOLO model for faster deployment.")
    parser.add_argument("--model", default=str(get_runtime_config().general_model_path), help="Path to the YOLO model.")
    parser.add_argument(
        "--format",
        dest="export_format",
        default="torchscript",
        choices=("torchscript", "engine"),
        help="Target export format. 'engine' requires TensorRT to be installed.",
    )
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size.")
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Export in FP32 instead of the runtime default FP16 mode.",
    )
    args = parser.parse_args()

    output_path = export_model(
        model_path=args.model,
        export_format=args.export_format,
        imgsz=args.imgsz,
        half=not args.fp32,
    )
    print(output_path)


if __name__ == "__main__":
    main()
