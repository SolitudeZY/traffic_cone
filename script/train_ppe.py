#!/usr/bin/env python3
"""Train a lightweight PPE detector for person and reflective-vest suppression."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml



def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=project_root / "dataset" / "person" / "PPE-supplement" / "20260612" / "data.yaml",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Initial weights or model YAML")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="For example 0, 0,1, cpu, or mps")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true", help="Cache training images in RAM")
    parser.add_argument("--project", type=Path, default=project_root / "script" / "runs")
    parser.add_argument("--name", default="ppe_person_detector")
    parser.add_argument(
        "--resume",
        type=Path,
        default=False,
        help="Resume from a last.pt checkpoint path",
    )
    return parser


def require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependency. Run: pip install -r script/requirements.txt"
        ) from exc
    return YOLO


def create_resolved_data_config(data_path: Path, output_dir: Path) -> Path:
    """Write a runtime YAML whose dataset root is independent of the current directory."""
    config = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    configured_root = data_path.parent.resolve()
    config["path"] = str(configured_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "ppe_data_resolved.yaml"
    resolved_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return resolved_path


def main() -> int:
    args = build_parser().parse_args()
    data_path = args.data.resolve()
    if not data_path.is_file():
        raise SystemExit(f"Dataset config does not exist: {data_path}")

    YOLO = require_ultralytics()
    if args.resume and not args.resume.is_file():
        raise SystemExit(f"Resume checkpoint does not exist: {args.resume}")
    model = YOLO(str(args.resume.resolve()) if args.resume else args.model)
    resolved_data_path = create_resolved_data_config(data_path, args.project.resolve())
    train_args = {
        "data": str(resolved_data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "patience": args.patience,
        "seed": args.seed,
        "cache": args.cache,
        "project": str(args.project.resolve()),
        "name": args.name,
        "pretrained": True,
        "plots": True,
        "amp": True,
        "close_mosaic": 10,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.4,
        "fliplr": 0.5,
    }
    if args.device is not None:
        train_args["device"] = args.device
    if args.resume:
        train_args["resume"] = True

    print(f"Training PPE model {args.model} on {data_path}")
    results = model.train(**train_args)
    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir:
        print(f"Training outputs: {save_dir}")
        print(f"PPE best weights: {Path(save_dir) / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
