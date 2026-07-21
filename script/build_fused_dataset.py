#!/usr/bin/env python3
"""Build a four-class OBB dataset from cone OBB and PPE AABB datasets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")
FUSED_NAMES = ("cone", "traffic_sign", "person", "vest")
PPE_CLASS_MAP = {0: 2, 2: 3}


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cone-root",
        type=Path,
        default=project_root / "dataset" / "data-26-07-14",
    )
    parser.add_argument(
        "--ppe-root",
        type=Path,
        default=project_root / "dataset" / "person" / "PPE-supplement" / "20260612",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "dataset" / "2026-07-15",
    )
    parser.add_argument(
        "--cone-train-repeat",
        type=int,
        default=3,
        help="Total copies of each cone training image; val/test are never repeated",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of creating space-efficient hard links",
    )
    return parser


def _images(root: Path, split: str) -> list[Path]:
    image_dir = root / "images" / split
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {image_dir}")
    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        duplicates = sorted(stem for stem, count in Counter(stems).items() if count > 1)
        raise ValueError(f"Duplicate image stems in {image_dir}: {duplicates[:10]}")
    return images


def _read_label(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing label for source image: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_obb_line(line: str, path: Path, line_number: int) -> tuple[int, str]:
    fields = line.split()
    if len(fields) != 9:
        raise ValueError(f"{path}:{line_number}: expected class + 8 OBB coordinates")
    try:
        values = [float(field) for field in fields]
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: non-numeric OBB label") from exc
    class_id = int(values[0])
    if values[0] != class_id or class_id not in (0, 1):
        raise ValueError(f"{path}:{line_number}: unsupported cone class {values[0]:g}")
    if any(not 0.0 <= value <= 1.0 for value in values[1:]):
        raise ValueError(f"{path}:{line_number}: OBB coordinate outside [0, 1]")
    return class_id, line


def _convert_ppe_line(line: str, path: Path, line_number: int) -> tuple[int, str] | None:
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"{path}:{line_number}: expected class + 4 AABB values")
    try:
        values = [float(field) for field in fields]
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: non-numeric AABB label") from exc
    source_class = int(values[0])
    if values[0] != source_class:
        raise ValueError(f"{path}:{line_number}: class ID must be an integer")
    target_class = PPE_CLASS_MAP.get(source_class)
    if target_class is None:
        return None

    center_x, center_y, width, height = values[1:]
    if width <= 0.0 or height <= 0.0:
        raise ValueError(f"{path}:{line_number}: box width and height must be positive")
    x1 = max(0.0, center_x - width / 2.0)
    y1 = max(0.0, center_y - height / 2.0)
    x2 = min(1.0, center_x + width / 2.0)
    y2 = min(1.0, center_y + height / 2.0)
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"{path}:{line_number}: box is outside the normalized image area")
    coordinates = (x1, y1, x2, y1, x2, y2, x1, y2)
    converted = " ".join([str(target_class), *(f"{value:.8f}" for value in coordinates)])
    return target_class, converted


def _place_image(source: Path, destination: Path, copy_images: bool) -> None:
    if copy_images:
        shutil.copy2(source, destination)
    else:
        os.link(source, destination)


def _write_sample(
    source_image: Path,
    output_root: Path,
    split: str,
    destination_stem: str,
    label_lines: list[str],
    copy_images: bool,
) -> None:
    destination_image = output_root / "images" / split / f"{destination_stem}{source_image.suffix.lower()}"
    destination_label = output_root / "labels" / split / f"{destination_stem}.txt"
    _place_image(source_image, destination_image, copy_images)
    destination_label.write_text(
        "\n".join(label_lines) + ("\n" if label_lines else ""),
        encoding="utf-8",
    )


def build_dataset(
    cone_root: Path,
    ppe_root: Path,
    output: Path,
    cone_train_repeat: int,
    copy_images: bool,
) -> dict[str, object]:
    cone_root = cone_root.resolve()
    ppe_root = ppe_root.resolve()
    output = output.resolve()
    if cone_train_repeat < 1:
        raise ValueError("--cone-train-repeat must be at least 1")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")

    temporary = output.parent / f".{output.name}.building"
    if temporary.exists():
        raise FileExistsError(f"Remove or inspect interrupted build directory first: {temporary}")
    for split in SPLITS:
        (temporary / "images" / split).mkdir(parents=True)
        (temporary / "labels" / split).mkdir(parents=True)

    image_counts: Counter[tuple[str, str]] = Counter()
    object_counts: Counter[tuple[str, int]] = Counter()
    empty_ppe_counts: Counter[str] = Counter()
    try:
        for split in SPLITS:
            repeat = cone_train_repeat if split == "train" else 1
            for image in _images(cone_root, split):
                source_label = cone_root / "labels" / split / f"{image.stem}.txt"
                source_lines = _read_label(source_label)
                validated: list[str] = []
                classes: list[int] = []
                for line_number, line in enumerate(source_lines, start=1):
                    class_id, valid_line = _validate_obb_line(line, source_label, line_number)
                    classes.append(class_id)
                    validated.append(valid_line)
                for repeat_index in range(repeat):
                    destination_stem = f"cone_r{repeat_index}__{image.stem}"
                    _write_sample(
                        image, temporary, split, destination_stem,
                        validated, copy_images,
                    )
                    image_counts[(split, "cone_source")] += 1
                    for class_id in classes:
                        object_counts[(split, class_id)] += 1

            for image in _images(ppe_root, split):
                source_label = ppe_root / "labels" / split / f"{image.stem}.txt"
                source_lines = _read_label(source_label)
                converted: list[str] = []
                for line_number, line in enumerate(source_lines, start=1):
                    result = _convert_ppe_line(line, source_label, line_number)
                    if result is None:
                        continue
                    class_id, converted_line = result
                    object_counts[(split, class_id)] += 1
                    converted.append(converted_line)
                if not converted:
                    empty_ppe_counts[split] += 1
                _write_sample(
                    image, temporary, split, f"ppe__{image.stem}",
                    converted, copy_images,
                )
                image_counts[(split, "ppe_source")] += 1

        config = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(FUSED_NAMES),
            "names": list(FUSED_NAMES),
        }
        (temporary / "data.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        manifest: dict[str, object] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": {"cone_obb": str(cone_root), "ppe_aabb": str(ppe_root)},
            "classes": {str(index): name for index, name in enumerate(FUSED_NAMES)},
            "ppe_class_mapping": {"0_person": 2, "2_vest": 3},
            "ignored_ppe_classes": [1, 3, 4, 5, 6, 7],
            "cone_train_repeat": cone_train_repeat,
            "image_storage": "copy" if copy_images else "hardlink",
            "images": {
                split: {
                    "cone": image_counts[(split, "cone_source")],
                    "ppe": image_counts[(split, "ppe_source")],
                    "total": image_counts[(split, "cone_source")] + image_counts[(split, "ppe_source")],
                    "empty_ppe_labels": empty_ppe_counts[split],
                }
                for split in SPLITS
            },
            "objects": {
                split: {FUSED_NAMES[class_id]: object_counts[(split, class_id)] for class_id in range(len(FUSED_NAMES))}
                for split in SPLITS
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            output.rmdir()
        temporary.replace(output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_dataset(
        args.cone_root,
        args.ppe_root,
        args.output,
        args.cone_train_repeat,
        args.copy_images,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Fused dataset: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
