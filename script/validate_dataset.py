#!/usr/bin/env python3
"""Validate YOLO OBB labels and optionally clip tiny boundary overflows."""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Issue:
    code: str
    path: Path
    message: str
    line: int | None = None

    def format(self) -> str:
        location = f"{self.path}:{self.line}" if self.line else str(self.path)
        return f"[{self.code}] {location}: {self.message}"


@dataclass
class ValidationReport:
    image_counts: Counter
    object_counts: Counter
    issues: list[Issue]

    @property
    def fatal_issues(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.code != "coordinate_range"]

    @property
    def coordinate_issues(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.code == "coordinate_range"]


def _read_config(root: Path) -> tuple[Path, int, list[str]]:
    config_path = root / "data.yaml"
    if not config_path.is_file():
        raise ValueError(f"Missing dataset config: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    names_value = config.get("names", [])
    if isinstance(names_value, dict):
        names = [str(names_value[key]) for key in sorted(names_value, key=int)]
    elif isinstance(names_value, list):
        names = [str(name) for name in names_value]
    else:
        raise ValueError("data.yaml 'names' must be a list or mapping")
    class_count = int(config.get("nc", len(names)))
    if class_count <= 0 or len(names) != class_count:
        raise ValueError(
            f"data.yaml has nc={class_count}, but {len(names)} class names"
        )
    return config_path, class_count, names


def validate_dataset(root: Path) -> ValidationReport:
    root = root.resolve()
    _, class_count, _ = _read_config(root)
    image_counts: Counter = Counter()
    object_counts: Counter = Counter()
    issues: list[Issue] = []

    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir():
            if split != "test":
                issues.append(Issue("missing_directory", image_dir, "directory is required"))
            continue
        if not label_dir.is_dir():
            issues.append(Issue("missing_directory", label_dir, "directory is required"))
            continue

        images = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        image_counts[split] = len(images)
        for stem in sorted(images.keys() - labels.keys()):
            issues.append(Issue("missing_label", images[stem], "image has no label file"))
        for stem in sorted(labels.keys() - images.keys()):
            issues.append(Issue("orphan_label", labels[stem], "label has no matching image"))

        for path in sorted(labels.values()):
            for line_number, raw_line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                line = raw_line.strip()
                if not line:
                    continue
                fields = line.split()
                if len(fields) != 9:
                    issues.append(
                        Issue(
                            "field_count",
                            path,
                            f"expected class + 8 OBB coordinates, got {len(fields)} fields",
                            line_number,
                        )
                    )
                    continue
                try:
                    values = [float(field) for field in fields]
                except ValueError:
                    issues.append(Issue("non_numeric", path, "contains a non-numeric value", line_number))
                    continue
                class_value = values[0]
                class_id = int(class_value)
                if class_value != class_id or not 0 <= class_id < class_count:
                    issues.append(
                        Issue(
                            "class_range",
                            path,
                            f"class {class_value:g} is outside [0, {class_count - 1}]",
                            line_number,
                        )
                    )
                    continue
                object_counts[class_id] += 1
                out_of_range = [value for value in values[1:] if not 0.0 <= value <= 1.0]
                if out_of_range:
                    issues.append(
                        Issue(
                            "coordinate_range",
                            path,
                            f"normalized coordinates outside [0, 1]: {out_of_range}",
                            line_number,
                        )
                    )

    return ValidationReport(image_counts, object_counts, issues)


def clip_coordinates(root: Path) -> int:
    """Clip valid OBB rows in place, preserving one .bak file per changed label."""
    _, class_count, _ = _read_config(root)
    changed_files = 0
    for split in SPLITS:
        label_dir = root / "labels" / split
        if not label_dir.is_dir():
            continue
        for path in sorted(label_dir.glob("*.txt")):
            lines = path.read_text(encoding="utf-8").splitlines()
            output: list[str] = []
            changed = False
            for raw_line in lines:
                fields = raw_line.split()
                if len(fields) != 9:
                    output.append(raw_line)
                    continue
                try:
                    values = [float(field) for field in fields]
                except ValueError:
                    output.append(raw_line)
                    continue
                class_id = int(values[0])
                if values[0] != class_id or not 0 <= class_id < class_count:
                    output.append(raw_line)
                    continue
                clipped = [min(1.0, max(0.0, value)) for value in values[1:]]
                if clipped != values[1:]:
                    changed = True
                    output.append(" ".join([str(class_id), *[f"{value:.10f}" for value in clipped]]))
                else:
                    output.append(raw_line)
            if not changed:
                continue
            backup = path.with_suffix(path.suffix + ".bak")
            if backup.exists():
                raise FileExistsError(
                    f"Refusing to replace existing backup {backup}; inspect it first"
                )
            shutil.copy2(path, backup)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text("\n".join(output) + "\n", encoding="utf-8")
            temporary.replace(path)
            changed_files += 1
    return changed_files


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "dataset" / "data-26-07-14",
        help="Dataset root containing data.yaml, images/, and labels/",
    )
    parser.add_argument(
        "--fix-boundaries",
        action="store_true",
        help="Clip only coordinates outside [0, 1], backing up changed labels as .txt.bak",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.data_root.resolve()
    if args.fix_boundaries:
        changed = clip_coordinates(root)
        print(f"Clipped boundary coordinates in {changed} label files.")

    report = validate_dataset(root)
    print("Images:", ", ".join(f"{split}={report.image_counts[split]}" for split in SPLITS))
    print("Objects:", ", ".join(f"class_{key}={value}" for key, value in sorted(report.object_counts.items())))
    for issue in report.issues:
        print(issue.format())
    print(
        f"Validation finished: {len(report.fatal_issues)} fatal issue(s), "
        f"{len(report.coordinate_issues)} boundary issue(s)."
    )
    return 1 if report.issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
