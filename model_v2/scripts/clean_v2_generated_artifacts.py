"""Clean generated V2 dataset/training artifacts for a fresh rerun.

This script removes only generated files under model_v2. It never touches
dataset/raw or any other source dataset directory.

Default mode is a dry run:
    python model_v2/scripts/clean_v2_generated_artifacts.py

Actually clean generated artifacts:
    python model_v2/scripts/clean_v2_generated_artifacts.py --yes
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SPLITS = ("train", "validation", "test")
CATEGORIES = (
    "positive",
    "negative_silence",
    "negative_background",
    "negative_speech_commands",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_inside(child: Path, parent: Path) -> Path:
    child = child.resolve()
    parent = parent.resolve()
    if child != parent and parent not in child.parents:
        raise RuntimeError(f"Refusing to clean path outside {parent}: {child}")
    return child


def remove_path(path: Path, dry_run: bool) -> None:
    if not path.exists():
        print(f"SKIP missing: {path}")
        return

    action = "DRY-RUN remove" if dry_run else "remove"
    print(f"{action}: {path}")
    if dry_run:
        return

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def recreate_layout(model_v2: Path, dry_run: bool) -> None:
    dirs = [
        model_v2 / "data" / "manifests",
        model_v2 / "data" / "train_augmented",
        model_v2 / "features",
        model_v2 / "checkpoints",
        model_v2 / "evaluation",
        model_v2 / "exports",
    ]
    for split in SPLITS:
        for category in CATEGORIES:
            dirs.append(model_v2 / "data" / split / category)

    for directory in dirs:
        print(f"{'DRY-RUN create' if dry_run else 'create'}: {directory}")
        if not dry_run:
            directory.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove generated V2 prep/training artifacts for a fresh rerun."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually remove generated artifacts. Without this flag, only print a dry run.",
    )
    args = parser.parse_args()

    root = project_root()
    model_v2 = ensure_inside(root / "model_v2", root)
    raw_dataset = ensure_inside(root / "dataset" / "raw", root)

    print(f"Project root: {root}")
    print(f"Protected raw dataset: {raw_dataset}")
    print(f"Mode: {'CLEAN' if args.yes else 'DRY RUN'}")
    print()

    targets = [
        model_v2 / "data" / "train",
        model_v2 / "data" / "validation",
        model_v2 / "data" / "test",
        model_v2 / "data" / "train_augmented",
        model_v2 / "data" / "manifests",
        model_v2 / "features",
        model_v2 / "checkpoints",
        model_v2 / "evaluation",
        model_v2 / "exports",
    ]

    for target in targets:
        safe_target = ensure_inside(target, model_v2)
        remove_path(safe_target, dry_run=not args.yes)

    print()
    recreate_layout(model_v2, dry_run=not args.yes)

    print()
    if args.yes:
        print("Generated V2 artifacts cleaned. dataset/raw was not touched.")
    else:
        print("Dry run complete. Re-run with --yes to clean generated artifacts.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
