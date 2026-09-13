from pathlib import Path
import random
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = (
    PROJECT_ROOT /
    "model_v1" /
    "augmented" /
    "train" /
    "positive"
)

INSPECTION_DIR = (
    PROJECT_ROOT /
    "model_v1" /
    "evaluation" /
    "augmentation_inspection"
)

NUM_FILES = 15

SEED = 42
random.seed(SEED)


def main():

    print("=" * 60)
    print("VAANI MODEL V1 — AUGMENTATION INSPECTION")
    print("=" * 60)

    files = sorted(SOURCE_DIR.glob("*.wav"))

    if not files:
        raise RuntimeError(
            f"No augmented files found in {SOURCE_DIR}"
        )

    selected = random.sample(
        files,
        min(NUM_FILES, len(files))
    )

    INSPECTION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for index, source in enumerate(selected, start=1):

        destination = (
            INSPECTION_DIR /
            f"sample_{index:02d}_{source.name}"
        )

        shutil.copy2(source, destination)

        print(f"{index:02d}: {destination.name}")

    print()
    print("Inspection files created:")
    print(INSPECTION_DIR)


if __name__ == "__main__":
    main()