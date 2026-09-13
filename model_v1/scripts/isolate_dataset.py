from pathlib import Path
import csv
import shutil

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = PROJECT_ROOT / "dataset" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "model_v1" / "data"

SPLITS = ["train", "validation", "test"]


# ============================================================
# HELPERS
# ============================================================

def find_column(fieldnames, candidates):
    """Find a CSV column using several possible names."""
    lowered = {name.lower(): name for name in fieldnames}

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return None


def copy_split(split_name):
    csv_path = SPLIT_DIR / f"{split_name}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing: {csv_path}")

    split_output = OUTPUT_DIR / split_name

    positive_dir = split_output / "positive"
    negative_dir = split_output / "negative"

    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print(f"COPYING {split_name.upper()}")
    print("=" * 60)

    positive_count = 0
    negative_count = 0

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        fieldnames = reader.fieldnames

        if not fieldnames:
            raise ValueError(f"No CSV header found in {csv_path}")

        path_column = find_column(
            fieldnames,
            ["path", "filepath", "file", "filename"]
        )

        label_column = find_column(
            fieldnames,
            ["label", "target", "class"]
        )

        if path_column is None:
            raise ValueError(
                f"Could not find path column in {csv_path}. "
                f"Columns: {fieldnames}"
            )

        if label_column is None:
            raise ValueError(
                f"Could not find label column in {csv_path}. "
                f"Columns: {fieldnames}"
            )

        for index, row in enumerate(reader, start=1):

            source = Path(row[path_column])

            # Handle relative paths from the project root.
            if not source.is_absolute():
                source = PROJECT_ROOT / source

            if not source.exists():
                raise FileNotFoundError(
                    f"Source audio does not exist:\n{source}"
                )

            label = str(row[label_column]).strip().lower()

            if label in {"1", "positive", "vaani", "keyword"}:
                destination_dir = positive_dir
                positive_count += 1
            else:
                destination_dir = negative_dir
                negative_count += 1

            destination = destination_dir / (
                f"{index:06d}_{source.name}"
            )

            shutil.copy2(source, destination)

            if index % 100 == 0:
                print(f"Copied {index} files")

    print()
    print(f"Positive: {positive_count}")
    print(f"Negative: {negative_count}")

    return positive_count, negative_count


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VAANI MODEL V1 — DATA ISOLATION")
    print("=" * 60)

    print()
    print(f"Source: {SPLIT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    if OUTPUT_DIR.exists():
        print()
        print("model_v1/data already exists.")
        print("Delete it manually if you want to rebuild it.")
        return

    total_positive = 0
    total_negative = 0

    for split in SPLITS:
        pos, neg = copy_split(split)
        total_positive += pos
        total_negative += neg

    print()
    print("=" * 60)
    print("ISOLATION COMPLETE")
    print("=" * 60)

    print()
    print(f"Total positive: {total_positive}")
    print(f"Total negative: {total_negative}")

    print()
    print("V1 now owns its own copy of the dataset.")
    print()
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()