from pathlib import Path
import csv
import random
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

POSITIVE_DIR = Path("dataset/processed/positive")

NEGATIVE_SPEECH_DIR = Path(
    "dataset/raw/negative_speech_commands"
)

NEGATIVE_BACKGROUND_DIR = Path(
    "dataset/raw/negative_background"
)

OUTPUT_DIR = Path("dataset/splits")

SEED = 42

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# HELPERS
# ============================================================

def split_list(items):
    """
    Split a list into train / validation / test.

    The split is deterministic because we use a fixed seed
    before shuffling.
    """

    items = list(items)

    random.shuffle(items)

    n = len(items)

    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]

    return train, val, test


def speaker_from_positive(filename):
    """
    Positive filenames look like:

        ananya_001.wav
        ark_037.wav
        vitthal_205.wav

    Everything before the final _number is the speaker.
    """

    return filename.rsplit("_", 1)[0]


def write_csv(path, rows):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filepath",
                "label",
                "speaker_id"
            ]
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(SEED)

    print("=" * 70)
    print("VAANI DATASET SPLIT")
    print("=" * 70)

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not POSITIVE_DIR.exists():
        print(f"\nERROR: Missing positive directory:")
        print(POSITIVE_DIR)
        return

    if not NEGATIVE_SPEECH_DIR.exists():
        print(f"\nERROR: Missing negative speech directory:")
        print(NEGATIVE_SPEECH_DIR)
        return

    if not NEGATIVE_BACKGROUND_DIR.exists():
        print(f"\nERROR: Missing negative background directory:")
        print(NEGATIVE_BACKGROUND_DIR)
        return

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load positive files
    # --------------------------------------------------------

    positive_files = sorted(
        POSITIVE_DIR.glob("*.wav")
    )

    print(
        f"\nPositive files found: "
        f"{len(positive_files)}"
    )

    # --------------------------------------------------------
    # Group positives by speaker
    # --------------------------------------------------------

    positive_by_speaker = {}

    for path in positive_files:

        speaker = speaker_from_positive(
            path.name
        )

        positive_by_speaker.setdefault(
            speaker,
            []
        ).append(path)

    print("\nPositive speakers:")

    for speaker in sorted(positive_by_speaker):

        print(
            f"  {speaker:10s}: "
            f"{len(positive_by_speaker[speaker])}"
        )

    # --------------------------------------------------------
    # Split positive files WITHIN EACH SPEAKER
    # --------------------------------------------------------

    train_positive = []
    val_positive = []
    test_positive = []

    print("\nPositive split:")

    for speaker in sorted(positive_by_speaker):

        files = positive_by_speaker[speaker]

        train, val, test = split_list(files)

        train_positive.extend(train)
        val_positive.extend(val)
        test_positive.extend(test)

        print(
            f"  {speaker:10s}: "
            f"train={len(train):3d} "
            f"val={len(val):2d} "
            f"test={len(test):2d}"
        )

    # --------------------------------------------------------
    # Load negatives
    #
    # Speech Commands and background are both negative class.
    # --------------------------------------------------------

    negative_speech = sorted(
        NEGATIVE_SPEECH_DIR.glob("*.wav")
    )

    negative_background = sorted(
        NEGATIVE_BACKGROUND_DIR.glob("*.wav")
    )

    negative_files = (
        negative_speech +
        negative_background
    )

    print(
        f"\nNegative speech-command files: "
        f"{len(negative_speech)}"
    )

    print(
        f"Negative background files:      "
        f"{len(negative_background)}"
    )

    print(
        f"Total negative files:            "
        f"{len(negative_files)}"
    )

    # --------------------------------------------------------
    # We want balanced classes.
    #
    # Use the same number of negatives as positives.
    # This avoids the huge Speech Commands dataset
    # overwhelming the positive class.
    # --------------------------------------------------------

    target_negative_count = len(positive_files)

    if len(negative_files) < target_negative_count:

        print(
            "\nWARNING: Not enough negative files "
            "for a 1:1 class balance."
        )

        selected_negative = negative_files

    else:

        selected_negative = random.sample(
            negative_files,
            target_negative_count
        )

    print(
        f"\nNegative files selected: "
        f"{len(selected_negative)}"
    )

    # --------------------------------------------------------
    # Split negatives
    # --------------------------------------------------------

    train_negative, val_negative, test_negative = (
        split_list(selected_negative)
    )

    # --------------------------------------------------------
    # Build CSV rows
    # --------------------------------------------------------

    def positive_rows(files):

        rows = []

        for path in files:

            rows.append({
                "filepath": str(path),
                "label": 1,
                "speaker_id": speaker_from_positive(
                    path.name
                )
            })

        return rows

    def negative_rows(files):

        rows = []

        for path in files:

            # Negative audio doesn't belong to a
            # meaningful speaker group for our purposes.
            #
            # We use the source type instead.

            if path.parent == NEGATIVE_SPEECH_DIR:
                speaker = "negative_speech_commands"
            else:
                speaker = "negative_background"

            rows.append({
                "filepath": str(path),
                "label": 0,
                "speaker_id": speaker
            })

        return rows

    train_rows = (
        positive_rows(train_positive)
        + negative_rows(train_negative)
    )

    val_rows = (
        positive_rows(val_positive)
        + negative_rows(val_negative)
    )

    test_rows = (
        positive_rows(test_positive)
        + negative_rows(test_negative)
    )

    # --------------------------------------------------------
    # Shuffle each final split
    # --------------------------------------------------------

    random.shuffle(train_rows)
    random.shuffle(val_rows)
    random.shuffle(test_rows)

    # --------------------------------------------------------
    # Write manifests
    # --------------------------------------------------------

    train_path = OUTPUT_DIR / "train.csv"
    val_path = OUTPUT_DIR / "validation.csv"
    test_path = OUTPUT_DIR / "test.csv"

    write_csv(
        train_path,
        train_rows
    )

    write_csv(
        val_path,
        val_rows
    )

    write_csv(
        test_path,
        test_rows
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL DATASET")
    print("=" * 70)

    def print_stats(name, rows):

        positives = sum(
            r["label"] == 1
            for r in rows
        )

        negatives = sum(
            r["label"] == 0
            for r in rows
        )

        print(
            f"\n{name}:"
        )

        print(
            f"  Total:     {len(rows)}"
        )

        print(
            f"  Positive:  {positives}"
        )

        print(
            f"  Negative:  {negatives}"
        )

        print(
            f"  Ratio:     "
            f"{positives}:{negatives}"
        )

        # Positive speaker distribution

        speaker_counts = {}

        for row in rows:

            if row["label"] == 1:

                speaker = row["speaker_id"]

                speaker_counts[speaker] = (
                    speaker_counts.get(speaker, 0)
                    + 1
                )

        print("  Positive speakers:")

        for speaker in sorted(speaker_counts):

            print(
                f"    {speaker:10s}: "
                f"{speaker_counts[speaker]}"
            )

    print_stats(
        "TRAIN",
        train_rows
    )

    print_stats(
        "VALIDATION",
        val_rows
    )

    print_stats(
        "TEST",
        test_rows
    )

    print("\n" + "=" * 70)
    print("MANIFEST FILES")
    print("=" * 70)

    print(f"\n{train_path}")
    print(f"{val_path}")
    print(f"{test_path}")

    print("\nDataset split complete.")


if __name__ == "__main__":
    main()