from pathlib import Path
import random

import numpy as np
import soundfile as sf


# ============================================================
# CONFIG
# ============================================================

DATASET = Path("dataset/processed/positive")

SAMPLE_RATE = 16000
EXPECTED_SAMPLES = 16000

NUM_RANDOM_SAMPLES = 20

RANDOM_SEED = 42


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("VERIFYING PROCESSED VAANI DATASET")
    print("=" * 60)

    files = sorted(DATASET.glob("*.wav"))

    print(f"\nTotal files: {len(files)}")

    if len(files) == 0:
        print("ERROR: No WAV files found.")
        return

    # --------------------------------------------------------
    # Format verification
    # --------------------------------------------------------

    bad_format = []

    for path in files:

        try:

            info = sf.info(path)

            if (
                info.samplerate != SAMPLE_RATE
                or info.channels != 1
                or info.frames != EXPECTED_SAMPLES
            ):

                bad_format.append(
                    (
                        path.name,
                        info.samplerate,
                        info.channels,
                        info.frames
                    )
                )

        except Exception as e:

            bad_format.append(
                (
                    path.name,
                    "ERROR",
                    str(e),
                    ""
                )
            )

    print("\nFormat verification:")

    if not bad_format:

        print("  PASS — every file is 16 kHz, mono, 1 second.")

    else:

        print(
            f"  FAIL — {len(bad_format)} invalid files."
        )

        for item in bad_format[:10]:
            print(" ", item)

    # --------------------------------------------------------
    # Speaker counts
    # --------------------------------------------------------

    speakers = {}

    for path in files:

        # filename format:
        # speaker_number.wav
        speaker = path.stem.rsplit("_", 1)[0]

        speakers[speaker] = speakers.get(speaker, 0) + 1

    print("\nSpeaker distribution:")

    for speaker, count in sorted(speakers.items()):

        percentage = 100 * count / len(files)

        print(
            f"  {speaker:10s}: {count:4d} "
            f"({percentage:5.1f}%)"
        )

    # --------------------------------------------------------
    # Random audio inspection
    # --------------------------------------------------------

    sample_count = min(
        NUM_RANDOM_SAMPLES,
        len(files)
    )

    selected = random.sample(
        files,
        sample_count
    )

    print("\nRandom samples:")
    print("-" * 60)

    for path in selected:

        audio, sr = sf.read(
            path,
            dtype="float32"
        )

        rms = np.sqrt(
            np.mean(audio ** 2)
        )

        peak = np.max(
            np.abs(audio)
        )

        print(
            f"{path.name:25s} "
            f"RMS={rms:.4f}  "
            f"peak={peak:.4f}"
        )

    # --------------------------------------------------------
    # Basic audio sanity
    # --------------------------------------------------------

    silent_files = []
    clipped_files = []

    for path in files:

        audio, _ = sf.read(
            path,
            dtype="float32"
        )

        rms = np.sqrt(
            np.mean(audio ** 2)
        )

        peak = np.max(
            np.abs(audio)
        )

        if rms < 0.005:
            silent_files.append(path.name)

        if peak > 0.999:
            clipped_files.append(path.name)

    print("\nAudio sanity:")

    print(
        f"  Nearly silent files: {len(silent_files)}"
    )

    print(
        f"  Potentially clipped files: {len(clipped_files)}"
    )

    if silent_files:

        print("\nFirst silent files:")

        for name in silent_files[:10]:
            print(" ", name)

    if clipped_files:

        print("\nFirst potentially clipped files:")

        for name in clipped_files[:10]:
            print(" ", name)

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()