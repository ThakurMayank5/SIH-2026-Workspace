from pathlib import Path
import random
import soundfile as sf
import librosa


POSITIVE = Path("dataset/raw/positive")

NUM_SAMPLES = 40


def main():

    files = sorted(POSITIVE.glob("*.wav"))

    if not files:
        print("No WAV files found in:", POSITIVE.resolve())
        return

    print("=" * 60)
    print("VAANI CLIP INSPECTION")
    print("=" * 60)

    print(f"Total files: {len(files)}")

    # Pick random files
    random.seed(42)

    samples = random.sample(
        files,
        min(NUM_SAMPLES, len(files))
    )

    print("\nRandom sample:")
    print("-" * 60)

    for path in samples:

        try:
            info = sf.info(path)

            audio, sr = librosa.load(
                path,
                sr=None,
                mono=True
            )

            trimmed, _ = librosa.effects.trim(
                audio,
                top_db=30
            )

            raw_duration = len(audio) / sr
            trimmed_duration = len(trimmed) / sr

            print(
                f"{path.name:35} "
                f"raw={raw_duration:5.2f}s  "
                f"trimmed={trimmed_duration:5.2f}s"
            )

        except Exception as e:

            print(
                f"{path.name}: ERROR: {e}"
            )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()