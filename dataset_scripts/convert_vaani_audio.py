import subprocess
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE = Path("dataset/raw/SIH_Vaani_Dataset")

OUTPUT = Path("dataset/raw/SIH_Vaani_Dataset_Converted")

# Audio format we want
SAMPLE_RATE = 16000
CHANNELS = 1

# Supported input formats
SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
}


# ============================================================
# CHECK FFMPEG
# ============================================================

def check_ffmpeg():

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if result.returncode != 0:
            raise RuntimeError

    except Exception:

        print("ERROR: FFmpeg was not found.")
        print()
        print("Make sure FFmpeg is installed and available")
        print("in your PATH.")
        print()
        print("Test it with:")
        print("    ffmpeg -version")
        raise SystemExit(1)


# ============================================================
# CONVERT ONE FILE
# ============================================================

def convert_file(input_path, output_path):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    command = [
        "ffmpeg",

        "-y",

        "-i",
        str(input_path),

        # Mono
        "-ac",
        str(CHANNELS),

        # 16 kHz
        "-ar",
        str(SAMPLE_RATE),

        # 16-bit PCM WAV
        "-c:a",
        "pcm_s16le",

        str(output_path),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:

        print(
            f"\nFAILED: {input_path}"
        )

        print(result.stderr)

        return False

    return True


# ============================================================
# MAIN CONVERSION
# ============================================================

def main():

    check_ffmpeg()

    if not SOURCE.exists():

        print(
            f"ERROR: Source folder does not exist:\n"
            f"{SOURCE.resolve()}"
        )

        return

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    total = 0
    converted = 0
    skipped = 0
    failed = 0

    print("=" * 60)
    print("VAANI AUDIO CONVERSION")
    print("=" * 60)

    print(f"Source:")
    print(f"  {SOURCE.resolve()}")

    print(f"\nOutput:")
    print(f"  {OUTPUT.resolve()}")

    print("\nTarget format:")
    print("  WAV")
    print("  16,000 Hz")
    print("  Mono")
    print("  PCM 16-bit")

    print("=" * 60)

    # Process speaker folders
    for speaker_dir in sorted(SOURCE.iterdir()):

        if not speaker_dir.is_dir():
            continue

        print(
            f"\nSpeaker: {speaker_dir.name}"
        )

        for input_path in sorted(
            speaker_dir.iterdir()
        ):

            if not input_path.is_file():
                continue

            if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            total += 1

            # Keep the same original filename,
            # but change extension to .wav
            output_name = (
                input_path.stem + ".wav"
            )

            output_path = (
                OUTPUT
                / speaker_dir.name
                / output_name
            )

            # Don't reconvert existing files
            if output_path.exists():

                skipped += 1

                continue

            print(
                f"  {input_path.name} -> "
                f"{output_name}"
            )

            success = convert_file(
                input_path,
                output_path
            )

            if success:
                converted += 1
            else:
                failed += 1


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)

    print(f"Input audio files found : {total}")
    print(f"Successfully converted  : {converted}")
    print(f"Already existed         : {skipped}")
    print(f"Failed                  : {failed}")

    print("=" * 60)

    print(
        "\nConverted dataset:"
    )

    print(
        f"{OUTPUT.resolve()}"
    )

    print(
        "\nYour original recordings have NOT been modified."
    )


if __name__ == "__main__":
    main()