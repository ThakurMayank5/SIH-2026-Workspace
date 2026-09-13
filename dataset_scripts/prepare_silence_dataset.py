from pathlib import Path
import wave
import subprocess
import tempfile
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "dataset" / "raw" / "silence" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "raw" / "silence" / "processed"

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2          # PCM16
CLIP_SECONDS = 1
CLIP_SAMPLES = SAMPLE_RATE * CLIP_SECONDS

# Skip clips that are essentially digital silence.
# We WANT quiet room noise, so this threshold is intentionally very low.
MIN_RMS_DBFS = -80.0

# Set to True if you want to delete/recreate processed clips.
CLEAR_OUTPUT = False


# ============================================================
# HELPERS
# ============================================================

def rms_dbfs(audio):
    """Calculate RMS level in dBFS for normalized float audio."""
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2) + 1e-12)
    return 20.0 * np.log10(rms)


def convert_to_wav(input_file, output_wav):
    """
    Convert any supported audio file to:
        16 kHz
        mono
        PCM16 WAV
    using ffmpeg.
    """

    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_file),
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(output_wav),
    ]

    subprocess.run(command, check=True)


def read_wav(path):
    """Read PCM16 WAV and return float32 audio in [-1, 1]."""

    with wave.open(str(path), "rb") as wf:

        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()

        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Unexpected sample rate: {sample_rate}"
            )

        if channels != CHANNELS:
            raise ValueError(
                f"Unexpected channels: {channels}"
            )

        if sample_width != SAMPLE_WIDTH:
            raise ValueError(
                f"Expected PCM16, got sample width={sample_width}"
            )

        raw = wf.readframes(n_frames)

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)

    return audio / 32768.0


def save_wav(path, audio):
    """Save float audio [-1,1] as PCM16 mono WAV."""

    audio = np.clip(audio, -1.0, 1.0)

    pcm = (audio * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SILENCE DATASET PREPARATION")
    print("=" * 70)

    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    if not INPUT_DIR.exists():
        print("ERROR: Input directory does not exist.")
        print(INPUT_DIR)
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if CLEAR_OUTPUT:
        print("Clearing existing processed clips...")

        for file in OUTPUT_DIR.glob("*.wav"):
            file.unlink()

        print("Done.\n")

    # --------------------------------------------------------
    # Find source audio
    # --------------------------------------------------------

    extensions = {
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".ogg",
        ".aac",
        ".wma",
    }

    source_files = sorted(
        p for p in INPUT_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not source_files:
        print("ERROR: No audio files found.")
        return

    print(f"Source recordings found: {len(source_files)}")
    print()

    total_clips = 0
    skipped_clips = 0

    # --------------------------------------------------------
    # Process every long recording
    # --------------------------------------------------------

    for source_index, source_file in enumerate(source_files, start=1):

        print(
            f"[{source_index}/{len(source_files)}] "
            f"{source_file.name}"
        )

        with tempfile.TemporaryDirectory(
            dir=PROJECT_ROOT / "temp"
        ) as temp_dir:

            temp_dir = Path(temp_dir)

            converted_wav = (
                temp_dir / "converted.wav"
            )

            print("  Converting to 16 kHz mono PCM16...")

            convert_to_wav(
                source_file,
                converted_wav
            )

            audio = read_wav(converted_wav)

        duration = len(audio) / SAMPLE_RATE

        print(f"  Duration: {duration:.2f} seconds")

        # ----------------------------------------------------
        # Split into non-overlapping 1-second clips
        # ----------------------------------------------------

        n_clips = len(audio) // CLIP_SAMPLES

        print(f"  Possible 1-second clips: {n_clips}")

        source_stem = source_file.stem

        for clip_index in range(n_clips):

            start = clip_index * CLIP_SAMPLES
            end = start + CLIP_SAMPLES

            clip = audio[start:end]

            level = rms_dbfs(clip)

            # Remove only completely silent/digital-zero clips.
            # Quiet room noise is kept.
            if level < MIN_RMS_DBFS:
                skipped_clips += 1
                continue

            output_name = (
                f"{source_stem}"
                f"_clip_{clip_index:05d}.wav"
            )

            output_path = OUTPUT_DIR / output_name

            save_wav(
                output_path,
                clip
            )

            total_clips += 1

        print()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Source recordings : {len(source_files)}")
    print(f"Processed clips   : {total_clips}")
    print(f"Skipped clips     : {skipped_clips}")
    print(f"Output directory  : {OUTPUT_DIR}")
    print()

    print("Each processed clip:")
    print("  - 1.0 second")
    print("  - 16 kHz")
    print("  - mono")
    print("  - PCM16")
    print()

    print("These clips are ready to be used as negative/silence data.")


if __name__ == "__main__":
    main()