from pathlib import Path
import shutil
import random

import numpy as np
import soundfile as sf
import librosa


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE = Path("dataset/raw/positive")

PROBLEMATIC = Path("dataset/raw/problematic")

OUTPUT = Path("dataset/processed/positive")

SAMPLE_RATE = 16000
CLIP_DURATION = 1.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)

# Silence trimming
TOP_DB = 30

# Reproducible randomness
RANDOM_SEED = 42

# Files known to contain Vaani more than once.
# Add filenames here after listening to them.
REPEATED_FILES = {
    # Example:
    # "ark_081.wav",
    # "ark_082.wav",
}


# ============================================================
# HELPERS
# ============================================================

def load_audio(path):
    """
    Load audio as mono at its original sample rate.
    """

    audio, sr = librosa.load(
        path,
        sr=None,
        mono=True
    )

    audio = audio.astype(np.float32)

    return audio, sr


def resample_if_needed(audio, sr):
    """
    Convert audio to 16 kHz if necessary.
    """

    if sr != SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=SAMPLE_RATE
        )

    return audio.astype(np.float32)


def trim_silence(audio):
    """
    Remove leading/trailing silence.
    """

    if len(audio) == 0:
        return audio

    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=TOP_DB
    )

    return trimmed.astype(np.float32)


def normalize_peak(audio):
    """
    Gentle peak normalization.

    We do NOT normalize RMS because we want to preserve
    realistic loudness differences between speakers.
    """

    if len(audio) == 0:
        return audio

    peak = np.max(np.abs(audio))

    if peak > 0.0:
        audio = audio / peak * 0.95

    return audio.astype(np.float32)


def make_fixed_length(audio):
    """
    Convert arbitrary-duration audio into exactly 1 second.

    Strategy:

    1. If longer than 1 second:
       center-crop around the speech/keyword region.

    2. If shorter than 1 second:
       place it in the center and zero-pad.

    This is NOT simply taking the first second.
    """

    target = CLIP_SAMPLES

    if len(audio) == target:
        return audio

    # --------------------------------------------------------
    # LONGER THAN 1 SECOND
    # --------------------------------------------------------

    if len(audio) > target:

        # Find approximate speech-active region.
        intervals = librosa.effects.split(
            audio,
            top_db=TOP_DB
        )

        if len(intervals) > 0:

            speech_start = intervals[0][0]
            speech_end = intervals[-1][1]

            speech_center = (speech_start + speech_end) // 2

        else:

            speech_center = len(audio) // 2

        start = speech_center - target // 2
        end = start + target

        # Keep crop inside audio.
        if start < 0:
            start = 0
            end = target

        if end > len(audio):
            end = len(audio)
            start = end - target

        clip = audio[start:end]

        return clip.astype(np.float32)

    # --------------------------------------------------------
    # SHORTER THAN 1 SECOND
    # --------------------------------------------------------

    else:

        result = np.zeros(target, dtype=np.float32)

        start = (target - len(audio)) // 2
        end = start + len(audio)

        result[start:end] = audio

        return result


def save_wav(path, audio):
    """
    Save 16-bit PCM WAV.
    """

    audio = np.clip(audio, -1.0, 1.0)

    sf.write(
        path,
        audio,
        SAMPLE_RATE,
        subtype="PCM_16"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(RANDOM_SEED)

    print("=" * 60)
    print("VAANI TRAINING DATA PREPARATION")
    print("=" * 60)

    if not SOURCE.exists():
        print(f"\nERROR: Source folder does not exist:")
        print(SOURCE)
        return

    PROBLEMATIC.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    files = sorted(SOURCE.glob("*.wav"))

    print(f"\nInput files: {len(files)}")
    print(f"Output directory: {OUTPUT}")

    processed = 0
    skipped = 0
    failed = 0

    durations_before = []
    durations_after_trim = []

    # --------------------------------------------------------
    # PROCESS EACH FILE
    # --------------------------------------------------------

    for source_path in files:

        filename = source_path.name

        # ----------------------------------------------------
        # Problematic/repeated keyword files
        # ----------------------------------------------------

        if filename in REPEATED_FILES:

            destination = PROBLEMATIC / filename

            if not destination.exists():

                shutil.copy2(
                    source_path,
                    destination
                )

            skipped += 1

            print(f"[SKIP] repeated keyword: {filename}")

            continue

        try:

            audio, sr = load_audio(source_path)

            durations_before.append(
                len(audio) / sr
            )

            audio = resample_if_needed(audio, sr)

            audio = trim_silence(audio)

            durations_after_trim.append(
                len(audio) / SAMPLE_RATE
            )

            audio = normalize_peak(audio)

            audio = make_fixed_length(audio)

            # ------------------------------------------------
            # Output name
            #
            # Example:
            # ark_037.wav -> ark_037.wav
            # ------------------------------------------------

            destination = OUTPUT / filename

            save_wav(
                destination,
                audio
            )

            processed += 1

        except Exception as e:

            failed += 1

            print(
                f"[ERROR] {filename}: {e}"
            )

    # ========================================================
    # VERIFY OUTPUT
    # ========================================================

    print("\n" + "=" * 60)
    print("VERIFYING OUTPUT")
    print("=" * 60)

    output_files = sorted(
        OUTPUT.glob("*.wav")
    )

    bad_files = []

    for path in output_files:

        try:

            info = sf.info(path)

            expected_samples = CLIP_SAMPLES

            if (
                info.samplerate != SAMPLE_RATE
                or info.channels != 1
                or info.frames != expected_samples
            ):

                bad_files.append(
                    (
                        path.name,
                        info.samplerate,
                        info.channels,
                        info.frames
                    )
                )

        except Exception as e:

            bad_files.append(
                (
                    path.name,
                    "ERROR",
                    str(e),
                    ""
                )
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Input files:             {len(files)}")
    print(f"Processed:               {processed}")
    print(f"Repeated/problematic:    {skipped}")
    print(f"Failed:                  {failed}")
    print(f"Output files:            {len(output_files)}")

    if durations_before:

        print("\nOriginal duration:")
        print(
            f"  min:  {min(durations_before):.3f}s"
        )
        print(
            f"  max:  {max(durations_before):.3f}s"
        )
        print(
            f"  mean: {np.mean(durations_before):.3f}s"
        )

    if durations_after_trim:

        print("\nAfter silence trimming:")
        print(
            f"  min:  {min(durations_after_trim):.3f}s"
        )
        print(
            f"  max:  {max(durations_after_trim):.3f}s"
        )
        print(
            f"  mean: {np.mean(durations_after_trim):.3f}s"
        )

    print("\nExpected processed format:")
    print("  Sample rate: 16000 Hz")
    print("  Channels:    1")
    print("  Duration:    1.000 s")
    print("  Samples:     16000")
    print("  Format:      PCM 16-bit")

    if bad_files:

        print("\nWARNING: Invalid output files found:")

        for item in bad_files[:20]:
            print(" ", item)

        print(
            f"\nTotal invalid files: {len(bad_files)}"
        )

    else:

        print(
            "\nALL OUTPUT FILES PASSED FORMAT CHECK."
        )

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()