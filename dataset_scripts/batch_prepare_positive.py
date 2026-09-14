from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
from IPython.display import Audio, display


# ============================================================
# CUSTOMIZE THESE TWO PATHS
# ============================================================

INPUT_DIR = Path(
    r"C:\Users\Mayank Singh\Codes\SIH 2026\dataset\raw\SIH_Vaani_Dataset_Converted\Vitthal"
)

OUTPUT_DIR = Path(
    r"C:\Users\Mayank Singh\Codes\SIH 2026\temp\positive_test\Vitthal"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET_SR = 16000
TARGET_SAMPLES = TARGET_SR       # 1 second
TOP_DB = 30
TARGET_PEAK = 0.95

# Number of files to play for manual inspection
PREVIEW_COUNT = 5


# ============================================================
# PREPARE ONE FILE
# ============================================================

def prepare_audio(audio):

    # --------------------------------------------------------
    # Silence trimming
    # --------------------------------------------------------

    trimmed, trim_indices = librosa.effects.trim(
        audio,
        top_db=TOP_DB
    )

    start_sample, end_sample = trim_indices

    trim_start = start_sample / TARGET_SR
    trim_end = end_sample / TARGET_SR

    audio = trimmed

    # --------------------------------------------------------
    # Peak normalization
    # --------------------------------------------------------

    peak = np.max(np.abs(audio))

    if peak > 0:
        audio = audio * (TARGET_PEAK / peak)

    # --------------------------------------------------------
    # Force exactly 1 second
    # --------------------------------------------------------

    if len(audio) < TARGET_SAMPLES:

        # Center padding
        pad_total = TARGET_SAMPLES - len(audio)

        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        audio = np.pad(
            audio,
            (pad_left, pad_right),
            mode="constant"
        )

        action = "CENTER PAD"

        action_info = (
            f"left={pad_left / TARGET_SR:.3f}s, "
            f"right={pad_right / TARGET_SR:.3f}s"
        )

    elif len(audio) > TARGET_SAMPLES:

        # Center crop
        crop_total = len(audio) - TARGET_SAMPLES

        crop_left = crop_total // 2
        crop_right = crop_left + TARGET_SAMPLES

        audio = audio[
            crop_left:crop_right
        ]

        action = "CENTER CROP"

        action_info = (
            f"start={crop_left / TARGET_SR:.3f}s, "
            f"end={crop_right / TARGET_SR:.3f}s"
        )

    else:

        action = "ALREADY 1 SECOND"
        action_info = ""

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    audio = np.clip(
        audio,
        -1.0,
        1.0
    )

    return (
        audio,
        trim_start,
        trim_end,
        action,
        action_info
    )


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find all WAV files
    # --------------------------------------------------------

    wav_files = sorted(
        INPUT_DIR.rglob("*.wav")
    )

    print("=" * 70)
    print("BATCH VAANI POSITIVE DATASET PREPARATION")
    print("=" * 70)

    print(f"Input directory:")
    print(f"  {INPUT_DIR}")

    print()
    print(f"Output directory:")
    print(f"  {OUTPUT_DIR}")

    print()
    print(f"Input WAV files: {len(wav_files)}")

    if len(wav_files) == 0:

        print("\nNo WAV files found.")
        return

    print()

    successful = 0
    failed = 0

    previews = []

    # ========================================================
    # PROCESS FILES
    # ========================================================

    for src_path in tqdm(
        wav_files,
        desc="Processing"
    ):

        try:

            # ------------------------------------------------
            # Load as mono, resample to 16 kHz
            # ------------------------------------------------

            original_audio, original_sr = librosa.load(
                str(src_path),
                sr=TARGET_SR,
                mono=True
            )

            original_duration = (
                len(original_audio) / TARGET_SR
            )

            # ------------------------------------------------
            # Prepare
            # ------------------------------------------------

            (
                processed_audio,
                trim_start,
                trim_end,
                action,
                action_info
            ) = prepare_audio(
                original_audio
            )

            # ------------------------------------------------
            # Preserve filename
            # ------------------------------------------------

            dst_path = OUTPUT_DIR / src_path.name

            # ------------------------------------------------
            # Save PCM16
            # ------------------------------------------------

            sf.write(
                str(dst_path),
                processed_audio,
                TARGET_SR,
                subtype="PCM_16"
            )

            successful += 1

            # ------------------------------------------------
            # Store preview information
            # ------------------------------------------------

            if len(previews) < PREVIEW_COUNT:

                previews.append(
                    (
                        src_path,
                        dst_path,
                        original_audio.copy(),
                        processed_audio.copy(),
                        original_duration,
                        trim_start,
                        trim_end,
                        action,
                        action_info
                    )
                )

        except Exception as e:

            failed += 1

            print()
            print(f"FAILED: {src_path}")
            print(f"Reason: {e}")

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(f"Input files:       {len(wav_files)}")
    print(f"Successfully made: {successful}")
    print(f"Failed:            {failed}")

    # ========================================================
    # VERIFY OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("VERIFYING OUTPUT")
    print("=" * 70)

    output_files = sorted(
        OUTPUT_DIR.glob("*.wav")
    )

    problems = []

    for wav_path in output_files:

        try:

            info = sf.info(
                str(wav_path)
            )

            duration = info.frames / info.samplerate

            if info.samplerate != TARGET_SR:

                problems.append(
                    (
                        wav_path.name,
                        f"Wrong sample rate: {info.samplerate}"
                    )
                )

            if info.channels != 1:

                problems.append(
                    (
                        wav_path.name,
                        f"Not mono: {info.channels} channels"
                    )
                )

            if abs(duration - 1.0) > 0.001:

                problems.append(
                    (
                        wav_path.name,
                        f"Wrong duration: {duration:.4f}s"
                    )
                )

        except Exception as e:

            problems.append(
                (
                    wav_path.name,
                    str(e)
                )
            )

    print(f"Output files: {len(output_files)}")
    print(f"Problems:     {len(problems)}")

    if problems:

        print("\nPROBLEMS:")

        for filename, reason in problems:

            print(
                f"  {filename}: {reason}"
            )

    else:

        print()
        print("✓ All output files are 16 kHz")
        print("✓ All output files are mono")
        print("✓ All output files are exactly 1 second")
        print("✓ Output is PCM16")

    # ========================================================
    # PREVIEW RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print(f"MANUAL PREVIEW — FIRST {len(previews)} FILES")
    print("=" * 70)

    for i, (
        src_path,
        dst_path,
        original_audio,
        processed_audio,
        original_duration,
        trim_start,
        trim_end,
        action,
        action_info
    ) in enumerate(previews, 1):

        print()
        print("-" * 70)
        print(f"FILE {i}")
        print("-" * 70)

        print(f"Name:             {src_path.name}")
        print(f"Original duration: {original_duration:.3f}s")
        print(
            f"Trimmed region:    "
            f"{trim_start:.3f}s → {trim_end:.3f}s"
        )
        print(f"Final operation:   {action}")

        if action_info:
            print(f"Details:           {action_info}")

        print(f"Output:            {dst_path}")

        print("\nORIGINAL:")
        display(
            Audio(
                original_audio,
                rate=TARGET_SR
            )
        )

        print("PROCESSED:")
        display(
            Audio(
                processed_audio,
                rate=TARGET_SR
            )
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()