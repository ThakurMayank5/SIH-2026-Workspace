from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from IPython.display import Audio, display


# ============================================================
# CHANGE THESE TWO PATHS
# ============================================================

INPUT_AUDIO = r"C:\Users\Mayank Singh\Codes\SIH 2026\dataset\raw\SIH_Vaani_Dataset_Converted\Mayank\Recording (2).wav"

OUTPUT_AUDIO = r"C:\Users\Mayank Singh\Codes\SIH 2026\temp\test_positive.wav"


# ============================================================
# SETTINGS
# ============================================================

TARGET_SR = 16000
TARGET_SAMPLES = TARGET_SR       # exactly 1 second
TOP_DB = 30
TARGET_PEAK = 0.95


# ============================================================
# LOAD ORIGINAL
# ============================================================

print("=" * 60)
print("ORIGINAL AUDIO")
print("=" * 60)

audio, sr = librosa.load(
    INPUT_AUDIO,
    sr=TARGET_SR,
    mono=True
)

print(f"Input:        {INPUT_AUDIO}")
print(f"Sample rate:  {sr} Hz")
print(f"Samples:      {len(audio)}")
print(f"Duration:     {len(audio) / sr:.3f} sec")
print(f"Peak:         {np.max(np.abs(audio)):.4f}")

print("\nPlaying ORIGINAL:")
display(Audio(audio, rate=sr))


# ============================================================
# TRIM SILENCE
# ============================================================

trimmed, trim_indices = librosa.effects.trim(
    audio,
    top_db=TOP_DB
)

start_sample, end_sample = trim_indices

print()
print("=" * 60)
print("AFTER SILENCE TRIMMING")
print("=" * 60)

print(f"Trim start:   {start_sample / sr:.3f} sec")
print(f"Trim end:     {end_sample / sr:.3f} sec")
print(f"Duration:     {len(trimmed) / sr:.3f} sec")

audio = trimmed


# ============================================================
# NORMALIZE
# ============================================================

peak = np.max(np.abs(audio))

if peak > 0:
    audio = audio * (TARGET_PEAK / peak)

print()
print("=" * 60)
print("AFTER NORMALIZATION")
print("=" * 60)

print(f"Peak:         {np.max(np.abs(audio)):.4f}")


# ============================================================
# FORCE EXACTLY 1 SECOND
# ============================================================

if len(audio) < TARGET_SAMPLES:

    # --------------------------------------------------------
    # SHORTER THAN 1 SECOND
    # Center-pad with silence
    # --------------------------------------------------------

    pad_total = TARGET_SAMPLES - len(audio)

    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    audio = np.pad(
        audio,
        (pad_left, pad_right),
        mode="constant"
    )

    print()
    print("=" * 60)
    print("1-SECOND PREPARATION")
    print("=" * 60)

    print("Action:       CENTER PAD")
    print(f"Left pad:     {pad_left / sr:.3f} sec")
    print(f"Right pad:    {pad_right / sr:.3f} sec")

elif len(audio) > TARGET_SAMPLES:

    # --------------------------------------------------------
    # LONGER THAN 1 SECOND
    # Center-crop
    # --------------------------------------------------------

    crop_total = len(audio) - TARGET_SAMPLES

    crop_left = crop_total // 2
    crop_right = crop_left + TARGET_SAMPLES

    audio = audio[
        crop_left:crop_right
    ]

    print()
    print("=" * 60)
    print("1-SECOND PREPARATION")
    print("=" * 60)

    print("Action:       CENTER CROP")
    print(f"Crop start:   {crop_left / sr:.3f} sec")
    print(f"Crop end:     {crop_right / sr:.3f} sec")

else:

    print()
    print("=" * 60)
    print("1-SECOND PREPARATION")
    print("=" * 60)

    print("Action:       ALREADY 1 SECOND")


# ============================================================
# FINAL SAFETY
# ============================================================

audio = np.asarray(
    audio,
    dtype=np.float32
)

audio = np.clip(
    audio,
    -1.0,
    1.0
)


# ============================================================
# SAVE
# ============================================================

output_path = Path(OUTPUT_AUDIO)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

sf.write(
    str(output_path),
    audio,
    TARGET_SR,
    subtype="PCM_16"
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print()
print("=" * 60)
print("FINAL AUDIO")
print("=" * 60)

print(f"Output:       {output_path}")
print(f"Sample rate:  {TARGET_SR} Hz")
print(f"Samples:      {len(audio)}")
print(f"Duration:     {len(audio) / TARGET_SR:.3f} sec")
print(f"Peak:         {np.max(np.abs(audio)):.4f}")

print()
print("Playing PROCESSED AUDIO:")
display(Audio(audio, rate=TARGET_SR))

print()
print("=" * 60)
print("DONE")
print("=" * 60)