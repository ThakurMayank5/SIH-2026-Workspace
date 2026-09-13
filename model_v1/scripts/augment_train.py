from pathlib import Path
import random

import numpy as np
import librosa
import soundfile as sf


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "model_v1" / "data" / "train"
OUTPUT_DIR = PROJECT_ROOT / "model_v1" / "augmented" / "train"

SAMPLE_RATE = 16000
CLIP_SAMPLES = 16000

AUGMENTATIONS_PER_FILE = 2

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# AUDIO HELPERS
# ============================================================

def load_audio(path):
    audio, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )

    audio = audio.astype(np.float32)

    # Ensure exactly 1 second.
    if len(audio) < CLIP_SAMPLES:
        padded = np.zeros(CLIP_SAMPLES, dtype=np.float32)

        start = (CLIP_SAMPLES - len(audio)) // 2
        padded[start:start + len(audio)] = audio

        audio = padded

    elif len(audio) > CLIP_SAMPLES:
        start = (len(audio) - CLIP_SAMPLES) // 2
        audio = audio[start:start + CLIP_SAMPLES]

    return audio


def random_gain(audio):
    """
    Random volume change.
    Approximately -6 dB to +6 dB.
    """
    db = random.uniform(-6.0, 6.0)
    gain = 10 ** (db / 20.0)

    return audio * gain


def random_time_shift(audio):
    """
    Shift audio by up to +/- 80 ms.
    """
    max_shift = int(0.08 * SAMPLE_RATE)

    shift = random.randint(-max_shift, max_shift)

    output = np.zeros_like(audio)

    if shift > 0:
        output[shift:] = audio[:-shift]

    elif shift < 0:
        output[:shift] = audio[-shift:]

    else:
        output = audio.copy()

    return output


def add_noise(audio):
    """
    Add low-level synthetic background noise.
    """
    rms = np.sqrt(np.mean(audio ** 2) + 1e-10)

    # Random SNR between 10 and 25 dB.
    snr_db = random.uniform(10.0, 25.0)

    noise_rms = rms / (10 ** (snr_db / 20.0))

    noise = np.random.normal(
        0.0,
        noise_rms,
        size=audio.shape
    ).astype(np.float32)

    return audio + noise


def pitch_shift(audio):
    """
    Very mild pitch variation.
    """
    semitones = random.uniform(-1.0, 1.0)

    shifted = librosa.effects.pitch_shift(
        audio,
        sr=SAMPLE_RATE,
        n_steps=semitones
    )

    return shifted.astype(np.float32)


def add_reverb(audio):
    """
    Small artificial room response.
    """
    reverb_probability = 0.35

    if random.random() > reverb_probability:
        return audio

    length_ms = random.uniform(50, 150)
    length = int(SAMPLE_RATE * length_ms / 1000)

    decay = random.uniform(0.2, 0.5)

    impulse = np.zeros(length, dtype=np.float32)
    impulse[0] = 1.0

    for i in range(1, length):
        impulse[i] = (
            np.random.randn() *
            np.exp(-decay * i / length)
        )

    # Keep reverb relatively weak.
    impulse *= 0.15

    output = np.convolve(audio, impulse, mode="full")

    return output[:CLIP_SAMPLES].astype(np.float32)


def normalize(audio):
    """
    Prevent clipping while retaining augmentation effects.
    """
    peak = np.max(np.abs(audio))

    if peak > 0.98:
        audio = audio * (0.98 / peak)

    return audio.astype(np.float32)


# ============================================================
# AUGMENT ONE FILE
# ============================================================

def augment_audio(audio):

    # Every augmented example gets a random subset
    # of realistic transformations.

    if random.random() < 0.80:
        audio = random_gain(audio)

    if random.random() < 0.70:
        audio = random_time_shift(audio)

    if random.random() < 0.70:
        audio = add_noise(audio)

    if random.random() < 0.25:
        audio = pitch_shift(audio)

    if random.random() < 0.35:
        audio = add_reverb(audio)

    return normalize(audio)


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_class(class_name):

    input_dir = INPUT_DIR / class_name
    output_dir = OUTPUT_DIR / class_name

    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.wav"))

    print()
    print(f"{class_name.upper()}")
    print(f"Source files: {len(files)}")

    generated = 0

    for index, path in enumerate(files, start=1):

        audio = load_audio(path)

        for aug_index in range(AUGMENTATIONS_PER_FILE):

            augmented = augment_audio(audio)

            output_name = (
                f"{path.stem}_aug{aug_index + 1}.wav"
            )

            output_path = output_dir / output_name

            sf.write(
                output_path,
                augmented,
                SAMPLE_RATE,
                subtype="PCM_16"
            )

            generated += 1

        if index % 100 == 0:
            print(f"Processed {index}/{len(files)}")

    print(f"Generated: {generated}")

    return len(files), generated


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VAANI MODEL V1 — TRAINING AUGMENTATION")
    print("=" * 60)

    print()
    print("Original training data is NOT modified.")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    if OUTPUT_DIR.exists():
        print()
        print("Augmented training directory already exists.")
        print("Delete it manually if you want to regenerate it.")
        return

    total_original = 0
    total_generated = 0

    for class_name in ["positive", "negative"]:

        original, generated = process_class(class_name)

        total_original += original
        total_generated += generated

    print()
    print("=" * 60)
    print("AUGMENTATION COMPLETE")
    print("=" * 60)

    print()
    print(f"Original files:  {total_original}")
    print(f"Augmented files: {total_generated}")
    print(f"Total training:  {total_original + total_generated}")

    print()
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()