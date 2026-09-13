from pathlib import Path

import librosa
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "model_v1" / "data"
AUGMENTED_DIR = PROJECT_ROOT / "model_v1" / "augmented" / "train"
FEATURE_DIR = PROJECT_ROOT / "model_v1" / "features"

SAMPLE_RATE = 16000

WINDOW_MS = 30
HOP_MS = 20

N_FFT = 480
N_MELS = 40

FMIN = 20
FMAX = 7600

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_MS / 1000)
HOP_SAMPLES = int(SAMPLE_RATE * HOP_MS / 1000)

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# AUDIO → LOG-MEL
# ============================================================

def extract_log_mel(filepath):

    audio, sr = librosa.load(
        filepath,
        sr=SAMPLE_RATE,
        mono=True
    )

    target_samples = SAMPLE_RATE

    if len(audio) < target_samples:

        padded = np.zeros(
            target_samples,
            dtype=np.float32
        )

        start = (target_samples - len(audio)) // 2

        padded[
            start:start + len(audio)
        ] = audio

        audio = padded

    elif len(audio) > target_samples:

        start = (len(audio) - target_samples) // 2

        audio = audio[
            start:start + target_samples
        ]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,

        n_fft=N_FFT,
        hop_length=HOP_SAMPLES,
        win_length=WINDOW_SAMPLES,

        window="hann",
        center=False,

        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,

        power=2.0
    )

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return log_mel.astype(np.float32)


# ============================================================
# GET FILES
# ============================================================

def get_files(split):

    files = []

    if split == "train":

        # Original + augmented training data
        for class_name, label in [
            ("positive", 1),
            ("negative", 0)
        ]:

            original_dir = DATA_DIR / "train" / class_name

            augmented_dir = (
                AUGMENTED_DIR / class_name
            )

            for path in sorted(
                original_dir.glob("*.wav")
            ):
                files.append(
                    (path, label)
                )

            for path in sorted(
                augmented_dir.glob("*.wav")
            ):
                files.append(
                    (path, label)
                )

    else:

        for class_name, label in [
            ("positive", 1),
            ("negative", 0)
        ]:

            directory = (
                DATA_DIR /
                split /
                class_name
            )

            for path in sorted(
                directory.glob("*.wav")
            ):
                files.append(
                    (path, label)
                )

    return files


# ============================================================
# PROCESS
# ============================================================

def process_split(split):

    print()
    print("=" * 60)
    print(f"PROCESSING {split.upper()}")
    print("=" * 60)

    files = get_files(split)

    print(f"Samples: {len(files)}")

    X = []
    y = []
    paths = []

    for index, (filepath, label) in enumerate(
        files,
        start=1
    ):

        feature = extract_log_mel(filepath)

        X.append(feature)
        y.append(label)
        paths.append(str(filepath))

        if index % 100 == 0 or index == len(files):
            print(
                f"Processed {index}/{len(files)}"
            )

    X = np.stack(X)

    y = np.asarray(
        y,
        dtype=np.int64
    )

    paths = np.asarray(
        paths
    )

    output_path = (
        FEATURE_DIR /
        f"{split}.npz"
    )

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        paths=paths
    )

    print()
    print(f"Feature shape: {X.shape}")
    print(f"Labels shape:  {y.shape}")
    print(f"Dtype:         {X.dtype}")
    print(
        f"Range:         "
        f"{X.min():.2f} to {X.max():.2f}"
    )
    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("VAANI MODEL V1 — LOG-MEL FEATURE EXTRACTION")
    print("=" * 60)

    print()
    print(f"Sample rate: {SAMPLE_RATE}")
    print(f"Window:      {WINDOW_MS} ms")
    print(f"Hop:         {HOP_MS} ms")
    print(f"FFT:         {N_FFT}")
    print(f"Mel bins:    {N_MELS}")

    process_split("train")
    process_split("validation")
    process_split("test")

    print()
    print("=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()