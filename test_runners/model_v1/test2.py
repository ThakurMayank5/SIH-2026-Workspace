from pathlib import Path
import subprocess
import numpy as np
import librosa
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "model_v1" / "exports" / "vaani_dscnn_v1.keras"
AUDIO_PATH = PROJECT_ROOT / "tests" / "silence.m4a"

TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

TEMP_WAV = TEMP_DIR / "audio.wav"


# ============================================================
# CONFIG
# ============================================================

SR = 16000

WINDOW_SECONDS = 1.0

N_FFT = 480
WIN_LENGTH = int(0.030 * SR)
HOP_LENGTH = int(0.020 * SR)

N_MELS = 40
FMIN = 20
FMAX = 7600

THRESHOLD = 0.90


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = tf.keras.models.load_model(MODEL_PATH)

print(f"Model: {MODEL_PATH}")


try:

    # ========================================================
    # CONVERT AUDIO
    # ========================================================

    print("\n" + "=" * 70)
    print("CONVERTING AUDIO")
    print("=" * 70)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(AUDIO_PATH),
        "-ar",
        str(SR),
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(TEMP_WAV),
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    # ========================================================
    # LOAD COMPLETE AUDIO
    # ========================================================

    audio, sr = librosa.load(
        TEMP_WAV,
        sr=SR,
        mono=True,
    )

    duration = len(audio) / SR

    print(f"Sample rate : {sr} Hz")
    print(f"Samples     : {len(audio):,}")
    print(f"Duration    : {duration:.2f} seconds")


    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def extract_features(y):

        target_samples = int(WINDOW_SECONDS * SR)

        # Exactly 1 second
        if len(y) < target_samples:

            y = np.pad(
                y,
                (0, target_samples - len(y)),
                mode="constant",
            )

        elif len(y) > target_samples:

            y = y[:target_samples]


        mel = librosa.feature.melspectrogram(
            y=y,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window="hann",
            center=False,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
            power=2.0,
        )

        # CURRENT V1 METHOD
        log_mel = librosa.power_to_db(
            mel,
            ref=np.max,
        )

        # SAME NORMALIZATION AS TRAINING
        features = (log_mel + 80.0) / 80.0
        features = np.clip(features, 0.0, 1.0)

        features = features.astype(np.float32)

        # (40, 49) -> (40, 49, 1)
        features = features[..., np.newaxis]

        # (40, 49, 1) -> (1, 40, 49, 1)
        features = features[np.newaxis, ...]

        return features


    # ========================================================
    # INDEPENDENT 1-SECOND SEGMENTS
    # ========================================================

    print("\n" + "=" * 70)
    print("INDEPENDENT 1-SECOND SEGMENT TEST")
    print("=" * 70)

    print()
    print(
        f"{'Segment':>8} "
        f"{'Start':>10} "
        f"{'End':>10} "
        f"{'P(Vaani)':>12} "
        f"{'P(Negative)':>14} "
        f"{'Status':>10}"
    )

    print("-" * 70)


    # Test several independent 1-second segments.
    #
    # These are deliberately chosen around the regions
    # where the previous sliding-window test behaved
    # differently.

    segments = [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 3.0),
        (3.0, 4.0),
        (4.0, 5.0),
        (5.0, 6.0),
        (6.0, 7.0),
        (7.0, 8.0),
    ]


    for i, (start, end) in enumerate(segments, 1):

        start_sample = int(start * SR)
        end_sample = int(end * SR)

        segment = audio[start_sample:end_sample]

        features = extract_features(segment)

        prediction = model.predict(
            features,
            verbose=0,
        )[0]

        p_negative = float(prediction[0])
        p_vaani = float(prediction[1])

        status = "DETECT" if p_vaani >= THRESHOLD else "-"

        print(
            f"{i:>8} "
            f"{start:>10.2f} "
            f"{end:>10.2f} "
            f"{p_vaani:>12.6f} "
            f"{p_negative:>14.6f} "
            f"{status:>10}"
        )


    # ========================================================
    # FEATURE SHAPE CHECK
    # ========================================================

    test_features = extract_features(audio[:SR])

    print("\n" + "=" * 70)
    print("FEATURE CHECK")
    print("=" * 70)

    print(f"Feature shape : {test_features.shape}")
    print(f"Feature range : {test_features.min():.6f} to {test_features.max():.6f}")


finally:

    # ========================================================
    # CLEAN TEMP FILE
    # ========================================================

    if TEMP_WAV.exists():
        TEMP_WAV.unlink()

    print("\nTemporary WAV deleted.")