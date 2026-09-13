from pathlib import Path
import subprocess
import numpy as np
import librosa
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "model_v1"
    / "exports"
    / "vaani_dscnn_v1.keras"
)

AUDIO_PATH = (
    PROJECT_ROOT
    / "tests"
    / "silence.m4a"
)

TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

TEMP_WAV = TEMP_DIR / "silence.wav"


# ============================================================
# CONFIG
# ============================================================

SR = 16000

WINDOW_SAMPLES = SR
HOP_SAMPLES = int(0.2 * SR)

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

print("=" * 80)
print("SILENCE SIGNAL + KWS ANALYSIS")
print("=" * 80)

model = tf.keras.models.load_model(MODEL_PATH)


try:

    # ========================================================
    # CONVERT
    # ========================================================

    print("\nConverting audio...")

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
    # LOAD
    # ========================================================

    audio, _ = librosa.load(
        TEMP_WAV,
        sr=SR,
        mono=True,
    )

    print(f"Duration: {len(audio) / SR:.2f} seconds")


    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    def extract_features(y):

        y = y[:WINDOW_SAMPLES]

        if len(y) < WINDOW_SAMPLES:

            y = np.pad(
                y,
                (0, WINDOW_SAMPLES - len(y)),
                mode="constant",
            )

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

        log_mel = librosa.power_to_db(
            mel,
            ref=np.max,
        )

        features = (log_mel + 80.0) / 80.0

        features = np.clip(
            features,
            0.0,
            1.0,
        )

        features = features.astype(np.float32)

        features = features[..., np.newaxis]
        features = features[np.newaxis, ...]

        return features


    # ========================================================
    # ANALYSIS
    # ========================================================

    print("\n" + "=" * 80)
    print("WINDOW ANALYSIS")
    print("=" * 80)

    print()
    print(
        f"{'Start':>8} "
        f"{'End':>8} "
        f"{'RMS':>12} "
        f"{'dBFS':>9} "
        f"{'Peak':>10} "
        f"{'P(Vaani)':>12} "
        f"{'Status':>10}"
    )

    print("-" * 80)


    results = []

    start = 0

    while start + WINDOW_SAMPLES <= len(audio):

        end = start + WINDOW_SAMPLES

        window = audio[start:end]

        # ----------------------------------------------------
        # SIGNAL LEVEL
        # ----------------------------------------------------

        rms = np.sqrt(
            np.mean(
                window ** 2
            )
        )

        if rms < 1e-10:
            dbfs = -100.0
        else:
            dbfs = 20 * np.log10(rms)

        peak = np.max(
            np.abs(window)
        )


        # ----------------------------------------------------
        # KWS
        # ----------------------------------------------------

        features = extract_features(window)

        prediction = model.predict(
            features,
            verbose=0,
        )[0]

        p_vaani = float(prediction[1])

        status = (
            "FALSE FIRE"
            if p_vaani >= THRESHOLD
            else "-"
        )


        print(
            f"{start / SR:8.2f} "
            f"{end / SR:8.2f} "
            f"{rms:12.7f} "
            f"{dbfs:9.2f} "
            f"{peak:10.7f} "
            f"{p_vaani:12.6f} "
            f"{status:>10}"
        )


        results.append(
            (
                start / SR,
                end / SR,
                rms,
                dbfs,
                peak,
                p_vaani,
            )
        )


        start += HOP_SAMPLES


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    db_values = [r[3] for r in results]
    p_values = [r[5] for r in results]

    false_fires = sum(
        p >= THRESHOLD
        for p in p_values
    )

    print(
        f"Windows analyzed : {len(results)}"
    )

    print(
        f"False fires      : {false_fires}"
    )

    print(
        f"Average level    : {np.mean(db_values):.2f} dBFS"
    )

    print(
        f"Min level        : {np.min(db_values):.2f} dBFS"
    )

    print(
        f"Max level        : {np.max(db_values):.2f} dBFS"
    )

    print(
        f"Average P(Vaani) : {np.mean(p_values):.6f}"
    )

    print(
        f"Max P(Vaani)     : {np.max(p_values):.6f}"
    )

    print("\nTemporary WAV deleted.")


finally:

    if TEMP_WAV.exists():
        TEMP_WAV.unlink()