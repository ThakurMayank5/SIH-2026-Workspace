from pathlib import Path
import queue
import sys

import numpy as np
import sounddevice as sd
import tensorflow as tf
import librosa


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


# ============================================================
# AUDIO CONFIGURATION
# ============================================================

SR = 16000
CHANNELS = 1

# Model input
WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(SR * WINDOW_SECONDS)

# Run inference every 200 ms
HOP_SECONDS = 0.20
HOP_SAMPLES = int(SR * HOP_SECONDS)

# Detection threshold
THRESHOLD = 0.90


# ============================================================
# FEATURE CONFIGURATION
# SAME AS V1 TRAINING
# ============================================================

N_FFT = 480
WIN_LENGTH = int(0.030 * SR)      # 30 ms
HOP_LENGTH = int(0.020 * SR)      # 20 ms

N_MELS = 40
FMIN = 20
FMAX = 7600


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("V1 REAL-TIME KEYWORD TEST")
print("=" * 70)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print(f"Model: {MODEL_PATH}")
print("Model loaded.")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio):

    # --------------------------------------------------------
    # Ensure exactly 1 second
    # --------------------------------------------------------

    if len(audio) < WINDOW_SAMPLES:

        audio = np.pad(
            audio,
            (0, WINDOW_SAMPLES - len(audio)),
            mode="constant",
        )

    elif len(audio) > WINDOW_SAMPLES:

        audio = audio[:WINDOW_SAMPLES]


    # --------------------------------------------------------
    # Mel spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
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


    # --------------------------------------------------------
    # SAME V1 LOG-MEL CONVERSION
    # --------------------------------------------------------

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
    )


    # --------------------------------------------------------
    # SAME V1 NORMALIZATION
    # --------------------------------------------------------

    features = (log_mel + 80.0) / 80.0

    features = np.clip(
        features,
        0.0,
        1.0,
    )

    features = features.astype(np.float32)


    # --------------------------------------------------------
    # Add dimensions
    #
    # (40,49)
    #     ↓
    # (40,49,1)
    #     ↓
    # (1,40,49,1)
    # --------------------------------------------------------

    features = features[..., np.newaxis]
    features = features[np.newaxis, ...]

    return features


# ============================================================
# AUDIO QUEUE
# ============================================================

audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):

    if status:
        print(f"\nAudio status: {status}", file=sys.stderr)

    # Copy because sounddevice reuses the callback buffer
    audio_queue.put(
        indata[:, 0].copy()
    )


# ============================================================
# REAL-TIME PROCESSING
# ============================================================

audio_buffer = np.zeros(
    WINDOW_SAMPLES,
    dtype=np.float32,
)

samples_since_inference = 0

inference_count = 0

print("\n" + "=" * 70)
print("STARTING MICROPHONE")
print("=" * 70)

print(f"Sample rate : {SR} Hz")
print(f"Window      : {WINDOW_SECONDS:.1f} sec")
print(f"Hop         : {HOP_SECONDS:.1f} sec")
print(f"Threshold   : {THRESHOLD:.2f}")

print("\nSpeak into the microphone.")
print("Press Ctrl+C to stop.\n")


try:

    with sd.InputStream(
        samplerate=SR,
        channels=CHANNELS,
        dtype="float32",
        blocksize=HOP_SAMPLES,
        callback=audio_callback,
    ):

        while True:

            # ------------------------------------------------
            # Get next audio block
            # ------------------------------------------------

            block = audio_queue.get()

            block_length = len(block)


            # ------------------------------------------------
            # Rolling 1-second buffer
            # ------------------------------------------------

            audio_buffer[:-block_length] = (
                audio_buffer[block_length:]
            )

            audio_buffer[-block_length:] = block


            samples_since_inference += block_length


            # ------------------------------------------------
            # Wait until we have one complete window
            # ------------------------------------------------

            if samples_since_inference < HOP_SAMPLES:
                continue

            samples_since_inference = 0


            # ------------------------------------------------
            # Don't process until buffer has filled
            # ------------------------------------------------

            if inference_count == 0:

                # First inference needs a full second of audio.
                #
                # The buffer initially contains zeros, so skip
                # until approximately one second has arrived.

                if np.count_nonzero(audio_buffer) < WINDOW_SAMPLES * 0.9:
                    continue


            # ------------------------------------------------
            # Extract features
            # ------------------------------------------------

            features = extract_features(
                audio_buffer.copy()
            )


            # ------------------------------------------------
            # MODEL INFERENCE
            # ------------------------------------------------

            prediction = model.predict(
                features,
                verbose=0,
            )[0]

            p_negative = float(prediction[0])
            p_vaani = float(prediction[1])


            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            inference_count += 1

            status_text = (
                "DETECT"
                if p_vaani >= THRESHOLD
                else "-"
            )

            elapsed = (
                inference_count * HOP_SECONDS
            )

            print(
                f"\r"
                f"Time {elapsed:6.1f}s | "
                f"P(Vaani) {p_vaani:.4f} | "
                f"P(Neg) {p_negative:.4f} | "
                f"{status_text:<6}",
                end="",
                flush=True,
            )


except KeyboardInterrupt:

    print("\n\nStopping microphone...")


except Exception as e:

    print("\n\nERROR:")
    print(e)


finally:

    print("\n")
    print("=" * 70)
    print("REAL-TIME TEST COMPLETE")
    print("=" * 70)