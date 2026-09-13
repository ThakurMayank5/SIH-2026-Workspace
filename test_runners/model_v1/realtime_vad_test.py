from pathlib import Path
import queue
import sys
import time

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

# Microphone callback block
FRAME_MS = 30
FRAME_SAMPLES = int(SR * FRAME_MS / 1000)

# KWS rolling window
KWS_WINDOW_SECONDS = 1.0
KWS_WINDOW_SAMPLES = int(SR * KWS_WINDOW_SECONDS)

# Run KWS every 200 ms
KWS_HOP_MS = 200
KWS_HOP_SAMPLES = int(SR * KWS_HOP_MS / 1000)

# KWS threshold
KWS_THRESHOLD = 0.90


# ============================================================
# VAD CONFIGURATION
# ============================================================

# Energy threshold in dBFS.
#
# Start with -45 dB.
#
# We print the measured level so this can be adjusted later.
#
# If quiet room is around -65 dB and speech is around -30 dB,
# -45 dB is a reasonable starting point.
#
# DO NOT assume this is the final value.
VAD_THRESHOLD_DB = -45.0


# Number of consecutive speech frames required before
# declaring speech active.
VAD_START_FRAMES = 2

# Number of consecutive silent frames required before
# declaring speech inactive.
VAD_END_FRAMES = 5


# ============================================================
# V1 FEATURE CONFIGURATION
# ============================================================

N_FFT = 480
WIN_LENGTH = int(0.030 * SR)       # 30 ms
HOP_LENGTH = int(0.020 * SR)       # 20 ms

N_MELS = 40
FMIN = 20
FMAX = 7600


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 72)
print("V1 REAL-TIME TEST WITH ENERGY VAD")
print("=" * 72)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print(f"Model     : {MODEL_PATH}")
print("Model     : loaded")


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio):

    # --------------------------------------------------------
    # Ensure exactly 1 second
    # --------------------------------------------------------

    if len(audio) < KWS_WINDOW_SAMPLES:

        audio = np.pad(
            audio,
            (0, KWS_WINDOW_SAMPLES - len(audio)),
            mode="constant",
        )

    elif len(audio) > KWS_WINDOW_SAMPLES:

        audio = audio[:KWS_WINDOW_SAMPLES]


    # --------------------------------------------------------
    # Mel spectrogram
    # SAME AS V1 TRAINING
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
    # V1 LOG-MEL
    # --------------------------------------------------------

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max,
    )


    # --------------------------------------------------------
    # V1 NORMALIZATION
    # --------------------------------------------------------

    features = (log_mel + 80.0) / 80.0

    features = np.clip(
        features,
        0.0,
        1.0,
    )

    features = features.astype(np.float32)


    # --------------------------------------------------------
    # Add channel + batch dimensions
    #
    # (40,49)
    #    ↓
    # (40,49,1)
    #    ↓
    # (1,40,49,1)
    # --------------------------------------------------------

    features = features[..., np.newaxis]
    features = features[np.newaxis, ...]

    return features


# ============================================================
# ENERGY / VAD
# ============================================================

def calculate_db(audio):

    """
    Calculate RMS level in dBFS.

    Float audio from sounddevice is normally in approximately
    [-1, +1].

    0 dBFS = maximum possible digital level.
    """

    rms = np.sqrt(
        np.mean(
            np.square(audio),
        )
    )

    if rms < 1e-10:
        return -100.0

    db = 20.0 * np.log10(rms)

    return float(db)


# ============================================================
# AUDIO QUEUE
# ============================================================

audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):

    if status:
        print(
            f"\nAudio status: {status}",
            file=sys.stderr,
        )

    # Copy because sounddevice reuses callback memory.
    audio_queue.put(
        indata[:, 0].copy()
    )


# ============================================================
# STATE
# ============================================================

kws_buffer = np.zeros(
    KWS_WINDOW_SAMPLES,
    dtype=np.float32,
)

samples_since_kws = 0

speech_frames = 0
silent_frames = 0

vad_active = False

total_kws_runs = 0
total_detections = 0

start_time = time.monotonic()


# ============================================================
# START
# ============================================================

print("\n" + "=" * 72)
print("CONFIGURATION")
print("=" * 72)

print(f"Sample rate       : {SR} Hz")
print(f"VAD frame         : {FRAME_MS} ms")
print(f"VAD threshold     : {VAD_THRESHOLD_DB:.1f} dBFS")
print(f"KWS window        : {KWS_WINDOW_SECONDS:.1f} sec")
print(f"KWS hop           : {KWS_HOP_MS} ms")
print(f"KWS threshold     : {KWS_THRESHOLD:.2f}")

print("\n" + "=" * 72)
print("REAL-TIME TEST")
print("=" * 72)

print("\nSpeak normally and say 'Vaani' when ready.")
print("Also test several seconds of complete silence.")
print("Press Ctrl+C to stop.\n")


# ============================================================
# MICROPHONE
# ============================================================

try:

    with sd.InputStream(
        samplerate=SR,
        channels=CHANNELS,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
        callback=audio_callback,
    ):

        while True:

            # =================================================
            # GET 30 ms FRAME
            # =================================================

            frame = audio_queue.get()

            # -------------------------------------------------
            # VAD ENERGY
            # -------------------------------------------------

            db = calculate_db(frame)

            frame_is_speech = (
                db >= VAD_THRESHOLD_DB
            )


            # -------------------------------------------------
            # VAD STATE MACHINE
            # -------------------------------------------------

            if frame_is_speech:

                speech_frames += 1
                silent_frames = 0

            else:

                silent_frames += 1
                speech_frames = 0


            # Speech starts after consecutive speech frames
            if (
                not vad_active
                and speech_frames >= VAD_START_FRAMES
            ):

                vad_active = True


            # Speech ends after consecutive silent frames
            if (
                vad_active
                and silent_frames >= VAD_END_FRAMES
            ):

                vad_active = False


            # =================================================
            # ROLLING KWS BUFFER
            # =================================================

            frame_length = len(frame)

            kws_buffer[:-frame_length] = (
                kws_buffer[frame_length:]
            )

            kws_buffer[-frame_length:] = frame

            samples_since_kws += frame_length


            # =================================================
            # KWS EVERY 200 ms
            # =================================================

            if samples_since_kws < KWS_HOP_SAMPLES:
                continue

            samples_since_kws = 0


            # Don't run until the rolling buffer is mostly full.
            if (
                np.count_nonzero(kws_buffer)
                < KWS_WINDOW_SAMPLES * 0.90
            ):
                continue


            # =================================================
            # IF NO SPEECH → SKIP KWS
            # =================================================

            if not vad_active:

                elapsed = time.monotonic() - start_time

                print(
                    f"\r"
                    f"Time {elapsed:6.1f}s | "
                    f"Level {db:7.1f} dB | "
                    f"VAD SILENCE | "
                    f"KWS SKIP       ",
                    end="",
                    flush=True,
                )

                continue


            # =================================================
            # SPEECH → RUN KWS
            # =================================================

            features = extract_features(
                kws_buffer.copy()
            )

            prediction = model.predict(
                features,
                verbose=0,
            )[0]

            p_negative = float(prediction[0])
            p_vaani = float(prediction[1])

            total_kws_runs += 1


            detected = (
                p_vaani >= KWS_THRESHOLD
            )

            if detected:
                total_detections += 1


            elapsed = time.monotonic() - start_time

            status_text = (
                "DETECT"
                if detected
                else "-"
            )


            # =================================================
            # PRINT
            # =================================================

            print(
                f"\r"
                f"Time {elapsed:6.1f}s | "
                f"Level {db:7.1f} dB | "
                f"VAD SPEECH  | "
                f"P(Vaani) {p_vaani:.4f} | "
                f"{status_text:<6}",
                end="",
                flush=True,
            )


# ============================================================
# STOP
# ============================================================

except KeyboardInterrupt:

    print("\n\nStopping microphone...")


except Exception as e:

    print("\n\nERROR:")
    print(type(e).__name__, ":", e)


finally:

    elapsed = time.monotonic() - start_time

    print("\n")
    print("=" * 72)
    print("REAL-TIME TEST SUMMARY")
    print("=" * 72)

    print(f"Runtime       : {elapsed:.1f} sec")
    print(f"KWS runs      : {total_kws_runs}")
    print(f"Raw detections: {total_detections}")

    print("\nTemporary files: none")
    print("=" * 72)