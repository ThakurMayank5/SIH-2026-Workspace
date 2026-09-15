import os
import sys
import subprocess
import tempfile

import numpy as np
import librosa
import tensorflow as tf


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "model_v1/exports/vaani_dscnn_v1.keras"

# Change this to your audio file
AUDIO_PATH = "tests/silence.m4a"

THRESHOLD = 0.90

# Audio
SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)

# Run KWS every 200 ms
HOP_SECONDS = 0.20
HOP_SAMPLES = int(SAMPLE_RATE * HOP_SECONDS)

# VAD
VAD_FRAME_MS = 30
VAD_FRAME_SAMPLES = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)

VAD_THRESHOLD_DB = -45.0

VAD_START_FRAMES = 2
VAD_END_FRAMES = 5

# Feature extraction
N_FFT = 480
N_MELS = 40
WIN_LENGTH = int(0.030 * SAMPLE_RATE)   # 30 ms
HOP_LENGTH = int(0.020 * SAMPLE_RATE)   # 20 ms
FMIN = 20
FMAX = 7600


# ============================================================
# AUDIO CONVERSION
# ============================================================

def convert_to_wav(input_path, output_path):
    """
    Convert arbitrary audio to:
        16 kHz
        mono
        PCM 16-bit WAV
    """

    command = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-sample_fmt", "s16",
        output_path,
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(path):
    audio, sr = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
    )

    return audio.astype(np.float32)


# ============================================================
# VAD
# ============================================================

def rms_dbfs(audio):
    """
    Calculate RMS level in dBFS.
    """

    if len(audio) == 0:
        return -120.0

    rms = np.sqrt(np.mean(audio ** 2))

    if rms <= 1e-10:
        return -120.0

    return 20.0 * np.log10(rms)


def vad_detect(audio):
    """
    Frame-based energy VAD.

    Returns:
        speech_detected
    """

    if len(audio) < VAD_FRAME_SAMPLES:
        return rms_dbfs(audio) > VAD_THRESHOLD_DB

    speech_frames = 0
    silent_frames = 0
    in_speech = False

    num_frames = len(audio) // VAD_FRAME_SAMPLES

    for i in range(num_frames):

        start = i * VAD_FRAME_SAMPLES
        end = start + VAD_FRAME_SAMPLES

        frame = audio[start:end]

        db = rms_dbfs(frame)

        is_speech = db > VAD_THRESHOLD_DB

        if is_speech:

            speech_frames += 1
            silent_frames = 0

        else:

            silent_frames += 1
            speech_frames = 0

        # Speech starts after consecutive speech frames
        if not in_speech and speech_frames >= VAD_START_FRAMES:
            in_speech = True

        # Speech ends after consecutive silent frames
        if in_speech and silent_frames >= VAD_END_FRAMES:
            in_speech = False

    return in_speech


# ============================================================
# FEATURE EXTRACTION
# EXACTLY V1 FRONTEND
# ============================================================

def extract_features(audio):
    """
    V1 Log-Mel feature extraction.

    Output:
        (40, 49)
    """

    # --------------------------------------------------------
    # Make exactly 1 second
    # --------------------------------------------------------

    if len(audio) < WINDOW_SAMPLES:

        pad_total = WINDOW_SAMPLES - len(audio)

        left = pad_total // 2
        right = pad_total - left

        audio = np.pad(
            audio,
            (left, right),
            mode="constant",
        )

    elif len(audio) > WINDOW_SAMPLES:

        # Center crop
        start = (len(audio) - WINDOW_SAMPLES) // 2

        audio = audio[
            start:start + WINDOW_SAMPLES
        ]

    # --------------------------------------------------------
    # V1 Log-Mel
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
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

    # V1 normalization
    features = np.clip(
        (log_mel + 80.0) / 80.0,
        0.0,
        1.0,
    )

    return features.astype(np.float32)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("Loading V1 model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded.")
    print()

    return model


# ============================================================
# PREDICTION
# ============================================================

def predict(model, audio):

    features = extract_features(audio)

    # (40,49)
    features = np.expand_dims(features, axis=-1)

    # (1,40,49,1)
    features = np.expand_dims(features, axis=0)

    prediction = model.predict(
        features,
        verbose=0,
    )

    # Class order:
    # 0 = negative
    # 1 = Vaani

    probability = float(prediction[0][1])

    return probability


# ============================================================
# MAIN PROCESSING
# ============================================================

def process_audio(audio):

    model = load_model()

    duration = len(audio) / SAMPLE_RATE

    print(f"Audio duration : {duration:.2f} sec")
    print(f"VAD threshold  : {VAD_THRESHOLD_DB} dBFS")
    print(f"KWS threshold  : {THRESHOLD}")
    print(f"KWS hop        : {HOP_SECONDS * 1000:.0f} ms")
    print()

    print("=" * 60)
    print("PROCESSING")
    print("=" * 60)

    detections = []

    # --------------------------------------------------------
    # Sliding 1-second window
    # --------------------------------------------------------

    position = 0

    while position < len(audio):

        window_start = position
        window_end = position + WINDOW_SAMPLES

        window = audio[
            window_start:window_end
        ]

        # Ignore extremely short final window
        if len(window) < int(0.5 * SAMPLE_RATE):
            break

        timestamp = window_start / SAMPLE_RATE

        # ----------------------------------------------------
        # VAD
        # ----------------------------------------------------

        vad = vad_detect(window)

        if not vad:

            print(
                f"[{timestamp:6.2f}s] "
                f"VAD=NO SPEECH   KWS=SKIP"
            )

            position += HOP_SAMPLES
            continue

        # ----------------------------------------------------
        # KWS
        # ----------------------------------------------------

        probability = predict(
            model,
            window,
        )

        if probability >= THRESHOLD:

            print(
                f"[{timestamp:6.2f}s] "
                f"VAD=SPEECH     "
                f"Vaani={probability:.4f}  "
                f"*** DETECTED ***"
            )

            detections.append(
                (
                    timestamp,
                    probability,
                )
            )

        else:

            print(
                f"[{timestamp:6.2f}s] "
                f"VAD=SPEECH     "
                f"Vaani={probability:.4f}"
            )

        position += HOP_SAMPLES

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    if not detections:

        print("No Vaani detections.")

    else:

        print(
            f"Detections: {len(detections)}"
        )

        print()

        for timestamp, probability in detections:

            print(
                f"Vaani detected at "
                f"{timestamp:.2f}s "
                f"(confidence={probability:.4f})"
            )


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    global AUDIO_PATH

    # --------------------------------------------------------
    # Allow:
    #
    # python test_custom_audio_vad.py my_audio.m4a
    # --------------------------------------------------------

    if len(sys.argv) >= 2:

        AUDIO_PATH = sys.argv[1]

    if not os.path.exists(AUDIO_PATH):

        print(
            f"ERROR: Audio file not found:\n"
            f"{AUDIO_PATH}"
        )

        sys.exit(1)

    if not os.path.exists(MODEL_PATH):

        print(
            f"ERROR: V1 model not found:\n"
            f"{MODEL_PATH}"
        )

        sys.exit(1)

    print("=" * 60)
    print("V1 Vaani KWS + VAD TEST")
    print("=" * 60)

    print()
    print(f"Input audio: {AUDIO_PATH}")

    # Temporary converted WAV
    with tempfile.TemporaryDirectory() as temp_dir:

        converted_wav = os.path.join(
            temp_dir,
            "input_16k_mono.wav",
        )

        print()
        print("Converting audio to 16 kHz mono...")

        convert_to_wav(
            AUDIO_PATH,
            converted_wav,
        )

        print("Conversion complete.")
        print()

        audio = load_audio(
            converted_wav
        )

        process_audio(audio)


if __name__ == "__main__":
    main()