"""
VAANI MODEL V1
Real Audio Wake-Word Diagnostic Test

Purpose:
    Test a real audio recording and print the model's
    probability for Vaani for EVERY sliding window.

NO OUTPUT FILES ARE CREATED.

Input:
    tests/audio1.m4a

Model:
    model_v1/exports/vaani_dscnn_v1.keras

Classes:
    0 = Negative
    1 = Vaani

V1 preprocessing:
    16 kHz mono
    40 Mel bins
    30 ms window
    20 ms hop
    480 FFT
    20 Hz - 7600 Hz
    log-Mel
    -80 ... 0 dB
    normalized to 0 ... 1

Threshold:
    0.90
"""

from pathlib import Path
import subprocess
import tempfile

import numpy as np
import librosa
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

# Project root:
# C:\Users\Mayank Singh\Codes\SIH 2026
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


# ============================================================
# V1 CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

WINDOW_SECONDS = 1.0

WINDOW_SAMPLES = int(
    SAMPLE_RATE * WINDOW_SECONDS
)

# Run the model every 200 ms.
HOP_SECONDS = 0.20

HOP_SAMPLES = int(
    SAMPLE_RATE * HOP_SECONDS
)

# Feature extraction
N_FFT = 480

WIN_LENGTH = int(
    SAMPLE_RATE * 0.030
)

HOP_LENGTH = int(
    SAMPLE_RATE * 0.020
)

N_MELS = 40

FMIN = 20
FMAX = 7600

# Current V1 operating threshold
THRESHOLD = 0.90


# ============================================================
# CHECK FILES
# ============================================================

def check_files():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not AUDIO_PATH.exists():

        raise FileNotFoundError(
            f"Audio file not found:\n{AUDIO_PATH}"
        )


# ============================================================
# CONVERT M4A → TEMPORARY WAV
# ============================================================

def convert_to_wav():

    """
    Converts the input M4A to:

        16 kHz
        mono
        PCM16 WAV

    The WAV is stored inside a temporary directory and is
    automatically deleted when the program finishes.
    """

    temp_dir = tempfile.TemporaryDirectory()

    wav_path = (
        Path(temp_dir.name)
        / "audio_16k_mono.wav"
    )

    command = [

        "ffmpeg",

        "-y",

        "-i",
        str(AUDIO_PATH),

        "-ar",
        str(SAMPLE_RATE),

        "-ac",
        "1",

        "-sample_fmt",
        "s16",

        str(wav_path)
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        print(result.stderr)

        temp_dir.cleanup()

        raise RuntimeError(
            "FFmpeg failed to convert the audio."
        )

    return temp_dir, wav_path


# ============================================================
# LOAD AUDIO
# ============================================================

def load_audio(wav_path):

    audio, sr = librosa.load(
        wav_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) == 0:

        raise ValueError(
            "Audio contains no samples."
        )

    duration = (
        len(audio) / SAMPLE_RATE
    )

    print()
    print("=" * 60)
    print("AUDIO")
    print("=" * 60)

    print(
        f"Sample rate : {sr} Hz"
    )

    print(
        f"Samples     : {len(audio):,}"
    )

    print(
        f"Duration    : {duration:.2f} seconds"
    )

    return audio


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_window):

    """
    Extract features exactly as V1 training does.
    """

    # --------------------------------------------------------
    # Ensure exactly 1 second
    # --------------------------------------------------------

    if len(audio_window) < WINDOW_SAMPLES:

        padded = np.zeros(
            WINDOW_SAMPLES,
            dtype=np.float32
        )

        padded[
            :len(audio_window)
        ] = audio_window

        audio_window = padded

    elif len(audio_window) > WINDOW_SAMPLES:

        audio_window = (
            audio_window[:WINDOW_SAMPLES]
        )

    # --------------------------------------------------------
    # Mel spectrogram
    # --------------------------------------------------------

    mel = librosa.feature.melspectrogram(

        y=audio_window,

        sr=SAMPLE_RATE,

        n_fft=N_FFT,

        win_length=WIN_LENGTH,

        hop_length=HOP_LENGTH,

        n_mels=N_MELS,

        fmin=FMIN,

        fmax=FMAX,

        window="hann",

        center=False,

        power=2.0
    )

    # --------------------------------------------------------
    # Convert to dB
    # --------------------------------------------------------

    log_mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # --------------------------------------------------------
    # Same normalization as training
    #
    # -80 ... 0
    #      ↓
    # 0 ... 1
    # --------------------------------------------------------

    log_mel = (
        log_mel + 80.0
    ) / 80.0

    log_mel = np.clip(
        log_mel,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Expected shape:
    #
    # (40, 49)
    # --------------------------------------------------------

    if log_mel.shape != (40, 49):

        raise ValueError(
            f"Unexpected feature shape: "
            f"{log_mel.shape}. "
            f"Expected (40, 49)."
        )

    # --------------------------------------------------------
    # Add:
    #
    # batch dimension
    # channel dimension
    #
    # (40,49)
    #    ↓
    # (1,40,49,1)
    # --------------------------------------------------------

    features = log_mel.astype(
        np.float32
    )

    features = features[
        np.newaxis,
        ...,
        np.newaxis
    ]

    return features


# ============================================================
# TIME FORMAT
# ============================================================

def format_time(seconds):

    minutes = int(
        seconds // 60
    )

    remaining = (
        seconds % 60
    )

    return (
        f"{minutes:02d}:"
        f"{remaining:05.2f}"
    )


# ============================================================
# RUN DIAGNOSTIC
# ============================================================

def run_diagnostic(
    model,
    audio
):

    print()
    print("=" * 78)
    print("PROBABILITY TIMELINE")
    print("=" * 78)

    print()

    print(
        f"{'Window':>10} "
        f"{'Start':>10} "
        f"{'End':>10} "
        f"{'P(Vaani)':>12} "
        f"{'P(Negative)':>14} "
        f"{'Status':>10}"
    )

    print("-" * 78)

    total_samples = len(audio)

    # --------------------------------------------------------
    # Generate sliding windows
    # --------------------------------------------------------

    if total_samples <= WINDOW_SAMPLES:

        starts = [0]

    else:

        starts = range(
            0,
            total_samples - WINDOW_SAMPLES + 1,
            HOP_SAMPLES
        )

    detection_count = 0

    probabilities = []

    # --------------------------------------------------------
    # Process every window
    # --------------------------------------------------------

    for window_number, start_sample in enumerate(
        starts,
        start=1
    ):

        end_sample = (
            start_sample +
            WINDOW_SAMPLES
        )

        window = audio[
            start_sample:end_sample
        ]

        # -----------------------------------------------
        # Feature extraction
        # -----------------------------------------------

        features = extract_features(
            window
        )

        # -----------------------------------------------
        # Model inference
        # -----------------------------------------------

        output = model.predict(
            features,
            verbose=0
        )[0]

        # Class mapping:
        #
        # 0 = Negative
        # 1 = Vaani

        negative_probability = float(
            output[0]
        )

        vaani_probability = float(
            output[1]
        )

        # -----------------------------------------------
        # Time
        # -----------------------------------------------

        start_time = (
            start_sample /
            SAMPLE_RATE
        )

        end_time = (
            min(
                end_sample,
                total_samples
            )
            /
            SAMPLE_RATE
        )

        # -----------------------------------------------
        # Threshold
        # -----------------------------------------------

        if (
            vaani_probability
            >= THRESHOLD
        ):

            status = "DETECT"

            detection_count += 1

        else:

            status = "-"

        probabilities.append(
            vaani_probability
        )

        # -----------------------------------------------
        # Print
        # -----------------------------------------------

        print(
            f"{window_number:>10} "
            f"{format_time(start_time):>10} "
            f"{format_time(end_time):>10} "
            f"{vaani_probability:>12.6f} "
            f"{negative_probability:>14.6f} "
            f"{status:>10}"
        )

    return probabilities


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    probabilities
):

    probabilities = np.array(
        probabilities
    )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print()

    print(
        f"Windows evaluated : "
        f"{len(probabilities)}"
    )

    print(
        f"Threshold         : "
        f"{THRESHOLD:.2f}"
    )

    print(
        f"Windows >= threshold : "
        f"{np.sum(probabilities >= THRESHOLD)}"
    )

    print()

    print(
        f"Minimum P(Vaani) : "
        f"{probabilities.min():.6f}"
    )

    print(
        f"Maximum P(Vaani) : "
        f"{probabilities.max():.6f}"
    )

    print(
        f"Mean P(Vaani)    : "
        f"{probabilities.mean():.6f}"
    )

    # --------------------------------------------------------
    # Highest scoring windows
    # --------------------------------------------------------

    print()
    print("Top 5 highest-probability windows:")

    print()

    top_indices = np.argsort(
        probabilities
    )[-5:][::-1]

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        start_time = (
            index *
            HOP_SECONDS
        )

        end_time = (
            start_time +
            WINDOW_SECONDS
        )

        probability = (
            probabilities[index]
        )

        print(
            f"{rank}. "
            f"{format_time(start_time)} - "
            f"{format_time(end_time)}  "
            f"P(Vaani) = "
            f"{probability:.6f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("VAANI DS-CNN V1 - CONTINUOUS AUDIO DIAGNOSTIC")
    print("=" * 72)

    print()
    print(
        "No output files will be created."
    )

    print(
        f"Audio: {AUDIO_PATH}"
    )

    print(
        f"Threshold: {THRESHOLD:.2f}"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    check_files()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print()
    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Convert audio
    # --------------------------------------------------------

    temp_dir, wav_path = (
        convert_to_wav()
    )

    try:

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        audio = load_audio(
            wav_path
        )

        # ----------------------------------------------------
        # Run diagnostic
        # ----------------------------------------------------

        probabilities = (
            run_diagnostic(
                model,
                audio
            )
        )

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        print_summary(
            probabilities
        )

    finally:

        # Temporary WAV is deleted here.
        temp_dir.cleanup()

    print()
    print("=" * 72)
    print("TEST COMPLETE")
    print("=" * 72)


if __name__ == "__main__":

    main()