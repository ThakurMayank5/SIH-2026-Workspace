import os
import queue
import wave

import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "model_v1/exports/vaani_dscnn_v1.keras"

OUTPUT_AUDIO = "tests/asr_simulation.wav"

SAMPLE_RATE = 16000
CHANNELS = 1

# ------------------------------------------------------------
# VAD
# ------------------------------------------------------------

VAD_FRAME_MS = 30
VAD_FRAME_SAMPLES = int(
    SAMPLE_RATE * VAD_FRAME_MS / 1000
)

VAD_THRESHOLD_DB = -45.0

VAD_START_FRAMES = 2
VAD_END_FRAMES = 5

# ------------------------------------------------------------
# KWS
# ------------------------------------------------------------

KWS_WINDOW_SECONDS = 1.0

KWS_WINDOW_SAMPLES = int(
    SAMPLE_RATE * KWS_WINDOW_SECONDS
)

KWS_HOP_SECONDS = 0.20

KWS_HOP_SAMPLES = int(
    SAMPLE_RATE * KWS_HOP_SECONDS
)

KWS_THRESHOLD = 0.90

# ------------------------------------------------------------
# Pre-roll
# Audio immediately before detection that would be sent
# to ASR.
# ------------------------------------------------------------

PRE_ROLL_SECONDS = 0.5

PRE_ROLL_SAMPLES = int(
    SAMPLE_RATE * PRE_ROLL_SECONDS
)

# ------------------------------------------------------------
# After Vaani is detected, wait this long for silence before
# stopping the simulated ASR recording.
# ------------------------------------------------------------

END_SILENCE_MS = 1500

END_SILENCE_FRAMES = int(
    END_SILENCE_MS / VAD_FRAME_MS
)

# ------------------------------------------------------------
# V1 feature extraction
# ------------------------------------------------------------

N_FFT = 480
N_MELS = 40

WIN_LENGTH = int(
    0.030 * SAMPLE_RATE
)

HOP_LENGTH = int(
    0.020 * SAMPLE_RATE
)

FMIN = 20
FMAX = 7600


# ============================================================
# QUEUE
# ============================================================

audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):

    if status:
        print(f"\nAudio status: {status}")

    audio_queue.put(
        indata[:, 0].copy()
    )


# ============================================================
# AUDIO LEVEL
# ============================================================

def rms_dbfs(audio):

    if len(audio) == 0:
        return -120.0

    rms = np.sqrt(
        np.mean(audio ** 2)
    )

    if rms <= 1e-10:
        return -120.0

    return 20.0 * np.log10(rms)


# ============================================================
# VAD
# ============================================================

class VAD:

    def __init__(self):

        self.in_speech = False

        self.speech_count = 0
        self.silence_count = 0

    def process(self, frame):

        speech = (
            rms_dbfs(frame)
            > VAD_THRESHOLD_DB
        )

        if speech:

            self.speech_count += 1
            self.silence_count = 0

        else:

            self.silence_count += 1
            self.speech_count = 0

        if (
            not self.in_speech
            and self.speech_count
            >= VAD_START_FRAMES
        ):

            self.in_speech = True

        if (
            self.in_speech
            and self.silence_count
            >= VAD_END_FRAMES
        ):

            self.in_speech = False

        return self.in_speech


# ============================================================
# V1 FEATURE EXTRACTION
# ============================================================

def extract_features(audio):

    # Exactly 1 second

    if len(audio) < KWS_WINDOW_SAMPLES:

        missing = (
            KWS_WINDOW_SAMPLES
            - len(audio)
        )

        left = missing // 2
        right = missing - left

        audio = np.pad(
            audio,
            (left, right),
            mode="constant",
        )

    elif len(audio) > KWS_WINDOW_SAMPLES:

        start = (
            len(audio)
            - KWS_WINDOW_SAMPLES
        ) // 2

        audio = audio[
            start:start + KWS_WINDOW_SAMPLES
        ]

    # Exact V1 frontend

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

    features = np.clip(
        (log_mel + 80.0) / 80.0,
        0.0,
        1.0,
    )

    return features.astype(
        np.float32
    )


# ============================================================
# MODEL
# ============================================================

def load_model():

    print("Loading V1 model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded.")

    return model


def predict(model, audio):

    features = extract_features(audio)

    features = np.expand_dims(
        features,
        axis=-1,
    )

    features = np.expand_dims(
        features,
        axis=0,
    )

    prediction = model.predict(
        features,
        verbose=0,
    )

    return float(
        prediction[0][1]
    )


# ============================================================
# SAVE PCM16 WAV
# ============================================================

def save_wav(audio):

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = np.clip(
        audio,
        -1.0,
        1.0,
    )

    pcm16 = (
        audio * 32767
    ).astype(
        np.int16
    )

    os.makedirs(
        os.path.dirname(OUTPUT_AUDIO),
        exist_ok=True,
    )

    with wave.open(
        OUTPUT_AUDIO,
        "wb",
    ) as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(
            SAMPLE_RATE
        )

        wav.writeframes(
            pcm16.tobytes()
        )


# ============================================================
# MAIN
# ============================================================

def main():

    model = load_model()

    vad = VAD()

    # --------------------------------------------------------
    # 1-second KWS rolling buffer
    # --------------------------------------------------------

    kws_buffer = np.zeros(
        KWS_WINDOW_SAMPLES,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Pre-roll history
    # --------------------------------------------------------

    history = np.zeros(
        PRE_ROLL_SAMPLES,
        dtype=np.float32,
    )

    samples_since_kws = 0
    total_samples = 0

    # --------------------------------------------------------
    # IMPORTANT:
    # This remains empty until Vaani is detected.
    # --------------------------------------------------------

    asr_audio = []

    wake_detected = False

    silence_after_wake = 0

    print()
    print("=" * 65)
    print("V1 WAKE WORD → ASR CAPTURE SIMULATION")
    print("=" * 65)

    print()
    print("The microphone will listen continuously.")
    print()
    print("Before Vaani:")
    print("  Audio is NOT saved.")
    print()
    print("After Vaani:")
    print("  Audio is captured as if being sent to ASR.")
    print()
    print("After 1.5 sec of silence:")
    print("  Capture stops and the program exits.")
    print()
    print(f"Output: {OUTPUT_AUDIO}")
    print()

    print("Say:")
    print("    Vaani, <your command>")
    print()

    print("Listening...")
    print("-" * 65)

    try:

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=VAD_FRAME_SAMPLES,
            callback=audio_callback,
        ):

            while True:

                chunk = audio_queue.get()

                total_samples += len(chunk)

                # ====================================================
                # UPDATE HISTORY
                # ====================================================

                if len(chunk) >= PRE_ROLL_SAMPLES:

                    history = (
                        chunk[-PRE_ROLL_SAMPLES:]
                    )

                else:

                    history = np.concatenate(
                        [
                            history[len(chunk):],
                            chunk,
                        ]
                    )

                # ====================================================
                # AFTER WAKE WORD
                # ====================================================

                if wake_detected:

                    # Everything after detection goes to
                    # simulated ASR buffer.

                    asr_audio.append(
                        chunk.copy()
                    )

                    speech = vad.process(
                        chunk
                    )

                    if speech:

                        silence_after_wake = 0

                    else:

                        silence_after_wake += 1

                    print(
                        f"\r"
                        f"ASR CAPTURE | "
                        f"silence frames: "
                        f"{silence_after_wake:2d}/"
                        f"{END_SILENCE_FRAMES}",
                        end="",
                        flush=True,
                    )

                    # Stop after silence

                    if (
                        silence_after_wake
                        >= END_SILENCE_FRAMES
                    ):

                        print()
                        print()
                        print(
                            "Command finished."
                        )

                        break

                    continue

                # ====================================================
                # BEFORE WAKE WORD
                # ====================================================

                # Update KWS buffer

                kws_buffer = np.roll(
                    kws_buffer,
                    -len(chunk),
                )

                kws_buffer[
                    -len(chunk):
                ] = chunk

                samples_since_kws += len(chunk)

                # KWS every 200 ms

                if (
                    samples_since_kws
                    < KWS_HOP_SAMPLES
                ):

                    continue

                samples_since_kws = 0

                timestamp = (
                    total_samples
                    / SAMPLE_RATE
                )

                # ----------------------------------------------------
                # VAD
                # ----------------------------------------------------

                speech = vad.process(
                    chunk
                )

                if not speech:

                    print(
                        f"\r"
                        f"[{timestamp:6.2f}s] "
                        f"VAD=SILENCE  "
                        f"KWS=SKIP       ",
                        end="",
                        flush=True,
                    )

                    continue

                # ----------------------------------------------------
                # KWS
                # ----------------------------------------------------

                probability = predict(
                    model,
                    kws_buffer,
                )

                if probability >= KWS_THRESHOLD:

                    print()
                    print()
                    print(
                        "=" * 65
                    )

                    print(
                        f"VAANI DETECTED"
                    )

                    print(
                        f"Time       : "
                        f"{timestamp:.2f}s"
                    )

                    print(
                        f"Confidence : "
                        f"{probability:.4f}"
                    )

                    print(
                        "=" * 65
                    )

                    print()
                    print(
                        "Starting simulated ASR capture..."
                    )

                    # ------------------------------------------------
                    # THIS IS THE IMPORTANT PART
                    #
                    # Simulate what ESP32 would send:
                    #
                    #   pre-roll
                    #   +
                    #   current/ongoing audio
                    # ------------------------------------------------

                    asr_audio.append(
                        history.copy()
                    )

                    asr_audio.append(
                        kws_buffer[
                            -KWS_HOP_SAMPLES:
                        ].copy()
                    )

                    wake_detected = True

                    silence_after_wake = 0

                else:

                    print(
                        f"\r"
                        f"[{timestamp:6.2f}s] "
                        f"VAD=SPEECH  "
                        f"Vaani={probability:.4f}",
                        end="",
                        flush=True,
                    )

    except KeyboardInterrupt:

        print()
        print()
        print(
            "Manually stopped."
        )

        # Don't save anything unless Vaani was detected.
        if not wake_detected:

            print(
                "No Vaani detected."
            )

            return

    # ============================================================
    # SAVE ONLY ASR AUDIO
    # ============================================================

    if not wake_detected:

        print()
        print(
            "Vaani was never detected."
        )

        print(
            "No ASR audio was saved."
        )

        return

    if not asr_audio:

        print(
            "No ASR audio captured."
        )

        return

    audio = np.concatenate(
        asr_audio
    )

    save_wav(audio)

    duration = (
        len(audio)
        / SAMPLE_RATE
    )

    print()
    print("=" * 65)
    print("ASR AUDIO SAVED")
    print("=" * 65)

    print(
        f"File     : {OUTPUT_AUDIO}"
    )

    print(
        f"Duration : {duration:.2f}s"
    )

    print(
        "Format   : 16 kHz / mono / PCM16"
    )

    print()
    print(
        "This is the audio that would have"
    )

    print(
        "been sent to the ASR system."
    )


if __name__ == "__main__":
    main()