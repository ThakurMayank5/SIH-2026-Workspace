import numpy as np
import wave

input_file = "test-servers/audio_chunk.raw"
output_file = "test-servers/audio.wav"

SAMPLE_RATE = 16000

# Read raw 32-bit samples
samples = np.fromfile(input_file, dtype=np.int32)

print("Samples:", len(samples))
print("Duration:", len(samples) / SAMPLE_RATE, "seconds")

samples = np.fromfile(input_file, dtype=np.int32)
samples = (samples >> 16).astype(np.int16)   # shift by 16, not 8 — no clipping needed

# Write WAV
with wave.open(output_file, "wb") as wav:
    wav.setnchannels(1)       # Mono
    wav.setsampwidth(2)       # 16-bit
    wav.setframerate(SAMPLE_RATE)
    wav.writeframes(samples.tobytes())

print("Created:", output_file)