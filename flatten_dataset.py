import os
import glob
import shutil
import soundfile as sf


import os

print("Current directory:")
print(os.getcwd())

print("\nContents of dataset/raw:")
print(os.listdir("dataset/raw"))

RAW = "dataset/raw"
SOURCE = os.path.join(RAW, "archive")

NEGATIVE_SPEECH = os.path.join(
    RAW, "negative_speech_commands"
)

NEGATIVE_BACKGROUND = os.path.join(
    RAW, "negative_background"
)

os.makedirs(NEGATIVE_SPEECH, exist_ok=True)
os.makedirs(NEGATIVE_BACKGROUND, exist_ok=True)


# ------------------------------------------------------------
# 1. COPY ALL SPEECH COMMAND WORDS
# ------------------------------------------------------------

speech_count = 0

for word in sorted(os.listdir(SOURCE)):

    word_folder = os.path.join(SOURCE, word)

    if not os.path.isdir(word_folder):
        continue

    print(f"Processing: {word}")

    for wav_path in glob.glob(
        os.path.join(word_folder, "*.wav")
    ):

        filename = os.path.basename(wav_path)
        stem = os.path.splitext(filename)[0]

        # Original:
        # 0a7c2a8d_nohash_0
        #
        # New:
        # 0a7c2a8d_yes_nohash_0

        new_filename = f"{word}_{filename}"

        destination = os.path.join(
            NEGATIVE_SPEECH,
            new_filename
        )

        if not os.path.exists(destination):
            shutil.copy2(
                wav_path,
                destination
            )

        speech_count += 1


print(
    f"\nCopied {speech_count} speech-command files."
)


# ------------------------------------------------------------
# 2. FIND BACKGROUND NOISE
# ------------------------------------------------------------

background_folder = os.path.join(
    SOURCE,
    "_background_noise_"
)

background_count = 0

if os.path.isdir(background_folder):

    print("\nProcessing background noise...")

    for wav_path in glob.glob(
        os.path.join(background_folder, "*.wav")
    ):

        filename = os.path.basename(wav_path)
        stem = os.path.splitext(filename)[0]

        audio, sample_rate = sf.read(wav_path)

        # Stereo → mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        clip_length = int(sample_rate * 1.0)
        stride = int(sample_rate * 0.5)

        clip_number = 0

        for start in range(
            0,
            len(audio) - clip_length + 1,
            stride
        ):

            clip = audio[
                start:start + clip_length
            ]

            output_name = (
                f"background_"
                f"{stem}_"
                f"{clip_number:04d}.wav"
            )

            output_path = os.path.join(
                NEGATIVE_BACKGROUND,
                output_name
            )

            sf.write(
                output_path,
                clip,
                sample_rate
            )

            clip_number += 1
            background_count += 1

else:

    print(
        "\nNo _background_noise_ folder found."
    )


print(
    f"Created {background_count} background clips."
)


# ------------------------------------------------------------
# 3. SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 50)
print("DATASET SUMMARY")
print("=" * 50)

for folder in [
    "negative_speech_commands",
    "negative_background",
]:

    path = os.path.join(RAW, folder)

    count = len(
        glob.glob(
            os.path.join(path, "*.wav")
        )
    )

    print(f"{folder}: {count}")

print("=" * 50)
print("\nDONE!")