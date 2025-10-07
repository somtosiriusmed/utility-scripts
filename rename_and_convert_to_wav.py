import os
from pydub import AudioSegment

FOLDER = "raw_files"

def rename_and_convert_to_wav(folder):
    files = sorted(os.listdir(folder))
    counter = 1

    for file in files:
        old_path = os.path.join(folder, file)

        # skip directories
        if not os.path.isfile(old_path):
            continue

        # extract extension (e.g. .mp3, .wav)
        ext = os.path.splitext(file)[1].lower()

        # new filename with 8-digit numbering
        new_name = f"{counter:08d}.wav"
        new_path = os.path.join(folder, new_name)

        # load audio and export as wav
        try:
            audio = AudioSegment.from_file(old_path, format=ext.replace('.', ''))
            audio.export(new_path, format="wav")
            print(f"✅ Converted & renamed: {file} → {new_name}")
        except Exception as e:
            print(f"❌ Failed to convert {file}: {e}")

        counter += 1

if __name__ == "__main__":
    rename_and_convert_to_wav(FOLDER)
    print("🎉 All files converted to WAV and renamed!")
