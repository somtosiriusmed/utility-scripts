import os

FOLDER = "2000_raw_wav"

def rename_files_sequentially(folder):
    files = sorted(os.listdir(folder))
    counter = 1

    for file in files:
        old_path = os.path.join(folder, file)

        # skip directories, only rename files
        if not os.path.isfile(old_path):
            continue

        # keep extension (.wav, .mp3, etc.)
        ext = os.path.splitext(file)[1]
        new_name = f"{counter:08d}{ext}"  # 8 digits with leading zeros
        new_path = os.path.join(folder, new_name)

        os.rename(old_path, new_path)
        print(f"Renamed: {file} → {new_name}")
        counter += 1

if __name__ == "__main__":
    rename_files_sequentially(FOLDER)
    print("✅ Renaming completed!")

