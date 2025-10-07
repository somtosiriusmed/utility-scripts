import os
import yt_dlp

# Path to your file with YouTube links
LINKS_FILE = "urls.txt"
# Output folder for MP3 files
OUTPUT_DIR = "downloads"

def download_audio(link, output_dir):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "postprocessors": [
            {  
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",  # you can change bitrate here
            }
        ],
        "quiet": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([link])

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(LINKS_FILE, "r", encoding="utf-8") as f:
        links = [line.strip() for line in f if line.strip()]

    for link in links:
        print(f"Downloading audio from: {link}")
        try:
            download_audio(link, OUTPUT_DIR)
        except Exception as e:
            print(f"❌ Failed to download {link}: {e}")

if __name__ == "__main__":
    main()

