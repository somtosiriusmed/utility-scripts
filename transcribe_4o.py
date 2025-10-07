import os
import requests
from .config import API_KEY, API_VERSION_GPT5, API_VERSION_TRANSCRIBE, AZURE_RESOURCE, TRANSCRIBE_DEPLOYMENT

# -----------------------------
# CONFIG
# -----------------------------

AUDIO_FOLDER = "output_audio"
OUTPUT_FILE = "pidgin_transcriptions.txt"
# -----------------------------

# Endpoints
TRANSCRIBE_URL = f"https://{AZURE_RESOURCE}.cognitiveservices.azure.com/openai/deployments/{TRANSCRIBE_DEPLOYMENT}/audio/transcriptions?api-version={API_VERSION_TRANSCRIBE}"
REFINE_URL = f"https://{AZURE_RESOURCE}.cognitiveservices.azure.com/openai/deployments/{REFINE_DEPLOYMENT}/chat/completions?api-version={API_VERSION_GPT5}"

HEADERS = {"api-key": API_KEY}

# -----------------------------
# Functions
# -----------------------------
def transcribe_audio(filepath, filename):
    with open(filepath, "rb") as f:
        response = requests.post(
            TRANSCRIBE_URL,
            headers=HEADERS,
            files={"file": (filename, f, "audio/wav")},
            data={"language": "en"}  # pidgin ~ english
        )
    if response.status_code == 200:
        return response.json().get("text", "")
    else:
        print(f"❌ Transcription failed for {filename}: {response.text}")
        return None

def refine_to_pidgin(raw_text, filename):
    system_prompt = """You are a Nigerian Pidgin transcription corrector.
Task:
- Convert the transcript into proper Nigerian Pidgin, exactly as spoken.
- Do NOT translate into Standard English.
- Preserve original words, slang, and style.
- Only fix casing, spacing, punctuation, and obvious mis-hearings.
- Output ONLY the corrected Pidgin transcript, nothing else."""

    body = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Transcript from {filename}:\n\n{raw_text}"}
        ],
        "temperature": 0.0,
        "max_tokens": 500
    }

    response = requests.post(
        REFINE_URL,
        headers={**HEADERS, "Content-Type": "application/json"},
        json=body
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print(f"⚠️ Refinement failed for {filename}: {response.text}")
        return raw_text

# -----------------------------
# Main pipeline
# -----------------------------
if __name__ == "__main__":
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for file in os.listdir(AUDIO_FOLDER):
            if file.lower().endswith((".wav", ".mp3")):
                filepath = os.path.join(AUDIO_FOLDER, file)

                raw = transcribe_audio(filepath, file)
                if raw:
                    refined = refine_to_pidgin(raw, file)

                    print(f"\n🎙️ {file}")
                    print(f"RAW: {raw}")
                    print(f"PIDGIN: {refined}")

                    out.write(f"{file}:\n{refined}\n\n")
                    out.flush()

    print("\n🏁 Finished transcription + Pidgin correction.")
