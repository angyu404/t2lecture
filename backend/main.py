from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pathlib import Path
import subprocess
import uuid
import os
import json

from google import genai

print("RUNNING FILE:", __file__)

app = FastAPI()

# Dev: allow all origins (lock down for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_DIR = BASE_DIR / "audio"
UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# Whisper tiny: lowest resource
model = WhisperModel("tiny", device="cpu", compute_type="int8")

# Gemini client
GEMINI_API_KEY = "AIzaSyCAF7unhPwVy_SpDHnH5PFD9M1zz30dHAQ"
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"


def extract_audio_to_wav(video_path: Path, wav_path: Path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        str(wav_path),
    ]
    # If ffmpeg isn't on PATH, Windows throws FileNotFoundError (WinError 2)
    subprocess.run(cmd, check=True)


def polish_transcript_with_gemini(raw_text: str, language_hint: str | None = None) -> dict:
    """
    Returns dict:
      {
        "polished": "...",
        "changes_summary": [...],
        "warnings": [...]
      }
    """
    if not gemini_client:
        raise RuntimeError("GEMINI_API_KEY not set (Gemini client not initialized).")

    rules = (
        "You are a transcript editor for lecture transcripts.\n"
        "STRICT RULES:\n"
        "- Do NOT add new facts, claims, or examples.\n"
        "- Do NOT remove important technical details.\n"
        "- Remove filler words, false starts, stutters, and repeated phrases.\n"
        "- Fix punctuation, casing, and sentence boundaries.\n"
        "- Keep acronyms, variable names, numbers, and proper nouns EXACTLY.\n"
        "- If something is unclear in the raw transcript, keep it unclear; do NOT invent.\n"
    )
    if language_hint:
        rules += f"\nLanguage hint: {language_hint}\n"

    prompt = (
        f"{rules}\n"
        "Return ONLY valid JSON with keys:\n"
        "  polished: string\n"
        "  changes_summary: array of short strings\n"
        "  warnings: array of short strings\n\n"
        f"RAW TRANSCRIPT:\n{raw_text}\n"
    )

    resp = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json"},
    )

    data = json.loads(resp.text)
    if "polished" not in data:
        raise RuntimeError("Gemini response missing 'polished' field.")
    return data


@app.get("/")
def root():
    return {"message": "t2lecture backend running. Visit /docs or /health."}


@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug-key")
def debug_key():
    k = os.getenv("GEMINI_API_KEY") or ""
    return {"has_key": bool(k), "key_prefix": k[:6], "key_len": len(k)}

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    polish: bool = Query(False),
):
    suffix = Path(file.filename).suffix or ".mp4"
    file_id = str(uuid.uuid4())

    video_path = UPLOAD_DIR / f"{file_id}{suffix}"
    with open(video_path, "wb") as f:
        f.write(await file.read())

    wav_path = AUDIO_DIR / f"{file_id}.wav"

    try:
        extract_audio_to_wav(video_path, wav_path)
    except FileNotFoundError:
        return {
            "error": "ffmpeg not found. Install ffmpeg and ensure it is on PATH, then restart the backend."
        }
    except subprocess.CalledProcessError:
        return {"error": "ffmpeg failed while extracting audio."}

    segments, info = model.transcribe(str(wav_path), vad_filter=True)

    seg_list = []
    lines = []
    for s in segments:
        t = (s.text or "").strip()
        if t:
            seg_list.append({
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": t
            })
            lines.append(t)

    raw_text = "\n".join(lines)

    final_text = raw_text
    polish_meta = None

    if polish and raw_text.strip():
        try:
            polish_meta = polish_transcript_with_gemini(
                raw_text,
                language_hint=getattr(info, "language", None)
            )
            final_text = polish_meta.get("polished", raw_text)
        except Exception as e:
            polish_meta = {"error": str(e)}
            final_text = raw_text

    return {
        "language": getattr(info, "language", None),
        "text": final_text,
        "raw_text": raw_text,
        "segments": seg_list,
        "polish": polish_meta,
    }