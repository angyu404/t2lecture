from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from pathlib import Path
import subprocess
import uuid

app = FastAPI()

# 開發階段先允許所有來源，部署後再改成只允許你的前端網址
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_DIR = BASE_DIR / "audio"
UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

# tiny 最省資源
model = WhisperModel("tiny", device="cpu", compute_type="int8")


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
    subprocess.run(cmd, check=True)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".mp4"
    file_id = str(uuid.uuid4())

    video_path = UPLOAD_DIR / f"{file_id}{suffix}"
    with open(video_path, "wb") as f:
        f.write(await file.read())

    wav_path = AUDIO_DIR / f"{file_id}.wav"

    try:
        extract_audio_to_wav(video_path, wav_path)
    except subprocess.CalledProcessError:
        return {"error": "ffmpeg failed. Please confirm ffmpeg is installed."}

    # ✅ 升級 3：回傳 segments（start/end/text）
    segments, info = model.transcribe(str(wav_path), vad_filter=True)

    seg_list = []
    lines = []
    for s in segments:
        t = s.text.strip()
        if t:
            seg_list.append({
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": t
            })
            lines.append(t)

    return {
        "language": info.language,
        "text": "\n".join(lines),
        "segments": seg_list
    }