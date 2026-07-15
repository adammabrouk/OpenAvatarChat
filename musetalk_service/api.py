"""
MuseTalk REST API + UI. One API; the UI and the LiveKit worker both build on `engine`.

Endpoints:
  GET  /                       -> the UI page
  GET  /personas               -> list personas
  POST /personas               -> create a persona from an initial video (multipart: name, video)
  POST /personas/{id}/speak    -> send a wav -> returns the lip-synced mp4 (multipart: audio)

Run inside the OpenAvatarChat env:  uvicorn api:app --host 0.0.0.0 --port 8000
"""
import os
import uuid
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

import engine

HERE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/musetalk_uploads")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/musetalk_outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="MuseTalk avatar service")


def _slug(name: str) -> str:
    s = "".join(c for c in (name or "") if c.isalnum() or c in "-_")
    return s or uuid.uuid4().hex[:8]


@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(HERE, "static", "index.html"), encoding="utf-8").read()


@app.get("/personas")
def personas():
    return engine.list_personas()


# sync def -> FastAPI runs it in a threadpool, keeping the GPU work off the event loop
@app.post("/personas")
def create_persona(name: str = Form(...), video: UploadFile = File(...)):
    pid = _slug(name)
    vpath = os.path.join(UPLOAD_DIR, f"{pid}_{video.filename}")
    with open(vpath, "wb") as f:
        shutil.copyfileobj(video.file, f)
    try:
        res = engine.prepare_persona(pid, vpath)
    except Exception as e:
        raise HTTPException(500, f"preparation failed: {e}")
    return JSONResponse({"ok": True, **res})


@app.post("/personas/{persona_id}/speak")
def speak(persona_id: str, audio: UploadFile = File(...)):
    if persona_id not in {p["id"] for p in engine.list_personas()}:
        raise HTTPException(404, "persona not found — set it up first")
    apath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{audio.filename}")
    with open(apath, "wb") as f:
        shutil.copyfileobj(audio.file, f)
    outp = os.path.join(OUTPUT_DIR, f"{persona_id}_{uuid.uuid4().hex[:8]}.mp4")
    try:
        engine.render_wav(persona_id, apath, outp)
    except Exception as e:
        raise HTTPException(500, f"synthesis failed: {e}")
    return FileResponse(outp, media_type="video/mp4", filename=os.path.basename(outp))
