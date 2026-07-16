"""
MuseTalk REST API + UI. One API; the UI and the LiveKit worker both build on `engine`.

Endpoints:
  GET  /                       -> the UI page
  GET  /gpu                    -> live GPU snapshot (util/mem/power via NVML)
  GET  /personas               -> list personas
  POST /personas               -> create a persona from an initial video (multipart: name, video)
  POST /personas/{id}/speak    -> send a wav -> returns the lip-synced mp4 (multipart: audio)
                                  perf headers: X-Render-Seconds, X-Audio-Seconds,
                                  X-Realtime-Factor, X-Fps; full detail in the
                                  <out>.mp4.profile.json sidecar + docker logs.

Run inside the OpenAvatarChat env:  uvicorn api:app --host 0.0.0.0 --port 8000
Env: MUSETALK_LOG_LEVEL=DEBUG for verbose logs, MUSETALK_DEBUG=1 for per-batch GPU timings.
"""
import os
import time
import uuid
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

import engine
from profiling import logger, gpu_snapshot

HERE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/tmp/musetalk_uploads")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/tmp/musetalk_outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="MuseTalk avatar service")


def _slug(name: str) -> str:
    s = "".join(c for c in (name or "") if c.isalnum() or c in "-_")
    return s or uuid.uuid4().hex[:8]


def _save_upload(upload: UploadFile, path: str) -> int:
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return os.path.getsize(path)


@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(HERE, "static", "index.html"), encoding="utf-8").read()


@app.get("/gpu")
def gpu():
    """Live GPU reading — poll this (or nvidia-smi on the VM) during a render."""
    snap = gpu_snapshot()
    if not snap:
        return JSONResponse({"available": False, "hint": "NVML unavailable — is nvidia-ml-py "
                             "installed and the container running with GPU access?"})
    return JSONResponse({"available": True, **snap})


@app.get("/personas")
def personas():
    return engine.list_personas()


# sync def -> FastAPI runs it in a threadpool, keeping the GPU work off the event loop
@app.post("/personas")
def create_persona(name: str = Form(...), video: UploadFile = File(...)):
    pid = _slug(name)
    vpath = os.path.join(UPLOAD_DIR, f"{pid}_{video.filename}")
    size = _save_upload(video, vpath)
    logger.info(f"POST /personas name={pid} video={video.filename} ({size / 1e6:.1f} MB)")
    t0 = time.perf_counter()
    try:
        res = engine.prepare_persona(pid, vpath)
    except Exception as e:
        logger.exception(f"persona preparation failed: {pid}")
        raise HTTPException(500, f"preparation failed: {e}")
    logger.info(f"POST /personas {pid} done in {time.perf_counter() - t0:.1f}s "
                f"(cycle_frames={res.get('cycle_frames')})")
    return JSONResponse({"ok": True, **res})


@app.post("/personas/{persona_id}/speak")
def speak(persona_id: str, audio: UploadFile = File(...)):
    if persona_id not in {p["id"] for p in engine.list_personas()}:
        raise HTTPException(404, "persona not found — set it up first")
    apath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{audio.filename}")
    size = _save_upload(audio, apath)
    logger.info(f"POST /personas/{persona_id}/speak audio={audio.filename} ({size / 1e3:.0f} KB)")
    outp = os.path.join(OUTPUT_DIR, f"{persona_id}_{uuid.uuid4().hex[:8]}.mp4")
    try:
        _, profile = engine.render_wav(persona_id, apath, outp)
    except Exception as e:
        logger.exception(f"synthesis failed: {persona_id}")
        raise HTTPException(500, f"synthesis failed: {e}")
    headers = {
        "X-Render-Seconds": str(profile.get("total_sec", "")),
        "X-Audio-Seconds": str(profile.get("audio_sec", "")),
        "X-Realtime-Factor": str(profile.get("realtime_factor", "")),
        "X-Fps": str(profile.get("fps", "")),
    }
    return FileResponse(outp, media_type="video/mp4", filename=os.path.basename(outp),
                        headers=headers)
