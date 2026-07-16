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
import json
import time
import uuid
import shutil
import struct
import asyncio

import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

import engine
import live as live_mode
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


@app.get("/personas/{persona_id}/idle.mp4")
def idle_video(persona_id: str):
    """The persona's idle cycle as a loopable mp4 (rendered once, cached).
    The live UI plays this natively; the WS only streams frames during speech."""
    if persona_id not in {p["id"] for p in engine.list_personas()}:
        raise HTTPException(404, "persona not found — set it up first")
    try:
        path = engine.idle_loop_mp4(persona_id)
    except Exception as e:
        logger.exception(f"idle loop render failed: {persona_id}")
        raise HTTPException(500, f"idle loop render failed: {e}")
    return FileResponse(path, media_type="video/mp4")


@app.websocket("/personas/{persona_id}/live")
async def live_ws(ws: WebSocket, persona_id: str):
    """Live mic mode: client streams 16kHz mono Int16 PCM (binary messages);
    server streams back JPEG frames (binary) + per-second JSON stats (text)."""
    await ws.accept()
    if persona_id not in {p["id"] for p in engine.list_personas()}:
        await ws.close(code=4004, reason="persona not found")
        return
    algo = await asyncio.to_thread(engine.load_persona, persona_id)
    sess = live_mode.LiveSession(algo)
    logger.info(f"live[{persona_id}] session start — fps={sess.fps}, "
                f"window={live_mode.WINDOW_SEC}s ({sess.window_samples} samples), batch={engine.BATCH}")

    async def receiver():
        try:
            while not sess.stopped:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    sess.add_audio_pcm16(msg["bytes"])
                elif msg.get("text"):
                    txt = msg["text"]
                    if txt == "stop":
                        break
                    try:
                        cfg = json.loads(txt)
                        if cfg.get("type") == "config" and "gate" in cfg:
                            sess.gate = max(0.0, float(cfg["gate"]))
                            logger.info(f"live[{persona_id}] speech gate -> {sess.gate}")
                        elif cfg.get("type") == "pos" and "idx" in cfg:
                            sess.report_client_idx(int(cfg["idx"]), time.perf_counter())
                    except (ValueError, TypeError):
                        pass
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            sess.stopped = True

    rx = asyncio.create_task(receiver())
    gen_task = None
    interval = 1.0 / sess.fps
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, live_mode.JPEG_QUALITY]
    sent = sent_prev = 0
    t0 = last_stats = time.perf_counter()
    next_t = t0
    try:
        while not sess.stopped:
            # kick one background generation whenever a full audio window is buffered
            if sess.has_window() and (gen_task is None or gen_task.done()):
                gen_task = asyncio.create_task(
                    asyncio.to_thread(sess.generate_window, time.perf_counter()))
            # Speech frames only, FIFO, on the fps clock. During silence NOTHING is
            # streamed — the browser falls back to the natively-looping idle.mp4.
            if sess.pending:
                frame, idx = sess.pending.popleft()
                ok, jpg = cv2.imencode(".jpg", frame, jpeg_params)
                if ok:
                    # 4-byte LE cycle index prefix — lets the client seek its idle
                    # loop back to where speech ended (continuity on fallback)
                    await ws.send_bytes(struct.pack("<I", idx % (1 << 32)) + jpg.tobytes())
                    sent += 1
            now = time.perf_counter()
            if now - last_stats >= 1.0:
                await ws.send_text(json.dumps({
                    "type": "stats",
                    "sent_fps": round((sent - sent_prev) / (now - last_stats), 1),
                    "target_fps": sess.fps,
                    "speaking": bool(sess.pending),
                    "audio_backlog_sec": round(sess.audio_backlog_sec(), 2),
                    "pending_frames": len(sess.pending),
                    "dropped_audio_sec": round(sess.dropped_sec, 1),
                    "level": sess.level,
                    "gate": round(sess.gate, 3),
                    "vad": sess.vad.available,
                }))
                last_stats = now
                sent_prev = sent
            next_t += interval
            delay = next_t - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_t = time.perf_counter()  # fell behind — reset the clock, don't spiral
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        sess.stopped = True
        rx.cancel()
        if gen_task is not None:
            try:
                await gen_task
            except Exception:
                pass
        logger.info(f"live[{persona_id}] session end — {sent} speech frames streamed "
                    f"in {time.perf_counter() - t0:.1f}s (idle played client-side)")
