"""
MuseTalk engine — shared core for BOTH the REST API (UI / wav-test) and the LiveKit worker.

Personas (an "initial video" prepared into cached masks+latents) live on disk under
  $PERSONA_DIR/v15/avatars/<persona_id>/   (latents.pt, masks.pkl, frames.pkl, ...)
so a persona set up via the UI is immediately usable by the LiveKit worker — same store, same engine.

Runs inside the OpenAvatarChat env (it has the MuseTalk deps + models via
`download_models.py --handler musetalk`). Set OAC_ROOT to the repo root.
"""
from __future__ import annotations

import os
import sys
import json
import uuid
import shutil
import subprocess
from typing import Optional

import numpy as np
import cv2
import librosa

# --- make OpenAvatarChat's MuseTalk engine importable + fix CWD-relative model paths ---
OAC_ROOT = os.environ.get("OAC_ROOT", "/root/open-avatar-chat")
sys.path.insert(0, os.path.join(OAC_ROOT, "src"))
sys.path.insert(0, os.path.join(OAC_ROOT, "src", "handlers", "avatar", "musetalk", "MuseTalk"))
os.chdir(OAC_ROOT)

# MuseTalk pulls in diffusers -> xformers, which HARD-FAILS at import if the installed flash-attn
# version is outside its pinned window (e.g. 2.8.3 vs the expected <=2.8.2). OpenAvatarChat's own
# handlers set this too — bypass the guard. Must be set BEFORE importing musetalk.
os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")

from handlers.avatar.musetalk.musetalk_algo import MuseTalkAlgoV15  # noqa: E402

# --- paths (match OpenAvatarChat's musetalk handler) ---
MODEL_DIR = os.path.join(OAC_ROOT, "models", "musetalk")
PERSONA_DIR = os.environ.get("PERSONA_DIR", os.path.join(MODEL_DIR, "avatar_model"))
UNET_PTH = os.path.join(MODEL_DIR, "musetalkV15", "unet.pth")
UNET_JSON = os.path.join(MODEL_DIR, "musetalkV15", "musetalk.json")
WHISPER_DIR = os.path.join(MODEL_DIR, "whisper")

FPS = int(os.environ.get("MUSETALK_FPS", "25"))
BATCH = int(os.environ.get("MUSETALK_BATCH", "4"))
ALGO_SR = 16000  # whisper feature sample rate (fixed by MuseTalk)


def _personas_root() -> str:
    return os.path.join(PERSONA_DIR, "v15", "avatars")


def list_personas() -> list[dict]:
    root = _personas_root()
    if not os.path.isdir(root):
        return []
    out = []
    for d in sorted(os.listdir(root)):
        if not os.path.isfile(os.path.join(root, d, "latents.pt")):
            continue
        info = {}
        ip = os.path.join(root, d, "avator_info.json")
        if os.path.isfile(ip):
            try:
                info = json.load(open(ip))
            except Exception:
                pass
        out.append({"id": d, "info": info})
    return out


def _make_algo(persona_id: str, video_path: str, force: bool) -> MuseTalkAlgoV15:
    return MuseTalkAlgoV15(
        avatar_id=persona_id,
        video_path=video_path or "",
        bbox_shift=0,
        batch_size=BATCH,
        force_preparation=force,
        fps=FPS,
        version="v15",
        result_dir=PERSONA_DIR,
        vae_type="sd-vae",
        unet_model_path=UNET_PTH,
        unet_config=UNET_JSON,
        whisper_dir=WHISPER_DIR,
        gpu_id=0,
    )


def prepare_persona(persona_id: str, video_path: str) -> dict:
    """Setup a persona from an initial video (one-time preparation: detect+mask+VAE-encode, cached)."""
    adir = os.path.join(_personas_root(), persona_id)
    if os.path.isdir(adir):
        shutil.rmtree(adir, ignore_errors=True)  # overwrite cleanly
    algo = _make_algo(persona_id, video_path, force=True)
    algo.init()  # runs prepare_material() and caches masks/latents/frames
    n = len(algo.frame_list_cycle) if algo.frame_list_cycle is not None else 0
    return {"id": persona_id, "cycle_frames": n}


# The MuseTalk model set is heavy (~GB). Keep ONE loaded persona in-process (LRU=1).
_loaded: dict = {}


def load_persona(persona_id: str) -> MuseTalkAlgoV15:
    if _loaded.get("id") != persona_id:
        algo = _make_algo(persona_id, "", force=False)
        algo.init()  # loads models + cached persona data
        _loaded.clear()
        _loaded.update(id=persona_id, algo=algo)
    return _loaded["algo"]


def render_wav(persona_id: str, wav_path: str, out_path: str) -> str:
    """Batch: wav -> lip-synced mp4 (audio muxed). Used by the UI test endpoint."""
    algo = load_persona(persona_id)
    audio, _ = librosa.load(wav_path, sr=ALGO_SR, mono=True)
    whisper_chunks = algo.extract_whisper_feature(audio, ALGO_SR)
    n = len(whisper_chunks)
    if n == 0:
        raise ValueError("no audio frames extracted")

    frames_dir = out_path + ".frames"
    os.makedirs(frames_dir, exist_ok=True)
    try:
        for i in range(n):
            bgr = algo.generate_frame(whisper_chunks[i:i + 1], i)  # idx=i walks the loop
            cv2.imwrite(os.path.join(frames_dir, f"{i:08d}.png"), bgr)
        silent = out_path + ".silent.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "warning", "-r", str(FPS), "-f", "image2",
             "-i", os.path.join(frames_dir, "%08d.png"),
             "-vcodec", "libx264", "-vf", "format=yuv420p", "-crf", "18", silent],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-v", "warning", "-i", silent, "-i", wav_path,
             "-c:v", "copy", "-c:a", "aac", "-shortest", out_path],
            check=True,
        )
        os.remove(silent)
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
    return out_path
