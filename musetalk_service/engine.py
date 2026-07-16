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
import time
import uuid
import shutil
import threading
import subprocess
from typing import Optional

import numpy as np
import cv2
import librosa

from profiling import logger, StageTimer, GpuSampler, write_profile

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
DEBUG = os.environ.get("MUSETALK_DEBUG", "0") == "1"  # per-batch UNet/VAE [PROFILE] logs in the algo
# Persona prep: cap frames taken from the source video (75 ≈ 3s idle loop @25fps). 0 = use all.
MAX_FRAMES = int(os.environ.get("MUSETALK_MAX_FRAMES", "75"))
# Output encoder: auto (nvenc if the container ffmpeg+driver support it, else libx264) | nvenc | libx264
ENCODER = os.environ.get("MUSETALK_ENCODER", "auto").lower()


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
        debug=DEBUG,
    )
    # NOTE: MuseTalkAlgoV15.__init__ calls self.init() itself — models are loaded and
    # (if force/missing) prepare_material() has already run when the ctor returns.
    # Do NOT call algo.init() again: it used to run the whole preparation TWICE.


def _count_video_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def prepare_persona(persona_id: str, video_path: str) -> dict:
    """Setup a persona from an initial video (one-time preparation: detect+mask+VAE-encode, cached)."""
    t = StageTimer("prepare", ref=persona_id, cuda_sync=True)
    sampler = GpuSampler().start()

    src_frames = _count_video_frames(video_path) if os.path.isfile(video_path) else 0
    used_path = video_path
    if MAX_FRAMES > 0 and src_frames > MAX_FRAMES:
        # Cap the idle loop: prep cost scales linearly with frames (2x for masks after
        # the forward+reverse doubling), and a short blink-loop is usually all we need.
        trimmed = f"{video_path}.trim{MAX_FRAMES}.mp4"
        logger.info(f"{t.tag} trimming source video {src_frames} -> {MAX_FRAMES} frames "
                    f"(MUSETALK_MAX_FRAMES={MAX_FRAMES}, set 0 to keep all)")
        with t.stage("trim_video"):
            subprocess.run(
                ["ffmpeg", "-y", "-v", "warning", "-i", video_path,
                 "-frames:v", str(MAX_FRAMES), "-c:v", "libx264", "-crf", "18", "-an", trimmed],
                check=True,
            )
        used_path = trimmed
    else:
        logger.info(f"{t.tag} using all {src_frames or '?'} source frames")

    adir = os.path.join(_personas_root(), persona_id)
    if os.path.isdir(adir):
        shutil.rmtree(adir, ignore_errors=True)  # overwrite cleanly

    with _load_lock:  # model construction is not thread-safe (see load_persona)
        _loaded.clear()  # evict any in-memory persona (stale same-id data / free GPU room for prep)
        with t.stage("models_and_prepare"):  # ctor = model load + prepare_material (see [PREP] logs)
            algo = _make_algo(persona_id, used_path, force=True)
        n = len(algo.frame_list_cycle) if algo.frame_list_cycle is not None else 0
        algo.force_preparation = False  # prepared; keep it warm so the first /speak skips the reload
        _loaded.update(id=persona_id, algo=algo)

    gpu = sampler.stop()
    profile = t.finish(gpu=gpu, source_frames=src_frames,
                       used_frames=min(src_frames, MAX_FRAMES) if MAX_FRAMES > 0 else src_frames,
                       cycle_frames=n)
    write_profile(os.path.join(adir, "prepare.profile.json"), profile)
    return {"id": persona_id, "cycle_frames": n, "profile": profile}


# The MuseTalk model set is heavy (~GB). Keep ONE loaded persona in-process (LRU=1).
_loaded: dict = {}
# Model loading is NOT thread-safe (diffusers' fast-init leaves "meta" tensors when two
# threads construct concurrently) — e.g. the live WS and GET /idle.mp4 fire together.
# One lock: the second caller waits and reuses the first caller's load.
_load_lock = threading.Lock()


def load_persona(persona_id: str) -> MuseTalkAlgoV15:
    with _load_lock:
        if _loaded.get("id") != persona_id:
            prev = _loaded.get("id")
            logger.info(f"load_persona[{persona_id}] cold load (was: {prev or 'none'}; LRU=1 — "
                        f"persona switches reload the full model set)")
            t0 = time.perf_counter()
            algo = _make_algo(persona_id, "", force=False)  # ctor loads models + cached persona data
            _loaded.clear()
            _loaded.update(id=persona_id, algo=algo)
            logger.info(f"load_persona[{persona_id}] ready in {time.perf_counter() - t0:.2f}s")
        return _loaded["algo"]


# --------------------------------------------------------------- video output ---

_encoder_cache: dict = {}


def _pick_encoder() -> tuple[str, list[str]]:
    """Choose the h264 encoder once per process. Returns (name, ffmpeg args)."""
    if _encoder_cache:
        return _encoder_cache["name"], _encoder_cache["args"]
    NVENC = ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"])
    X264 = ("libx264", ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
    name, args = X264
    if ENCODER in ("auto", "nvenc"):
        ok = False
        try:
            encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                      capture_output=True, text=True, timeout=15).stdout
            if "h264_nvenc" in encoders:
                # listed != usable (needs the nvidia 'video' driver capability) — probe one frame
                probe = subprocess.run(
                    ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=black:size=256x256",
                     "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"],
                    capture_output=True, text=True, timeout=30)
                ok = probe.returncode == 0
                if not ok:
                    logger.warning(f"h264_nvenc listed but probe failed "
                                   f"({(probe.stderr or '').strip().splitlines()[-1] if probe.stderr else 'no error output'}) "
                                   f"— container may need NVIDIA_DRIVER_CAPABILITIES=compute,utility,video")
        except Exception as e:
            logger.warning(f"nvenc probe failed: {e}")
        if ok:
            name, args = NVENC
        elif ENCODER == "nvenc":
            logger.warning("MUSETALK_ENCODER=nvenc forced but probe failed — trying it anyway")
            name, args = NVENC
    logger.info(f"video encoder: {name} (MUSETALK_ENCODER={ENCODER})")
    _encoder_cache.update(name=name, args=args)
    return name, args


def idle_loop_mp4(persona_id: str) -> str:
    """Render the persona's idle cycle (fwd+rev source frames) to a seamless-loop mp4,
    cached in the persona dir. The live UI plays this natively in the browser and only
    receives streamed frames while the avatar is actually speaking."""
    adir = os.path.join(_personas_root(), persona_id)
    out = os.path.join(adir, f"idle_loop_{FPS}fps.mp4")
    if os.path.isfile(out):
        return out
    algo = load_persona(persona_id)
    frames = algo.frame_list_cycle
    h, w = frames[0].shape[:2]
    enc_name, enc_args = _pick_encoder()
    logger.info(f"idle_loop_mp4[{persona_id}] rendering {len(frames)} frames @{FPS}fps ({enc_name})")
    t0 = time.perf_counter()
    cmd = ["ffmpeg", "-y", "-v", "warning",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "pipe:0",
           *enc_args,
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
           "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    try:
        for f in frames:
            proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed rendering idle loop")
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    logger.info(f"idle_loop_mp4[{persona_id}] done in {time.perf_counter() - t0:.1f}s -> {out}")
    return out


def render_wav(persona_id: str, wav_path: str, out_path: str) -> tuple[str, dict]:
    """Batch: wav -> lip-synced mp4 (audio muxed). Returns (out_path, profile dict).

    Pipeline: whisper features once for the whole clip, then BATCHED UNet+VAE
    (MUSETALK_BATCH frames per GPU call), each blended frame piped raw (bgr24)
    into a SINGLE ffmpeg process that encodes video + muxes audio in one pass —
    no intermediate PNGs, no second ffmpeg run.
    """
    t = StageTimer("speak", ref=persona_id, cuda_sync=True)
    sampler = GpuSampler().start()

    with t.stage("load_persona"):
        algo = load_persona(persona_id)
    with t.stage("audio_load"):
        audio, _ = librosa.load(wav_path, sr=ALGO_SR, mono=True)
    audio_sec = len(audio) / ALGO_SR
    with t.stage("whisper_extract"):
        whisper_chunks = algo.extract_whisper_feature(audio, ALGO_SR)
    n = len(whisper_chunks)
    if n == 0:
        raise ValueError("no audio frames extracted")

    h, w = algo.frame_list_cycle[0].shape[:2]
    enc_name, enc_args = _pick_encoder()
    logger.info(f"{t.tag} audio={audio_sec:.2f}s frames={n} fps_target={FPS} "
                f"size={w}x{h} batch={BATCH} encoder={enc_name}")

    ffmpeg_log = out_path + ".ffmpeg.log"
    cmd = ["ffmpeg", "-y", "-v", "warning",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "pipe:0",
           "-i", wav_path,
           *enc_args,
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
           "-c:a", "aac", "-shortest", out_path]
    log_f = open(ffmpeg_log, "w")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log_f, stderr=log_f)
    done = 0
    t_start = time.perf_counter()
    try:
        for i in range(0, n, BATCH):
            chunk = whisper_chunks[i:i + BATCH]
            b = len(chunk)
            with t.stage("frame_gen", items=b):
                results = algo.generate_frames(chunk, i, b)  # one UNet+VAE call for b frames
            for recon, idx in results:
                with t.stage("blend", items=1):
                    bgr = algo.res2combined(recon, idx)
                with t.stage("pipe_write", items=1):
                    proc.stdin.write(np.ascontiguousarray(bgr, dtype=np.uint8).tobytes())
                done += 1
            if done % 25 < BATCH and done < n:
                rolling = done / (time.perf_counter() - t_start)
                logger.debug(f"{t.tag} frame {done}/{n} ({rolling:.1f} fps rolling)")
        with t.stage("ffmpeg_wait"):
            proc.stdin.close()
            ret = proc.wait()
    except BrokenPipeError:
        proc.wait()
        ret = proc.returncode or 1
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        log_f.close()
    if ret != 0:
        tail = open(ffmpeg_log).read()[-2000:]
        raise RuntimeError(f"ffmpeg failed (exit {ret}): {tail}")
    os.remove(ffmpeg_log)

    gpu = sampler.stop()
    profile = t.finish(audio_sec=audio_sec, frames=n, gpu=gpu,
                       batch=BATCH, encoder=enc_name, size=f"{w}x{h}", debug=DEBUG)
    write_profile(out_path + ".profile.json", profile)
    return out_path, profile
