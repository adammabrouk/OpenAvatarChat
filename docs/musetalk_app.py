"""
Standalone MuseTalk (v1.5) Gradio service — Setup + Inference tabs + HTTP API.

WHAT IT DOES
  • Setup tab   : upload a short (looping) video of a face → one-time "preparation"
                  (face detection + VAE-encode every frame + build the mouth masks,
                  ping-pong looped) → cached under ./results/v15/avatars/<id>/.
  • Inference   : pick a prepared avatar + upload an AUDIO clip → MuseTalk swaps the
                  mouth region frame-by-frame to match the audio → returns an mp4.
                  The video loops (forward+reverse) for as long as the audio lasts.

HOW IT WORKS INTERNALLY
  This wraps the official MuseTalk `scripts/realtime_inference.py::Avatar` class. That
  class reads several MODULE-LEVEL globals (args, device, vae, unet, pe, timesteps,
  audio_processor, whisper, weight_dtype, fp) that upstream only sets inside `__main__`.
  We load the models here and inject those globals into the module, then drive Avatar.

REQUIREMENTS
  • Run from the MuseTalk repo root (so ./models/... and ./musetalk/utils/dwpose/... resolve).
  • Weights downloaded via MuseTalk's download_weights.sh (see the guideline doc).
  • Env: MUSETALK_ROOT (default: this file's dir), GPU_ID (0), MUSETALK_BATCH (4).
"""

import os
import sys
import time
import uuid
import shutil
from types import SimpleNamespace

# --- Run from the MuseTalk repo root: the vendored code uses CWD-relative model paths ---
MUSETALK_ROOT = os.environ.get("MUSETALK_ROOT", os.path.dirname(os.path.abspath(__file__)))
os.chdir(MUSETALK_ROOT)
sys.path.insert(0, MUSETALK_ROOT)

import torch
import gradio as gr
from transformers import WhisperModel

# NOTE: importing this also triggers DWPose (mmpose) init at import time, which needs
# ./models/dwpose/dw-ll_ucoco_384.pth to exist and CWD == MuseTalk root (done above).
import scripts.realtime_inference as rt
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.face_parsing import FaceParsing

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VERSION = "v15"
UNET_JSON = "./models/musetalkV15/musetalk.json"
UNET_PTH = "./models/musetalkV15/unet.pth"
WHISPER_DIR = "./models/whisper"
GPU_ID = int(os.environ.get("GPU_ID", "0"))
DEFAULT_BATCH = int(os.environ.get("MUSETALK_BATCH", "4"))   # T4-friendly; raise if VRAM allows
AVATAR_ROOT = f"./results/{VERSION}/avatars"

device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Load models ONCE, then inject them into the realtime_inference module namespace
# so Avatar's methods (which reference module globals) can find them.
# ---------------------------------------------------------------------------
print(f"[musetalk] loading models on {device} ...")
vae, unet, pe = load_all_model(
    unet_model_path=UNET_PTH, vae_type="sd-vae", unet_config=UNET_JSON, device=device,
)
timesteps = torch.tensor([0], device=device)
pe = pe.half().to(device)
vae.vae = vae.vae.half().to(device)
unet.model = unet.model.half().to(device)
weight_dtype = unet.model.dtype

audio_processor = AudioProcessor(feature_extractor_path=WHISPER_DIR)
whisper = WhisperModel.from_pretrained(WHISPER_DIR).to(device=device, dtype=weight_dtype).eval()
whisper.requires_grad_(False)
fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

rt.args = SimpleNamespace(
    version=VERSION, extra_margin=10, parsing_mode="jaw",
    left_cheek_width=90, right_cheek_width=90,
    audio_padding_length_left=2, audio_padding_length_right=2,
    skip_save_images=False,
)
rt.device = device
rt.vae = vae
rt.unet = unet
rt.pe = pe
rt.timesteps = timesteps
rt.weight_dtype = weight_dtype
rt.audio_processor = audio_processor
rt.whisper = whisper
rt.fp = fp
print("[musetalk] ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def list_avatars():
    if not os.path.isdir(AVATAR_ROOT):
        return []
    return sorted(
        d for d in os.listdir(AVATAR_ROOT)
        if os.path.isfile(os.path.join(AVATAR_ROOT, d, "latents.pt"))
    )


_avatar_cache = {}


def _get_avatar(avatar_id, batch_size):
    key = (avatar_id, int(batch_size))
    if key not in _avatar_cache:
        # preparation=False -> loads the cached masks/latents (no re-preparation)
        _avatar_cache[key] = rt.Avatar(
            avatar_id=avatar_id, video_path="", bbox_shift=0,
            batch_size=int(batch_size), preparation=False,
        )
    return _avatar_cache[key]


# ---------------------------------------------------------------------------
# Tab 1 — Setup: prepare an avatar from a (looping) video  [ONE-TIME per video]
# ---------------------------------------------------------------------------
def setup_avatar(video_path, avatar_id, batch_size):
    if not video_path:
        raise gr.Error("Upload a short video clip of the person's face first.")
    avatar_id = (avatar_id or "").strip() or f"avatar_{int(time.time())}"
    adir = os.path.join(AVATAR_ROOT, avatar_id)
    if os.path.exists(adir):
        shutil.rmtree(adir)          # overwrite (avoids the upstream interactive input() prompt)
    _avatar_cache.pop((avatar_id, int(batch_size)), None)

    t0 = time.time()
    rt.Avatar(
        avatar_id=avatar_id, video_path=video_path, bbox_shift=0,
        batch_size=int(batch_size), preparation=True,   # runs prepare_material() (mask+latent cache)
    )
    dt = time.time() - t0
    avatars = list_avatars()
    msg = f"✅ Prepared avatar '{avatar_id}' in {dt:.1f}s. Available avatars: {', '.join(avatars)}"
    return msg, gr.update(choices=avatars, value=avatar_id)


# ---------------------------------------------------------------------------
# Tab 2 — Inference: audio -> talking video (for a prepared avatar)
# ---------------------------------------------------------------------------
def synthesize(avatar_id, audio_path, fps, batch_size):
    if not avatar_id:
        raise gr.Error("Pick a prepared avatar (run the Setup tab first).")
    if not audio_path:
        raise gr.Error("Upload an audio clip.")
    avatar = _get_avatar(avatar_id, batch_size)
    out_name = f"out_{uuid.uuid4().hex[:8]}"
    avatar.inference(audio_path, out_name, int(fps), skip_save_images=False)
    out_path = os.path.join(AVATAR_ROOT, avatar_id, "vid_output", out_name + ".mp4")
    if not os.path.exists(out_path):
        raise gr.Error("Inference finished but produced no file — check the server logs.")
    return out_path


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="MuseTalk — audio-driven talking video") as demo:
    gr.Markdown(
        "# 🎬 MuseTalk — audio-driven lip-sync\n"
        "**Setup** a face video once (it gets face-detected + mask-cached, looped forward/reverse), "
        "then on **Inference** feed any audio and get a video where that face speaks your audio."
    )

    with gr.Tab("1 · Setup avatar (from video)"):
        gr.Markdown(
            "Upload a short clip (5–20s) of the face — it can already be talking; the mouth is "
            "replaced. This runs the one-time preparation (~seconds–minutes) and caches the avatar."
        )
        with gr.Row():
            with gr.Column():
                setup_video = gr.Video(label="Face video (looping source)")
                setup_id = gr.Textbox(label="Avatar name", placeholder="e.g. presenter_1")
                setup_batch = gr.Slider(1, 16, value=DEFAULT_BATCH, step=1, label="Batch size")
                setup_btn = gr.Button("Prepare avatar", variant="primary")
            with gr.Column():
                setup_status = gr.Textbox(label="Status", lines=3)

    with gr.Tab("2 · Inference (audio → video)"):
        gr.Markdown("Pick a prepared avatar and upload/record audio → get the talking video.")
        with gr.Row():
            with gr.Column():
                infer_avatar = gr.Dropdown(choices=list_avatars(), label="Prepared avatar")
                refresh_btn = gr.Button("↻ Refresh list", size="sm")
                infer_audio = gr.Audio(label="Audio to speak", sources=["upload", "microphone"], type="filepath")
                infer_fps = gr.Slider(15, 30, value=25, step=1, label="FPS")
                infer_batch = gr.Slider(1, 16, value=DEFAULT_BATCH, step=1, label="Batch size")
                infer_btn = gr.Button("Generate talking video", variant="primary")
            with gr.Column():
                infer_out = gr.Video(label="Result")

    # wiring (api_name exposes stable HTTP endpoints: /setup and /infer)
    setup_btn.click(setup_avatar, [setup_video, setup_id, setup_batch],
                    [setup_status, infer_avatar], api_name="setup")
    refresh_btn.click(lambda: gr.update(choices=list_avatars()), None, infer_avatar)
    infer_btn.click(synthesize, [infer_avatar, infer_audio, infer_fps, infer_batch],
                    infer_out, api_name="infer")

if __name__ == "__main__":
    # MuseTalk holds one GPU model -> serialize requests.
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=7860)
