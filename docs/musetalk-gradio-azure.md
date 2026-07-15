# MuseTalk as a Gradio service on an Azure T4 VM (Setup tab + Inference tab + API)

Goal: run **MuseTalk v1.5** on your Azure T4 as a small web service with two tabs and an HTTP API:

- **Setup tab** — upload a (looping) face video → one-time *preparation* (face detect + VAE-encode
  every frame + build the mouth masks, cached). No training.
- **Inference tab** — pick a prepared avatar + upload **just audio** → get an mp4 where that face
  speaks your audio. The source video loops (forward+reverse) for as long as the audio lasts; only
  the mouth region is regenerated.

> **What this is / isn't.** This is a *file-based* service (audio in → full video out), which is exactly
> "give it an audio and have it speak that audio." It is **not** the low-latency WebRTC streaming avatar —
> that's the full OpenAvatarChat pipeline. MuseTalk itself is realtime-capable on a T4 (~20–30 fps), so a
> short clip renders in a few seconds.

We deliberately use the **official MuseTalk repo + its own weights** (self-consistent model paths), not the
copy vendored inside OpenAvatarChat (which uses a different, patched model layout).

---

## 0. Prerequisites

- The T4 VM from the avatar deploy, with the **NVIDIA driver working** (`nvidia-smi` shows the T4).
  See `docs/deploy-azure-t4.md` §3a if not.
- ~25 GB free disk (weights + your videos + outputs).
- Open a web port (we use **7860**).

```bash
# open the app port (NSG rule)
az vm open-port -g $RG -n $VM --port 7860 --priority 1010
```

---

## 1. System packages

```bash
sudo apt-get update
sudo apt-get install -y git build-essential ffmpeg libsndfile1 python3-venv
ffmpeg -version   # must print a version
```

---

## 2. Get MuseTalk + a Python env

```bash
cd ~
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
```

**Torch (CUDA 11.8 build — works on the T4 / sm_75):**
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu118
```

**MuseTalk deps + mmlab stack** (the fragile part — pin these exactly):
```bash
pip install -r requirements.txt
pip install --no-cache-dir -U openmim
mim install mmengine "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"
```

**Gradio (for the app) + client (for the API):**
```bash
pip install "gradio==5.49.1" gradio_client soundfile librosa
```

> If `mim install mmcv==2.0.1` tries to build from source and stalls, it's a torch/mmcv wheel mismatch —
> keep torch pinned to **2.0.1+cu118** (above) so the prebuilt mmcv wheel is used.

---

## 3. Download the weights

MuseTalk ships a script that lays the models out exactly where the code expects
(`./models/musetalkV15/…`, `./models/sd-vae/…`, `./models/whisper/…`, `./models/dwpose/dw-ll_ucoco_384.pth`,
`./models/face-parse-bisent/…`):

```bash
# from the MuseTalk repo root, with the venv active
sh download_weights.sh
```

Verify the key files exist (the app needs these paths):
```bash
ls models/musetalkV15/unet.pth models/musetalkV15/musetalk.json \
   models/sd-vae models/whisper models/dwpose/dw-ll_ucoco_384.pth \
   models/face-parse-bisent/79999_iter.pth
```
If any are missing, re-run `download_weights.sh` (it's resumable) or fetch that repo manually with
`huggingface-cli download`.

---

## 4. Drop in the Gradio app

Copy **`musetalk_app.py`** (provided alongside this guide) into the **MuseTalk repo root**:

```bash
# from your Mac, or wherever you keep this repo:
scp docs/musetalk_app.py azureuser@<VM_PUBLIC_IP>:~/MuseTalk/musetalk_app.py
```

It must live at the MuseTalk repo root because MuseTalk resolves model paths **relative to the current
working directory** (`./models/...`, `./musetalk/utils/dwpose/...`). The app `os.chdir()`s to its own
directory to guarantee this.

What the app does (recap of the mechanics you asked about):
- **Setup** calls MuseTalk's `Avatar(..., preparation=True)` → runs the one-time first pass that
  face-detects every frame, VAE-encodes them, and **writes a mouth mask for every frame** into
  `results/v15/avatars/<id>/` (`masks`, `latents.pt`, `coords.pkl`). Cached; never redone unless you
  re-prepare.
- **Inference** calls `Avatar(..., preparation=False).inference(audio)` → loads the cached masks/latents
  and, per audio-driven frame, inpaints the mouth and blends it back, then muxes your audio → mp4.

---

## 5. Run it

```bash
cd ~/MuseTalk
source .venv/bin/activate
# optional knobs: GPU_ID=0, MUSETALK_BATCH=4 (raise to 8 on the 16 GB T4 if you want)
python musetalk_app.py
```

First launch loads the models (a few seconds) and initializes DWPose. Then open:

```
http://<VM_PUBLIC_IP>:7860
```

> Plain HTTP is fine here (no microphone-permission requirement like the WebRTC avatar). If you want the
> microphone-record option to work in the browser, put it behind HTTPS (self-signed or a reverse proxy).

**Use it:** Setup tab → upload your looping face clip → name it → *Prepare avatar*. Then Inference tab →
select the avatar → upload audio → *Generate*.

---

## 6. The HTTP API

Gradio exposes both actions as endpoints (`/setup`, `/infer`):

```python
from gradio_client import Client, handle_file

c = Client("http://<VM_PUBLIC_IP>:7860")

# one-time: prepare an avatar from a video
status, _ = c.predict(handle_file("face.mp4"), "presenter_1", 4, api_name="/setup")
print(status)

# then: audio -> talking video (returns a local path to the mp4)
video = c.predict("presenter_1", handle_file("speech.wav"), 25, 4, api_name="/infer")
print("saved:", video)
```

---

## 7. Keep it running (systemd)

```bash
sudo tee /etc/systemd/system/musetalk.service >/dev/null <<'UNIT'
[Unit]
Description=MuseTalk Gradio
After=network.target

[Service]
User=azureuser
WorkingDirectory=/home/azureuser/MuseTalk
Environment=GPU_ID=0
Environment=MUSETALK_BATCH=4
ExecStart=/home/azureuser/MuseTalk/.venv/bin/python /home/azureuser/MuseTalk/musetalk_app.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now musetalk
journalctl -u musetalk -f      # watch logs
```

---

## 8. Performance on the T4

- **VRAM**: MuseTalk uses ~4–6 GB — plenty of room on the 16 GB T4. You can raise `MUSETALK_BATCH` to 8.
- **Speed**: ~20–30 fps generation. A 10 s audio clip renders in roughly real time (a handful of seconds).
- **Don't run this at the same time as the FlashHead avatar** — FlashHead is VRAM-heavy and they'd contend.
- Preparation cost is **one-time per video**; after that, inference reuses the cache.

---

## 9. Troubleshooting

- **`ModuleNotFoundError: mmpose` / mmcv build errors** → torch not on 2.0.1+cu118; reinstall torch (Step 2)
  then `mim install "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"`.
- **`FileNotFoundError ... models/dwpose/dw-ll_ucoco_384.pth`** (at import) → weights not downloaded, or you
  didn't run from the MuseTalk repo root. Re-run `download_weights.sh`; keep `musetalk_app.py` at the repo root.
- **`ffmpeg: command not found`** in the output step → `sudo apt-get install -y ffmpeg`.
- **A face isn't detected / mouth looks off** → the source clip needs a clear, mostly front-facing face; try a
  cleaner clip. (`bbox_shift` tuning exists in MuseTalk v1; v1.5 fixes it at 0.)
- **Output has no audio** → ensure `ffmpeg` is on PATH (the app muxes audio in the final step).
- **Slower than expected / CPU** → confirm `nvidia-smi` shows the process on the GPU; if it fell back to CPU,
  the torch install isn't the CUDA build.

---

## 10. Where this fits with the rest

- This gives you **audio → talking-head video** on the T4, file-based, with your own face video.
- For a **live conversational** avatar (mic → ASR → LLM → TTS → streaming lips), that's the OpenAvatarChat
  pipeline with the MuseTalk avatar handler (`config/chat_with_openai_compatible_bailian_cosyvoice_musetalk.yaml`),
  which is the realtime WebRTC path — a separate, heavier setup.
