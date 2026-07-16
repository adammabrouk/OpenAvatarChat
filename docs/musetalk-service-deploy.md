# From your Mac to Azure — deploy the MuseTalk avatar service

This is the **glue runbook**: take the code you have locally (`musetalk_service/`, the config, the docs) and
get the API + UI running on your Azure T4, then test it. When it's green you add LiveKit.

- Service internals & endpoints → `musetalk_service/README.md`
- Driver / Docker image / firewall basics → `docs/deploy-azure-t4.md`
- LiveKit worker → `docs/musetalk-livekit-worker.md`

---

## 0. What you have locally vs. what's on the VM

- **Local (Mac):** the OpenAvatarChat repo *plus* your new, uncommitted files:
  `musetalk_service/`, `config/chat_with_musetalk_english.yaml`, `docs/musetalk-*.md`.
- **VM:** may have the repo from the avatar work (with the Docker image built), or nothing yet.

The plan: make sure the repo + Docker image exist on the VM, then **copy your new files up**, download the
MuseTalk weights, and run the service inside the image (which already has torch + all MuseTalk deps).

**Each step below is tagged 🖥️ *your Mac* or ☁️ *the VM (after `ssh`)* — run it on that machine.**
`az …` and `rsync …` run on your **Mac** (needs `az login`); `docker …` / `download_models` run **on the VM**.

Set these once (🖥️ **Mac** shell — `az login` first):
```bash
export RG=oac-rg                          # your resource group (from deploy-azure-t4.md)
export VM=oac-t4                           # your VM name
export VM_USER=azureuser
export VM_IP=$(az vm show -d -g $RG -n $VM --query publicIps -o tsv)   # or paste the IP
echo "VM = $VM_IP"
```

---

## 1. Ensure the repo + image exist on the VM  · ☁️ on the VM

**If the VM is fresh**, do the driver + clone + build steps from `docs/deploy-azure-t4.md` first:
- §3a NVIDIA driver, §3b Docker + NVIDIA Container Toolkit
- clone the repo to `~/OpenAvatarChat` (+ submodules)
- build the image: `./build_cuda128.sh` → `open-avatar-chat:latest` (has all MuseTalk deps via `install.py --all`)

**If you already did the avatar deploy**, the repo (`~/OpenAvatarChat`) and image are already there — skip to Step 2.

> Why reuse the image: MuseTalk's `mmcv`/`mmpose`/`dwpose` stack is the painful part, and `install.py --all`
> already baked it into `open-avatar-chat:latest`. We run our service *inside* that image.

---

## 2. Copy your new files up to the VM  · 🖥️ on your Mac

Your `musetalk_service/` (and the config/docs) aren't in upstream, so push them from the Mac:
```bash
# from the repo root on your Mac
rsync -av musetalk_service            $VM_USER@$VM_IP:~/OpenAvatarChat/
rsync -av src/handlers/avatar/musetalk/musetalk_algo.py \
          $VM_USER@$VM_IP:~/OpenAvatarChat/src/handlers/avatar/musetalk/   # patched engine ([PREP] timings, used via the src/ mount)
rsync -av config/chat_with_musetalk_english.yaml $VM_USER@$VM_IP:~/OpenAvatarChat/config/
rsync -av docs/musetalk-*.md docs/livekit_*.py   $VM_USER@$VM_IP:~/OpenAvatarChat/docs/
```
(If you prefer git: commit these on a branch, push, and `git pull` on the VM — same result.)

---

## 3. Download the MuseTalk weights (once)  · ☁️ on the VM (SSH in first)

```bash
ssh $VM_USER@$VM_IP
cd ~/OpenAvatarChat

docker run --rm -v $(pwd)/models:/root/open-avatar-chat/models \
  --entrypoint uv open-avatar-chat:latest \
  run --no-sync scripts/download_models.py --handler musetalk
```
This fills `models/musetalk/...` (unet, whisper, dwpose, sd-vae, face-parse, s3fd) — a few GB.

---

## 4. Open the one port you need  · 🖥️ on your Mac (Azure CLI — or use the Portal)

This opens the VM's firewall so your browser can reach the service. The API/UI path returns a file
(no WebRTC), so you only need **one TCP port: 8000**.

> ⚠️ **Run this on your Mac, NOT inside the VM.** `az` is the Azure control-plane CLI — it configures the
> VM's network from outside. If you're still SSH'd in from Step 3, open a **new Mac terminal** (or type
> `exit` to leave the SSH), then run the command there. It needs `az login` and the `$RG`/`$VM` you set in
> Step 0.

**Option A — Azure CLI (on your Mac):**
```bash
# these are from Step 0 (RG=oac-rg, VM=oac-t4 by default); make sure they're set in THIS terminal:
export RG=oac-rg VM=oac-t4
az vm open-port -g $RG -n $VM --port 8000 --priority 1010
```

**Option B — Azure Portal (no CLI):** portal.azure.com → your VM → **Networking** →
**Add inbound port rule** → Destination port ranges **8000**, Protocol **TCP**, Action **Allow** → **Add**.

Either way, you're just adding an inbound-allow rule for TCP 8000. Nothing runs on the VM in this step.

---

## 5. Run the service inside the image  · ☁️ on the VM

We mount only `musetalk_service/` + `models/` into the image (NOT the whole repo — that would shadow the
baked-in venv), install the few extra deps, and launch uvicorn on 8000:

```bash
cd ~/OpenAvatarChat

docker run -d --name musetalk-svc --gpus all --restart unless-stopped \
  -e XFORMERS_IGNORE_FLASH_VERSION_CHECK=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
  -e MUSETALK_LOG_LEVEL=INFO \
  -v $(pwd)/musetalk_service:/root/open-avatar-chat/musetalk_service \
  -v $(pwd)/src:/root/open-avatar-chat/src \
  -v $(pwd)/models:/root/open-avatar-chat/models \
  -p 8000:8000 \
  --entrypoint bash open-avatar-chat:latest -c '
    cd /root/open-avatar-chat &&
    uv pip install python-multipart nvidia-ml-py &&
    OAC_ROOT=/root/open-avatar-chat \
      uv run --no-sync uvicorn --app-dir musetalk_service api:app --host 0.0.0.0 --port 8000'

docker logs -f musetalk-svc      # watch it load models, then "Uvicorn running on 0.0.0.0:8000"
```
> Things that bite here (already handled above):
> - **Only install `python-multipart` + `nvidia-ml-py`.** fastapi/uvicorn/numpy/opencv/librosa are already
>   in the image; installing opencv/numpy again upgrades numpy to 2.x and breaks the pinned env.
> - **`XFORMERS_IGNORE_FLASH_VERSION_CHECK=1`** — MuseTalk→diffusers→xformers hard-fails on a flash-attn
>   version mismatch without it. (`engine.py` also sets this itself as a backstop.)
> - **`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`** — the `video` capability exposes NVENC so the
>   service can hardware-encode the output mp4 (it auto-falls back to libx264 if unavailable).
> - **Mount `src/` too** — the service imports the repo's MuseTalk engine from `src/handlers/avatar/musetalk`;
>   mounting it means engine fixes (and its prep-stage `[PREP]` timing logs) apply without a rebuild.
>   Mount only `musetalk_service/`, `src/`, `models/` — never the whole repo (it would shadow the baked venv).
> If a previous attempt left a crashed container: `docker rm -f musetalk-svc` before re-running.

### Performance / profiling knobs

| Env var | Default | Meaning |
|---|---|---|
| `MUSETALK_LOG_LEVEL` | `INFO` | `DEBUG` adds rolling per-25-frame progress lines |
| `MUSETALK_DEBUG` | `0` | `1` = per-batch UNet/VAE/blend `[PROFILE]` logs from the algo (drill-down) |
| `MUSETALK_BATCH` | `4` | frames per GPU call in `/speak`; try `8` on the T4, watch `mem_peak_mb` |
| `MUSETALK_ENCODER` | `auto` | `auto` = NVENC if usable else libx264; or force `nvenc` / `libx264` |
| `MUSETALK_MAX_FRAMES` | `75` | cap source frames in persona prep (≈3 s loop @25fps); `0` = use all |
| `MUSETALK_GPU_SAMPLE_SEC` | `1.0` | GPU sampling interval during operations; `0` disables |
| `MUSETALK_FPS` | `25` | avatar frame rate. **For live mic mode, set it to your measured throughput** (e.g. `15` on the T4) so generation keeps up with your voice |
| `MUSETALK_WINDOW_SEC` | `1.0` | live mode: audio window per inference round — also the baseline avatar lag behind your voice |
| `MUSETALK_MAX_LAG_SEC` | `2.0` | live mode: max buffered speech; older audio is **dropped** so the avatar stays near-realtime instead of drifting behind |
| `MUSETALK_VAD_THRESHOLD` | `0.5` | live mode: **Silero VAD** speech-probability gate (streaming, uses the repo's own `silero_vad.onnx`) — windows below it idle instead of running inference. **Adjustable live from the UI slider**; the UI shows your live speech probability (green = will lip-sync). One hangover window after speech renders the mouth closing naturally |
| `MUSETALK_SILENCE_RMS` | `0.01` | live mode: fallback energy gate used only if the Silero model can't load |
| `MUSETALK_CONTEXT_SEC` | `0.3` | live mode: audio from the previous window prepended before whisper extraction (context frames dropped after) — stabilizes the mouth at window boundaries; raise to 0.5 if lips still jitter |
| `MUSETALK_JPEG_QUALITY` | `80` | live mode: JPEG quality of streamed frames |

Every `/speak` logs a **STAGE SUMMARY** (whisper / frame_gen / blend / pipe_write / ffmpeg + realtime
factor + GPU util/mem) and writes a sidecar `<output>.mp4.profile.json` in `OUTPUT_DIR`
(`/tmp/musetalk_outputs` in the container). Persona prep logs `[PREP]` stage timings and writes
`prepare.profile.json` into the persona folder (persisted in the mounted `models/`). The mp4 response
carries `X-Render-Seconds`, `X-Audio-Seconds`, `X-Realtime-Factor`, `X-Fps` headers.

### Monitoring the GPU from your side

```bash
curl http://$VM_IP:8000/gpu                  # live util/mem/power via the service (NVML)
# ☁️ on the VM:
nvidia-smi dmon -s pucm -d 1                 # 1 Hz power/util/clock/mem stream while a render runs
watch -n1 nvidia-smi                         # classic view
docker exec musetalk-svc nvidia-smi          # confirm the container actually sees the GPU
# read a profile sidecar:
docker exec musetalk-svc sh -c 'ls -t /tmp/musetalk_outputs/*.profile.json | head -1 | xargs cat'
```
How to read it: **low GPU util with `frame_gen` dominant** → raise `MUSETALK_BATCH`; **high `blend`/
`pipe_write`** → CPU-bound compositing/IO; **high `ffmpeg_wait`** → encoder-bound (check the
`video encoder:` log line says `h264_nvenc`, not `libx264`).
Prepared personas land in the mounted `models/musetalk/avatar_model/` → they persist across restarts and are
shared with the LiveKit worker.

> Native alternative (no Docker): create a venv, `uv run install.py --config config/chat_with_musetalk_english.yaml`
> (installs MuseTalk deps), `pip install -r musetalk_service/requirements.txt`, then
> `OAC_ROOT=$(pwd) uvicorn api:app --app-dir musetalk_service --host 0.0.0.0 --port 8000`.

---

## 6. Test it  · 🖥️ browser / Mac

Open **http://$VM_IP:8000** in your browser:
1. **Set up a persona** — name it, upload a short face video → *Prepare persona* (one-time; watch `docker logs`).
2. **Send a wav** — pick the persona, upload a wav → *Send audio* → the lip-synced video plays back.

Or from the Mac via the API:
```bash
curl -F name=presenter_1 -F video=@face.mp4  http://$VM_IP:8000/personas
curl                       http://$VM_IP:8000/personas
curl -F audio=@speech.wav  http://$VM_IP:8000/personas/presenter_1/speak -o out.mp4 && open out.mp4
```

If it works, you've validated the whole MuseTalk path (persona setup + audio→avatar) end-to-end.

---

## 6b. Live mic mode — speak and watch the avatar follow  · 🖥️ Mac + ☁️ VM

Section 3 of the UI streams your microphone to the server over a WebSocket (same TCP port 8000 —
no WebRTC/TURN needed) and streams JPEG frames back: the avatar idles on its loop while you're
silent and lip-syncs your speech about one audio window (~1 s) behind you.

**Exact steps:**

1. **Re-create the container with the fps matched to your GPU's measured throughput.** If your
   `/speak` STAGE SUMMARY showed ~15 effective fps on the T4, run at 15 — at 25 the GPU can't keep
   up with incoming speech and the avatar drifts further behind the longer you talk:
   ```bash
   # ☁️ on the VM
   docker rm -f musetalk-svc
   # re-run the Step 5 `docker run` with ONE extra env line:
   #   -e MUSETALK_FPS=15 \
   # (keep MUSETALK_DEBUG off for live use — its per-batch logging costs real throughput)
   ```
   Personas do NOT need re-preparing — fps only affects inference pacing, not the cached data.

2. **Open the UI through an SSH tunnel** — browsers only allow the microphone on a *secure
   context*, and `http://<VM_IP>:8000` isn't one; `http://localhost:8000` is:
   ```bash
   # 🖥️ on your Mac (leave it running)
   ssh -L 8000:localhost:8000 $VM_USER@$VM_IP
   ```
   Then open **http://localhost:8000** (not the VM IP).

3. **Section 3 → pick your persona (in section 2's dropdown) → "🎙️ Start live"** → allow the mic
   and talk. The status line updates every second:
   - `stream 15/15 fps` — the server is holding its frame clock;
   - `lag ~1.2s` — how far the avatar is behind your voice (window + queued frames);
   - `audio buffered` climbing past ~2 s ⚠️ — generation can't keep up: lower `MUSETALK_FPS`
     or raise `MUSETALK_BATCH`.

   Server-side, `docker logs -f musetalk-svc` shows the session (`live[<persona>] session start/end`).

**How it works / knobs:** the browser plays the persona's **idle loop natively** (`GET
/personas/<id>/idle.mp4` — rendered once from the prepared cycle and cached, so the very first live
session takes a few extra seconds) and the WebSocket only carries frames **while there is speech**:
audio is sliced into `MUSETALK_WINDOW_SEC` (default 1.0 s) windows → silence gate → whisper features
→ batched UNet+VAE (`MUSETALK_BATCH`) → speech frames streamed FIFO on a `MUSETALK_FPS` clock, overlaid
on the loop (with a short cross-fade); when frames stop, the overlay fades back to the loop. Silence
costs zero GPU AND zero bandwidth. Smaller windows = lower lag but worse GPU batching; 0.6–1.0 s is
the sweet spot. This is the same design as the LiveKit worker (§8) — the live tab is the single-user,
one-port preview of it.

**Staying realtime (drop policy):** live mode never lets the avatar drift more than
`MUSETALK_MAX_LAG_SEC` (default 2 s) behind you — if the GPU can't keep up with your speech, the
*oldest* buffered audio is dropped (the stats line shows `dropped Xs audio`; the avatar skips those
words but stays current). Likewise the browser paints on its own fps clock and skips burst frames
after a network stall, so the video never "fast-forwards". If you see constant drops, the fps is
still above the GPU's real throughput → lower `MUSETALK_FPS`.

**Idle↔speech continuity:** the browser reports its idle-loop position twice a second; when a
speech segment starts, generation continues from (about) that cycle position, and each streamed
frame carries its cycle index so that when speech ends the loop is seeked back to exactly where the
generated frames left off — no pose jump in either direction.

**Mouth stability:** three layers — the browser captures with `noiseSuppression` +
`echoCancellation` + `autoGainControl`; Silero VAD keeps noise from triggering lip motion at all;
and `MUSETALK_CONTEXT_SEC` of the previous window is prepended before whisper extraction so
window boundaries don't make the lips flutter.

**Choosing the persona source video:** during silence the avatar plays your source clip untouched —
only speech repaints the mouth. So pick (or trim to) a segment where the person is **not talking**
(mouth closed/neutral, natural blinks). A clip of someone mid-speech will look like silent
mouth-flapping whenever the avatar idles.

---

## 7. Manage the container  · ☁️ on the VM

```bash
docker logs -f musetalk-svc      # logs
docker restart musetalk-svc      # restart (deps are already installed in the layer cache? no — reinstalled;
                                 #  to avoid re-install on restart, see the note below)
docker stop musetalk-svc && docker rm musetalk-svc   # tear down
```
> The `uv pip install` runs on each container start. To make restarts instant, either bake those 5 packages
> into the image (add them to the root `pyproject.toml` and rebuild) or commit the running container
> (`docker commit musetalk-svc musetalk-svc:latest`) and run that tagged image without the install line.

---

## 8. Next: LiveKit

Once the service is green, wire the LiveKit worker (same personas, same engine):
`docs/musetalk-livekit-worker.md` — it covers the agent side, dispatch, and the LiveKit ports
(`7880/tcp`, `7881/tcp`, `50000–60000/udp` or TURN). Run it with `AVATAR_ID=<the persona you just made>`.

---

## Troubleshooting

- **`ModuleNotFoundError: handlers...`** → `OAC_ROOT` wrong, or you mounted the whole repo over the image
  (shadowing `src/`). Mount only `musetalk_service/` + `models/` as shown.
- **`FileNotFoundError models/musetalk/...`** → Step 3 didn't complete; re-run the download.
- **CUDA / slow / CPU** → confirm `--gpus all` and that `nvidia-smi` works on the host (`deploy-azure-t4.md` §3a).
- **Port not reachable** → Step 4 NSG rule, and check `docker ps` shows `0.0.0.0:8000->8000`.
- **First persona prep is slow** → normal (one-time face-detect + VAE-encode + mask build); it's cached after.
