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
  -v $(pwd)/musetalk_service:/root/open-avatar-chat/musetalk_service \
  -v $(pwd)/models:/root/open-avatar-chat/models \
  -p 8000:8000 \
  --entrypoint bash open-avatar-chat:latest -c '
    cd /root/open-avatar-chat &&
    uv pip install python-multipart &&
    OAC_ROOT=/root/open-avatar-chat \
      uv run --no-sync uvicorn --app-dir musetalk_service api:app --host 0.0.0.0 --port 8000'

docker logs -f musetalk-svc      # watch it load models, then "Uvicorn running on 0.0.0.0:8000"
```
> Two things that bite here (already handled above):
> - **Only install `python-multipart`.** fastapi/uvicorn/numpy/opencv/librosa are already in the image;
>   installing opencv/numpy again upgrades numpy to 2.x and breaks the pinned env.
> - **`XFORMERS_IGNORE_FLASH_VERSION_CHECK=1`** — MuseTalk→diffusers→xformers hard-fails on a flash-attn
>   version mismatch without it. (`engine.py` also sets this itself as a backstop.)
> If a previous attempt left a crashed container: `docker rm -f musetalk-svc` before re-running.
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
