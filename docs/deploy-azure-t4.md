# Running OpenAvatarChat on an Azure T4 VM

Config: `config/chat_with_flashhead_english.yaml`
(ASR=SenseVoice local · LLM=Azure OpenAI API · TTS=Edge TTS free · Avatar=SoulX-FlashHead on GPU)

An **English** end-to-end talking-avatar. **No DashScope / Alibaba key needed** — the LLM
goes to your Azure OpenAI deployment and TTS uses keyless Microsoft Edge voices:

```
mic → SenseVoice ASR (T4) → Azure OpenAI LLM (API) → Edge TTS (free, English) → SoulX-FlashHead avatar (T4) → WebRTC
```

Only **SenseVoice + FlashHead** run on the GPU; the LLM and TTS are remote.

> **Two files aren't in the upstream repo** and must reach the VM:
> `config/chat_with_flashhead_english.yaml`, and a one-line edit to
> `src/handlers/llm/openai_compatible/llm_handler_openai_compatible.py` (so the LLM reads
> `AZURE_OPENAI_API_KEY`). **Step 4** is where they get there — via a fork you push from
> your Mac, or `scp` (Step 5) as a fallback. Nothing to do about them until Step 4.

## 0. Prerequisites
- Azure subscription with **NCasT4_v3** GPU quota (already unlocked ✅).
- An **Azure OpenAI deployment** (in AI Foundry) for the LLM: note your **resource
  endpoint** (`https://<resource>.openai.azure.com`), a **deployment name**, and an
  **API key**. TTS uses free Edge voices — no key. (No DashScope/Alibaba key needed.)
- `az` CLI logged in; your Mac only needs a browser + SSH.

Pick the region where you unlocked quota, and pin an availability zone that has T4
capacity (in `francecentral`, T4/NCasT4_v3 is only in **zones 2 and 3**):
```bash
export RG=oac-rg
export LOC=francecentral       # must match the region your T4 quota is in
export VM=oac-t4
export ZONE=2                  # francecentral T4 → 2 or 3 only
az group create -n $RG -l $LOC
```

## 1. Create the VM from a plain Ubuntu 22.04 image
We deliberately **avoid the NVIDIA GPU-Optimized marketplace VMI** — it has a poor
track record on Azure (stale drivers, marketplace-terms/plan friction, breakage on
newer sizes). Instead we boot a clean, first-party **Ubuntu 22.04 LTS Gen2** image
and install the driver + Docker + toolkit ourselves in Step 3. No marketplace terms,
no `--plan-*` flags.

```bash
# Create the VM (pinned to a zone with T4 capacity)
az vm create -g $RG -n $VM \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
  --size Standard_NC8as_T4_v3 \
  --zone $ZONE \
  --admin-username azureuser --generate-ssh-keys \
  --os-disk-size-gb 150 --public-ip-sku Standard
```
Notes:
- `Standard_NC8as_T4_v3` = 8 vCPU / 56 GB / 1× T4 16 GB. `NC4as_T4_v3` also works.
- Plain Ubuntu ships **no GPU driver** — that's expected. Step 3 installs everything
  and reboots once. Budget ~5 min for that the first time.
- **Faster alternative (driver preinstalled):** Microsoft's first-party Ubuntu-HPC
  image ships the NVIDIA driver + CUDA — `--image microsoft-dsvm:ubuntu-hpc:2204:latest`.
  You still add Docker + the container toolkit (Step 3b), but skip the driver install
  and reboot. This is *not* the NVIDIA marketplace image and has none of its baggage.

### Disk sizing — read before you pick `--os-disk-size-gb`
Everything lands on the **OS disk** (the VM's temp/ephemeral disk is wiped on
deallocate, so we don't rely on it). The two big consumers are the Docker image and
the per-*brick* model weights:

| What | Approx. size |
|---|---|
| Docker image `open-avatar-chat:latest` (CUDA 12.8 **devel** + cuDNN + torch cu128 stack) | ~15–20 GB |
| Docker build cache (transient, in `/var/lib/docker` during `build`) | +10–15 GB |
| SenseVoice ASR (used by this config) | ~1 GB |
| Avatar brick — **FlashHead** (this config's default; SoulX-FlashHead-1.3B Lite+Pro + wav2vec2) | ~5–8 GB |
| Avatar brick — **MuseTalk** (MuseTalk + SD-VAE + Whisper + DWPose + SyncNet + face-parse + s3fd) | ~6–8 GB |
| Avatar brick — **LiteAvatar** (lighter fallback) | ~1.5 GB |
| Avatar brick — **LAM** (wav2vec2 + audio2exp) | ~1 GB |
| HuggingFace cache duplication (`hf download --local-dir` may keep a 2nd copy in `~/.cache/huggingface`) | +30–50% of the model sizes |

- **`--os-disk-size-gb 150` (used above)** is the recommended default: comfortable
  for the image + the FlashHead brick, and still fine even if you pull **all** bricks
  (peak usage ~60–70 GB with cache duplication).
- Bare minimum for just the FlashHead path: ~80 GB would work, but leaves little room
  and no margin for a second brick — not worth the pennies saved.
- If you plan to benchmark **MuseTalk *and* FlashHead *and* several avatars**, bump to
  **256 GB** for breathing room.
- You can grow the OS disk later without rebuilding the VM: `az vm deallocate`, then
  `az disk update -g $RG -n <osDiskName> --size-gb 256`, then `az vm start`.

## 2. Open the firewall (NSG) — WebRTC needs UDP
```bash
# Web UI (HTTPS)
az vm open-port -g $RG -n $VM --port 8282 --priority 1001
# TURN signalling + relay (coturn) for WebRTC media
az network nsg rule create -g $RG --nsg-name ${VM}NSG \
  -n turn --priority 1002 --access Allow --protocol Udp \
  --destination-port-ranges 3478 49152-65535
```

## 3. SSH in and install the GPU stack
```bash
# get the public IP
az vm show -d -g $RG -n $VM --query publicIps -o tsv
ssh azureuser@<VM_PUBLIC_IP>
```

**3a. NVIDIA driver** — *skip this sub-step if you used the Ubuntu-HPC image* (driver
already present; jump to 3b). On plain Ubuntu, install a **pinned** driver explicitly.

> ⚠️ Don't use `ubuntu-drivers install --gpgpu` on Azure — it's flaky here (often
> selects a `-server` branch or nothing) and the failure is silent. And **never**
> `apt install nvidia-utils-XXX` from apt's "command not found" suggestion: that
> installs only the `nvidia-smi` *binary*, not the kernel driver, so you'd get an
> `nvidia-smi` that runs but reports "No devices found".

```bash
# Kernel headers for the RUNNING kernel — DKMS needs these to build the module.
sudo apt-get update
sudo apt-get install -y build-essential "linux-headers-$(uname -r)"

# Install the FULL driver, pinned to a version known-good for T4 + CUDA 12.8.
sudo apt-get install -y nvidia-driver-550

# Confirm the module actually BUILT before rebooting.
dkms status          # want: nvidia/550.x, <kernel>, x86_64: installed

sudo reboot          # REQUIRED — the kernel module only loads after a reboot
```
Reconnect after the reboot, then confirm the GPU is up:
```bash
ssh azureuser@<VM_PUBLIC_IP>
nvidia-smi                              # should show Tesla T4, driver 550.x
```
If `dkms status` was empty or showed a build error, the headers didn't match your
running kernel (Azure often boots an `-azure` kernel). Fix and reinstall:
```bash
sudo apt-get install -y linux-headers-azure linux-modules-extra-azure
sudo apt-get install --reinstall -y nvidia-driver-550
dkms status && sudo reboot
```

**3b. Docker + NVIDIA Container Toolkit** (needed on both images):
```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# NVIDIA container toolkit repo, then install + wire it into Docker
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

**3c. Verify the GPU is visible inside Docker:**
```bash
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi     # shows Tesla T4
```

## 4. Get the code (with your two files)
Recommended: put the two files on a **fork you own**, then clone that fork on the VM so
the code and your files arrive together. Order matters — **4a on your Mac, then 4b on the VM.**

**4a. On your Mac — push your two files to a fork** (one-time):
```bash
# Fork first in the browser: https://github.com/HumanAIGC-Engineering/OpenAvatarChat → "Fork"
cd ~/OpenAvatarChat                                   # your Mac clone, where the 2 files already exist
git remote add fork https://github.com/<your-gh-user>/OpenAvatarChat.git
git checkout -b english-flashhead-azure
git add config/chat_with_flashhead_english.yaml \
        src/handlers/llm/openai_compatible/llm_handler_openai_compatible.py
git commit -m "English FlashHead + Azure OpenAI pipeline"
git push -u fork english-flashhead-azure
```
Make the fork **public** so the VM can clone it without credentials (private fork → the VM
needs a GitHub token/deploy key, or just use the Step 5 `scp` fallback instead).

**4b. On the VM — clone that fork's branch:**
```bash
git clone -b english-flashhead-azure \
  https://github.com/<your-gh-user>/OpenAvatarChat.git OpenAvatarChat
cd OpenAvatarChat
git submodule update --init --recursive --depth 1
```
Because the branch already contains both files, you can **skip Step 5**.

> **Updating a file later:** edit it on your Mac → `git commit` + `git push` (from the
> `english-flashhead-azure` branch) → on the VM `git pull`. If the change was under `src/`
> (e.g. the handler), also rerun the Step 7 build so it's baked into the image; config-only
> changes just need `docker compose restart` (the `config/` dir is mounted).

**Didn't fork?** Clone upstream instead, and copy the two files in Step 5:
```bash
git clone https://github.com/HumanAIGC-Engineering/OpenAvatarChat.git
cd OpenAvatarChat
git submodule update --init --recursive --depth 1
```

## 5. (Only if you didn't fork) Copy the two files to the VM
**Skip this step if you cloned your fork in Step 4** — the files are already there.

Otherwise, copy them from your Mac (where they live) to the upstream clone on the VM.
Run these **on your Mac**, in the repo root, replacing `<VM_PUBLIC_IP>`:
```bash
# the pipeline config (volume-mounted at runtime → takes effect without a rebuild)
scp config/chat_with_flashhead_english.yaml \
    azureuser@<VM_PUBLIC_IP>:~/OpenAvatarChat/config/

# one-line change so the LLM handler reads AZURE_OPENAI_API_KEY (baked into the image → Step 7 rebuild)
scp src/handlers/llm/openai_compatible/llm_handler_openai_compatible.py \
    azureuser@<VM_PUBLIC_IP>:~/OpenAvatarChat/src/handlers/llm/openai_compatible/
```
Everything from here runs **on the VM** over SSH (`cd ~/OpenAvatarChat`).

## 6. Configure the LLM (Azure OpenAI)
**6a. Put your Azure key in `.env`** (loaded automatically; stays out of git):
```bash
cd ~/OpenAvatarChat
echo "AZURE_OPENAI_API_KEY=xxxxxxxx" > .env
```
**6b. Point the config at your Azure deployment** — edit two lines in
`config/chat_with_flashhead_english.yaml`:
```yaml
      LLMOpenAICompatible:
        model_name: "your-deployment-name"                              # Azure DEPLOYMENT name (Foundry → Deployments), NOT "gpt-4o"
        api_url: "https://YOUR-RESOURCE.openai.azure.com/openai/v1/"     # your resource endpoint + /openai/v1/
```
- Azure OpenAI's **v1** endpoint (`/openai/v1/`) is OpenAI-SDK compatible — no
  `api-version` needed, and the API key is passed as-is.
- `model_name` is the **deployment** name you chose in Foundry, not the base model id.
- Optional: change the English voice under `Edge_TTS.voice`
  (`en-US-AriaNeural`, `en-GB-SoniaNeural`, `en-AU-NatashaNeural`, …), or swap the
  avatar portrait via `FlashHead.cond_image_path`.

## 7. Build the CUDA image
```bash
chmod +x build_cuda128.sh
./build_cuda128.sh            # produces open-avatar-chat:latest (first build ~10-25 min)
```
The Dockerfile runs `install.py --all`, so the image contains every handler's deps
(FlashHead + Edge-TTS included). The handler change from Step 5 is baked in here — if
you ever re-copy a `src/` file, rerun this (it's **cache-fast**: only the source-copy
layer rebuilds, not the dependency install).

## 8. Download the FlashHead models (~5–8 GB) into the mounted ./models
```bash
docker run --rm -v $(pwd)/models:/root/open-avatar-chat/models \
  --entrypoint uv open-avatar-chat:latest \
  run --no-sync scripts/download_models.py --handler flashhead
```
(SenseVoice ASR auto-downloads on first run.)

## 9. Point docker compose at the config and run
Edit the `open-avatar-chat` service `command:` in `docker-compose.yml`:
```yaml
    command: >
      --config=/root/open-avatar-chat/config/chat_with_flashhead_english.yaml
```
Then bring it up (also starts coturn for WebRTC):
```bash
docker compose up
```
The app listens on `https://0.0.0.0:8282`. The `config/` dir is volume-mounted, so later
config tweaks need no rebuild — just `docker compose restart`.

> ⚠️ **FlashHead does NOT run on a T4.** SoulX-FlashHead uses FlashAttention-2, which
> **requires an Ampere GPU (SM 8.0+)**. A T4 is Turing (SM 7.5), so the pipeline loads,
> connects, and then crashes the moment it renders a frame:
> `RuntimeError: FlashAttention only supports Ampere GPUs or newer`. Use **LiteAvatar**
> on the T4 (next section), or move FlashHead to an Ampere+ GPU (A100 / L4 / A10 — e.g.
> Azure NC A100 v4, NVadsA10 v5). The English LLM + TTS are identical either way.

## 9c. T4: use LiteAvatar instead of FlashHead
A ready config is provided: **`config/chat_with_liteavatar_english.yaml`** (same Azure
LLM + Edge TTS, avatar swapped to LiteAvatar — light and realtime on a T4). Edit its
`turn_config` IP and the two Azure LLM lines, then:

**1. Mount `./src`** so LiteAvatar's downloaded NN weights persist and your on-VM code
edits take effect without a rebuild. In `docker-compose.yml`, under the
`open-avatar-chat` service `volumes:`, add:
```yaml
      - ./src:/root/open-avatar-chat/src
```
(The image already contains every handler's deps — the Dockerfile runs `install.py --all`
— so no dependency rebuild is needed to switch avatars.)

**2. Point compose at the LiteAvatar config** — change the `command:`:
```yaml
    command: >
      --config=/root/open-avatar-chat/config/chat_with_liteavatar_english.yaml
```

**3. Download LiteAvatar's NN weights** once (writes into the now-mounted `./src`):
```bash
docker compose run --rm --entrypoint "" open-avatar-chat \
  uv run --no-sync scripts/download_models.py --handler liteavatar
```
The avatar **data** (`20250408/sample_data`) auto-downloads from ModelScope into the
volume-mounted `./resource/avatar/liteavatar/` on first run — nothing to do there.

**4. Run:**
```bash
docker compose up -d
docker compose logs -f open-avatar-chat     # want handlers load, then "Uvicorn running on https://…"
```

> To keep FlashHead for a future Ampere box, the original
> `config/chat_with_flashhead_english.yaml` is untouched — just point `command:` back at it.

## 9b. Wire up WebRTC / TURN — **required, or the avatar never renders**
`docker compose up` starts coturn, but nothing connects it to the app or the browser out
of the box. **Symptom if you skip this:** the page connects and your self-view appears,
then the browser re-POSTs `/webrtc/offer` over and over and no avatar/audio ever shows.
The server log says it **twice**:
```
No valid rtc provider configuration found, STUN/TURN will not be valid.
```
On a cloud VM the browser and the container can't reach each other directly (NAT), and
STUN alone won't help (aiortc opens random UDP ports the NSG doesn't allow) — so media
must go through coturn's TURN **relay**. Three edits make that happen. All are hand edits
in volume-mounted files — no image rebuild.

**1. Point the app at coturn** — open `config/chat_with_flashhead_english.yaml`. The TURN
settings go under the **`RtcClient` handler** (`chat_engine.handler_configs.RtcClient`),
as a `turn_config` key — **not** under `service:` (there is no `rtc_config` field on the
service; putting it there makes the app exit on startup with `exited with code 0`).

Find this block near the top of `handler_configs:` and add the indented `turn_config`
lines, using the VM's **public** IP:
```yaml
  chat_engine:
    handler_configs:
      RtcClient:
        module: client/rtc_client/client_handler_rtc
        connection_ttl: 900
        turn_config:                                 # <-- add this block (8-space indent)
          turn_provider: turn_server
          urls:
            - "turn:<VM_PUBLIC_IP>:3478?transport=udp"
          username: "admin"                          # must match coturn's user=admin:admin
          credential: "admin"
```
Indentation matters in YAML: `turn_config:` lines up with `module:`/`connection_ttl:`
(8 spaces), its children are 10 spaces, and the `- "turn:…"` list item is 12 spaces.

**2. Fix coturn's config** — open `coturn-data/turnserver.conf`. As shipped it has two
bugs. The relay-port lines use `:` instead of `=` (coturn logs `Bad configuration format:
min-port:49152` and silently ignores them), and it advertises its **private** IP for
relays. Change these two lines:
```
min-port:49152      ->   min-port=49152
max-port:65535      ->   max-port=65535
```
and add one new line at the end (this is what makes relayed media reach the browser on
Azure — without it, the connection negotiates but no video/audio flows):
```
external-ip=<VM_PUBLIC_IP>
```
Optional tidy-up: the `key=/etc/turn_key.pem` line uses coturn's wrong option name
(should be `pkey=`), which causes the harmless `Bad configuration format: key` warning.
Since you're not using TURN-over-TLS here, you can delete the `tls-listening-port=5349`,
`cert=…`, and `key=…` lines to silence the TLS warnings entirely. Leaving them is fine too.

**3. Confirm the NSG UDP rule from Step 2 is actually applied** (UDP `3478` +
`49152-65535`). Without it the relay is unreachable and you're back to the same symptom.

Then restart and verify the app now advertises TURN:
```bash
docker compose down && docker compose up -d
docker compose logs open-avatar-chat 2>&1 | grep -i "rtc provider"
# want:  Use turn_server as rtc turn provider.    (NOT "No valid rtc provider…")
```
`config/` and `coturn-data/` are volume-mounted, so these edits need **no** image rebuild.

> The self-signed cert also trips coturn's TLS listener (`cannot start TLS … private key
> file is not set properly`) — **harmless**. The browser only uses TURN over plain UDP
> 3478 here; TLS/DTLS on 5349 isn't needed.

## 10. Open from your Mac
Browse to: `https://<VM_PUBLIC_IP>:8282`
- Accept the self-signed cert warning.
- Allow microphone (and camera if you enable video).
- Start talking in English — ASR + avatar run on the T4, the LLM goes to your Azure
  OpenAI deployment, and TTS uses free Edge voices.

## Cost note
NC8as_T4_v3 is roughly ~$0.75/hr pay-as-you-go (region-dependent), or far less
on Spot. **Deallocate the VM when idle** — you're billed for the running VM,
not just GPU time:
```bash
az vm deallocate -g $RG -n $VM     # stop billing (keeps disk)
az vm start      -g $RG -n $VM     # resume later
```

## Native (non-Docker) alternative
If you run directly on the VM without Docker, the two files from Step 5 are already in
the working tree — skip Steps 5 and 7 and use the three README commands (still set
`AZURE_OPENAI_API_KEY` in `.env` and edit the config as in Step 6):
```bash
uv run install.py --config config/chat_with_flashhead_english.yaml
uv run scripts/download_models.py --handler flashhead
uv run src/demo.py --config config/chat_with_flashhead_english.yaml
```

## Troubleshooting
- **`nvidia-smi: command not found` or "no devices"** → the driver isn't loaded.
  First confirm the GPU is attached: `lspci | grep -i nvidia` should list the Tesla T4.
  If it does, the driver just isn't built/loaded — redo Step 3a's pinned install.
  Do **not** `apt install nvidia-utils-XXX` (the "command not found" suggestion): that
  gives you the `nvidia-smi` binary with **no** kernel driver. Check the module built
  with `dkms status` (want `nvidia/550.x ... installed`) and that you **rebooted** — the
  module only loads after a reboot. If DKMS shows a build error, install the matching
  headers (`sudo apt-get install -y linux-headers-$(uname -r)` or `linux-headers-azure`),
  then `sudo apt-get install --reinstall -y nvidia-driver-550 && sudo reboot`.
- **`docker: could not select device driver ... [[gpus]]`** → the container toolkit isn't
  wired into Docker; re-run the `nvidia-ctk runtime configure` + `systemctl restart docker`
  lines in Step 3b.
- **`az vm create` capacity/zone error** → the pinned zone has no T4 stock right now; try
  the other allowed zone (`ZONE=3`) or another region where you hold quota.
- **Page loads, self-view shows, but no avatar/audio and `/webrtc/offer` repeats in the
  log** → WebRTC media never established. Almost always the **Step 9b** wiring: the log
  shows `No valid rtc provider configuration found` (missing `turn_config` under the
  `RtcClient` handler), and/or
  coturn logs `Bad configuration format: min-port:49152` / advertises a private relay IP
  (missing `external-ip`). Fix all three edits in Step 9b. If the app log already says
  `Use turn_server as rtc turn provider.` yet it still won't connect, then it's the NSG:
  re-check the UDP rule (3478 + 49152-65535) and that coturn is up (`docker compose ps`).
- **`open-avatar-chat exited with code 0 (restarting)` right after "Load config"** → a
  bad edit to the config YAML. Most common cause: the `turn_config` block was put under
  `service:` (which has no such field) instead of under the `RtcClient` handler — see
  Step 9b. Move it, or check the YAML indentation of whatever you last added.
- **`RuntimeError: FlashAttention only supports Ampere GPUs or newer`** → you're running
  the **FlashHead** avatar on a Turing GPU (T4 = SM 7.5). FlashHead can't run here. Switch
  to LiteAvatar per Step 9c, or move to an Ampere+ GPU. Everything else in the log up to
  that point (ASR, LLM, TTS) is working — it's only the avatar renderer that fails.
- **Browser blocks mic** → must be HTTPS; don't use http:// or the raw IP over http.
- **401 / auth error from the LLM** → `AZURE_OPENAI_API_KEY` missing or wrong in `.env`,
  or the endpoint is malformed. `api_url` must end in **`/openai/v1/`** and `model_name`
  must be the **deployment** name (Foundry → Deployments), not the base model id.
  `404 DeploymentNotFound` means the deployment name is wrong; a URL without `/openai/v1/`
  typically yields `api-version is required`. Remember the handler change from Step 5 must
  be built into the image (Step 7) for `AZURE_OPENAI_API_KEY` to be read.
- **TTS silent / no speech** → Edge TTS reaches a Microsoft endpoint; confirm the VM has
  outbound internet. No API key is involved.
- **Want a cleaner URL / real cert** → put an NGINX + Let's Encrypt reverse proxy in front, or use an Azure DNS name.
