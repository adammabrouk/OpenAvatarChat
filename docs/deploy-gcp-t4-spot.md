# Running OpenAvatarChat on a GCP T4 Spot VM

Target config: `config/chat_with_openai_compatible_bailian_cosyvoice.yaml`
(ASR=SenseVoice local · LLM+TTS=Bailian API · Avatar=LiteAvatar on the T4)

VM shape (matches the console screenshot): **n1-highmem-8** (8 vCPU / 52 GB) + **1× NVIDIA T4**, Spot provisioning.
`n1-standard-8` (30 GB) also works and is cheaper.

---

## 0. Prerequisites
- `gcloud` CLI authenticated: `gcloud auth login` then `gcloud config set project <PROJECT_ID>`.
- A **DASHSCOPE_API_KEY** (Alibaba Bailian) — this config's LLM + TTS.
- GPU quota: new projects often have `GPUS_ALL_REGIONS = 0`. Check / raise:
  `gcloud compute regions describe europe-west4 --format="value(quotas)"` or
  IAM & Admin → Quotas → filter "GPUs (all regions)" / "NVIDIA T4 GPUs". Request ≥1.
  (You already created one in the console, so quota is likely fine.)

Set shell variables (edit PROJECT/ZONE):
```bash
export PROJECT=$(gcloud config get-value project)
export ZONE=europe-west4-a          # T4 zones: europe-west4-{a,b,c}, us-central1-{a,b,f}, etc.
export VM=oac-t4
```

## 1. Firewall — open web + WebRTC UDP
WebRTC media needs UDP, not just the web port. coturn (bundled via docker compose)
uses 3478 + a relay range.
```bash
gcloud compute firewall-rules create oac-web \
  --allow tcp:8282 --direction INGRESS --network default \
  --source-ranges 0.0.0.0/0 --target-tags oac

gcloud compute firewall-rules create oac-turn \
  --allow udp:3478,udp:49152-65535 --direction INGRESS --network default \
  --source-ranges 0.0.0.0/0 --target-tags oac
```

## 2. Create the Spot VM (Deep Learning image = drivers + Docker preinstalled)
```bash
gcloud compute instances create $VM \
  --zone=$ZONE \
  --machine-type=n1-highmem-8 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=150GB \
  --boot-disk-type=pd-ssd \
  --tags=oac
```
Notes:
- `--provisioning-model=SPOT` = cheap but can be preempted; `STOP` (not DELETE) on
  preemption so the disk survives and you can restart.
- Image `common-cu129-ubuntu-2204-nvidia-580` (Ubuntu 22.04, driver 580, CUDA 12.9)
  ships the NVIDIA driver + Docker + nvidia-container-toolkit preinstalled — no
  `install-nvidia-driver` metadata needed. Driver 580 runs the CUDA 12.8 container
  natively. Ubuntu 22.04 matches the container base image.

## 3. SSH in and verify the GPU
```bash
gcloud compute ssh $VM --zone=$ZONE
# inside the VM:
nvidia-smi            # must show Tesla T4
docker --version      # preinstalled on the DL image
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi   # GPU visible in Docker
```
If `docker run --gpus` fails, install the toolkit:
```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

## 4. Get the code
```bash
git clone https://github.com/HumanAIGC-Engineering/OpenAvatarChat.git
cd OpenAvatarChat
git submodule update --init --recursive --depth 1
```

## 5. Secret
```bash
echo "DASHSCOPE_API_KEY=sk-xxxxxxxx" > .env
```

## 6. Build the CUDA image
```bash
chmod +x build_cuda128.sh
./build_cuda128.sh          # open-avatar-chat:latest, ~10-25 min
```

## 7. Download the LiteAvatar model (into the mounted ./models)
```bash
docker run --rm -v $(pwd)/models:/root/open-avatar-chat/models \
  --entrypoint uv open-avatar-chat:latest \
  run --no-sync scripts/download_models.py --handler liteavatar
```

## 8. Run (docker compose brings up coturn + the app)
```bash
docker compose up
# app listens on https://0.0.0.0:8282
```

## 9. Open from your Mac
```bash
gcloud compute instances describe $VM --zone=$ZONE \
  --format="get(networkInterfaces[0].accessConfigs[0].natIP)"   # external IP
```
Browse to `https://<EXTERNAL_IP>:8282` → accept the self-signed cert → allow mic.

---

## Cost & lifecycle (Spot)
- T4 Spot + n1-highmem-8 ≈ **$0.15–0.25/hr** (region-dependent), vs ~$0.55 on-demand.
- **Stop when idle** — you pay for the running VM + disk:
  ```bash
  gcloud compute instances stop  $VM --zone=$ZONE   # stop compute billing (disk persists)
  gcloud compute instances start $VM --zone=$ZONE   # resume (Spot: subject to capacity)
  ```
- Spot preemption: the VM can be stopped by GCP anytime; just `start` it again.
  For a demo you can't risk being preempted, drop `--provisioning-model=SPOT`.

## Troubleshooting
- **`nvidia-smi` not found right after create** → driver still installing on first boot;
  wait 1-2 min and re-SSH, or `sudo /opt/deeplearning/install-driver.sh`.
- **Page loads, audio/video never connects** → WebRTC UDP blocked; recheck the
  `oac-turn` firewall rule and that the VM has the `oac` network tag; confirm
  `docker compose ps` shows coturn running.
- **Browser blocks mic** → must be HTTPS (the self-signed cert is fine, just accept it).
- **401 from LLM/TTS** → `DASHSCOPE_API_KEY` missing/wrong in `.env`.
- **CUDA container fails with driver error** → host driver too old; install 550:
  `sudo apt-get install -y nvidia-driver-550 && sudo reboot`.
