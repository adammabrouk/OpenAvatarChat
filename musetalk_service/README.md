# MuseTalk avatar microservice (API + UI + LiveKit plugin)

One engine, one persona store, two consumers:

```
                 ┌── REST API + UI  (setup persona from video, send wav → see the avatar react)
   engine.py ────┤
 (MuseTalk +     └── LiveKit worker (plugin-style: agent TTS → lip-synced video in a LiveKit room)
  persona store)
```

- **Personas** = an initial video prepared into cached masks+latents, stored under
  `$PERSONA_DIR/v15/avatars/<id>/`. Set one up once (UI or `POST /personas`) → usable by BOTH the
  wav-test UI and the LiveKit worker.
- Runs **inside the OpenAvatarChat env** (it already has torch + all MuseTalk deps + the model
  downloader). No fragile mmcv/mmpose install to redo.

## Files
| File | Role |
|---|---|
| `engine.py` | shared core: `list/prepare/load_persona`, `render_wav` (wav→mp4) |
| `api.py` + `static/index.html` | FastAPI REST + the setup/test UI |
| `livekit_worker.py` | LiveKit avatar worker (idle-loop + speak) on the same engine |

## Setup (on the Azure T4, inside OpenAvatarChat)
```bash
# 1. MuseTalk weights (once)
uv run scripts/download_models.py --handler musetalk

# 2. Service deps (into the OAC env / image)
pip install -r musetalk_service/requirements.txt

# 3. Point the engine at the repo root
export OAC_ROOT=$(pwd)
```

## Run the API + UI  (this is the "workable UI" — one TCP port)
```bash
cd musetalk_service
OAC_ROOT=/path/to/OpenAvatarChat uvicorn api:app --host 0.0.0.0 --port 8000
```
Open **http://<VM_PUBLIC_IP>:8000** →
1. **Set up a persona**: name + upload a face video → *Prepare persona* (one-time, cached).
2. **Test**: pick the persona, upload a **wav**, *Send audio* → the lip-synced video plays back.

**Azure port:** just one TCP port:
```bash
az vm open-port -g $RG -n $VM --port 8000 --priority 1010
```
(No UDP/TURN needed — this path returns a file, it doesn't stream over WebRTC.)

## Use the API directly (same endpoints the UI calls)
```bash
# setup a persona
curl -F name=presenter_1 -F video=@face.mp4 http://<VM>:8000/personas
# list
curl http://<VM>:8000/personas
# send a wav, save the reaction
curl -F audio=@speech.wav http://<VM>:8000/personas/presenter_1/speak -o out.mp4
```

## LiveKit integration (the plugin)
The worker reuses the persona you prepared in the UI. It needs a LiveKit server + a voice agent
(see `docs/livekit_agent_example.py` and `docs/musetalk-livekit-worker.md` for the agent side, ports,
and dispatch). Run:
```bash
export LIVEKIT_URL=... LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...
export OAC_ROOT=$(pwd) AVATAR_ID=presenter_1
python musetalk_service/livekit_worker.py start
```
LiveKit ports (self-hosted): `7880/tcp`, `7881/tcp`, `50000–60000/udp` (or TURN `3478/udp`+`5349/tcp`).

## Status / caveats
- **API + UI + engine**: buildable and self-contained — the shape you asked for (setup persona, send
  wav, see reaction). Not yet run on a GPU box; expect to smoke-test and adjust.
- **LiveKit worker**: matches the current `livekit.agents.voice.avatar` API; TUNE `MUSETALK_WINDOW_SEC`,
  fps pacing, and whisper overlap on a real run.
- **Personas load models per active persona (LRU=1)** — switching persona reloads (~GB). Fine for the
  UI/test and single-avatar rooms; shard across workers for many concurrent personas.
