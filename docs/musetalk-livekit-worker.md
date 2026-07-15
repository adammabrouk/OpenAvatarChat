# MuseTalk as a LiveKit avatar-worker microservice (idle-loop + speak-on-audio)

This is the **streaming** counterpart to `docs/musetalk-gradio-azure.md` (which is batch: audio→mp4).
Here MuseTalk becomes a **standalone LiveKit avatar worker** that plugs into LiveKit Agents.

```
  caller/mic ─▶ LiveKit voice agent (STT→LLM→TTS)
                     │  TTS audio over DataStream
                     ▼
              MuseTalk avatar worker  ──(video + audio tracks)──▶  LiveKit room  ──▶  client
                     ▲
             prepared face video (looped when idle)
```

- **Idle** (no agent audio): the worker publishes the prepared face **looping** (ping-pong), no GPU inference.
- **Speaking** (agent audio arrives): the worker lip-syncs to that audio and re-publishes it, in sync.
- From the client's side it's **one continuous stream** that transitions between looping and talking.

## Why this shape

LiveKit Agents has a first-class avatar framework (`livekit.agents.voice.avatar`). We implement its
`VideoGenerator` and let `AvatarRunner` handle track publishing + A/V sync; the agent forwards its TTS via
`DataStreamAudioReceiver`/`DataStreamAudioOutput`. This is exactly how Tavus/Simli/Hedra plug in — so our
MuseTalk worker "works smoothly with LiveKit agents" the same way, but self-hosted on your T4.

Refs: [Virtual avatars](https://docs.livekit.io/agents/models/avatar/) · [Video output](https://docs.livekit.io/agents/multimodality/vision/video.md)

## Files (in this repo's `docs/`)

- `livekit_musetalk_worker.py` — the avatar worker (a `VideoGenerator` on OpenAvatarChat's per-frame
  MuseTalk engine `MuseTalkAlgoV15`: `generate_idle_frame` / `extract_whisper_feature` / `generate_frame`).
- `livekit_agent_example.py` — the voice-agent side (routes TTS to the worker; `audio_enabled=False`).

## How the worker maps to MuseTalk

| LiveKit `VideoGenerator` | MuseTalk |
|---|---|
| `__aiter__` yields a frame every `1/fps` | idle → `generate_idle_frame(idx)`; speaking → `generate_frame(chunk, idx)` |
| continuous `idx` (never reset) | same ping-pong cycle index for idle & speak → no head jump |
| `push_audio(AudioFrame)` | buffer PCM → resample to 16 kHz → `extract_whisper_feature` → per-frame `generate_frame` |
| `push_audio(AudioSegmentEnd)` | flush remaining audio, emit segment-end marker |
| `clear_buffer()` (barge-in) | drop pending speech, return to idle (idx keeps advancing) |
| yielded `AudioFrame` (passthrough) | the agent's TTS audio, sliced `1/fps` per frame so lips & voice align |

## Setup on the Azure T4

Reuse your OpenAvatarChat env (it already has the MuseTalk deps + model downloader):

```bash
# 1) MuseTalk models (OpenAvatarChat layout)
uv run scripts/download_models.py --handler musetalk

# 2) LiveKit deps (into the same env / image)
pip install "livekit-agents>=1.0" livekit "livekit-plugins-silero" "livekit-plugins-openai" librosa opencv-python-headless

# 3) Prepare the avatar ONCE from your looping face video
#    (either run the Gradio Setup tab from the other guide, or set AVATAR_VIDEO below for a one-time prep)
```

Environment for the worker:
```bash
export LIVEKIT_URL=wss://<your-livekit>          # or your self-hosted LiveKit
export LIVEKIT_API_KEY=...      LIVEKIT_API_SECRET=...
export OAC_ROOT=$(pwd)                            # OpenAvatarChat repo root
export AVATAR_ID=presenter_1                      # a prepared avatar id
# export AVATAR_VIDEO=/path/face.mp4              # set ONLY to (re)prepare on first launch, then unset
export MUSETALK_FPS=25   AVATAR_AUDIO_SR=24000    # AVATAR_AUDIO_SR must be divisible by FPS
```

Run the worker (registers as agent `musetalk-avatar`):
```bash
python docs/livekit_musetalk_worker.py start
```

Run the voice agent (dispatches the worker into the room):
```bash
python docs/livekit_agent_example.py dev
```

Then connect any LiveKit client (e.g. the [agent-starter-react](https://github.com/livekit-examples/agent-starter-react)) to the room — you'll see the avatar looping, and speaking when the agent talks.

## Wiring notes

- **Same room, two participants.** The voice agent and the avatar worker must be in the same room. Dispatch
  the worker via `RoomConfiguration(agents=[RoomAgentDispatch(agent_name="musetalk-avatar")])` at room
  creation, or run explicit agent dispatch. The agent sends audio to `destination_identity="avatar_worker"`.
- **The agent must not publish audio** (`RoomOutputOptions(audio_enabled=False)`) — the worker owns the
  audio track so voice and lips stay in sync.
- **Plug in your Darija TTS** by swapping the `tts=` in `livekit_agent_example.py` (any streaming TTS works;
  the worker only ever sees audio frames).

## What this needs to run (and why it isn't the quickest test)

The worker has **no UI of its own** — it's a room participant. To see/hear anything you need the full
LiveKit stack:

1. A **LiveKit server** — cloud (livekit.io) or self-hosted. Self-hosted ports to open on the VM:
   - `7880/tcp` (signaling/WS), `7881/tcp` (RTC/TCP fallback)
   - `50000–60000/udp` (RTC media) **or** a TURN server on `3478/udp` + `5349/tcp`
   ```bash
   az vm open-port -g $RG -n $VM --port 7880 --priority 1020
   az network nsg rule create -g $RG --nsg-name ${VM}NSG -n lk-rtc --priority 1021 \
     --access Allow --protocol '*' --destination-port-ranges 7881 3478 5349 50000-60000
   ```
2. The **voice agent** (`livekit_agent_example.py`) — STT+LLM+TTS.
3. A **web client** to join the room (e.g. `livekit-examples/agent-starter-react`).

So: real target architecture, but several moving parts. **To just SEE the avatar (idle-loop + speak) with a
working UI today, use `config/chat_with_musetalk_english.yaml` instead** (browser UI, one web port + TURN,
your own initial video) — see below. Bring in this LiveKit worker once that behavior is validated.

## T4 performance

- MuseTalk ~20–30 fps on the T4; set `MUSETALK_FPS=25` (or 20 if it can't sustain). VRAM ~4–6 GB.
- One GPU model → one avatar session per worker (run more workers for more concurrent rooms).
- Don't co-run with FlashHead.

## What's solid vs. what to tune on the box

**Solid:** the LiveKit contract (VideoGenerator / AvatarRunner / DataStreamAudioReceiver), the idle↔speak
switch on a continuous cycle index, BGR→RGBA and PCM→AudioFrame conversion, barge-in via `clear_buffer`.

**Tune on a real run (marked in the worker):**
- `MUSETALK_WINDOW_SEC` (0.6s) — audio batched before inference; lower = less latency, more overhead.
- fps pacing vs. `AVSynchronizer` — if you see A/V drift or a stalling track, adjust the pacing/sleep.
- Whisper context at window boundaries — if mouth quality dips at chunk seams, add a small audio overlap.

> This worker is a **scaffold**: the LiveKit wiring matches the current API and the MuseTalk calls match
> `musetalk_algo.py`, but it hasn't been run on a GPU+LiveKit box yet — expect to nudge the TUNE constants.
