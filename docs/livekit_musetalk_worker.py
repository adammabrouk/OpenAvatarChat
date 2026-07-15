"""
MuseTalk × LiveKit — standalone avatar-worker microservice.

WHAT THIS IS
  A LiveKit *avatar worker* (a separate room participant) that:
    • publishes a continuous video track that LOOPS the prepared face video when idle, and
    • lip-syncs to the voice agent's TTS audio when the agent sends audio, and
    • re-publishes that audio so the room hears the agent's voice in sync with the lips.

  It plugs into the LiveKit Agents avatar framework:
    voice agent ──(TTS audio over DataStream)──▶ this worker ──(video+audio tracks)──▶ room ──▶ client

  Built on:
    - livekit.agents.voice.avatar : AvatarRunner, AvatarOptions, DataStreamAudioReceiver, VideoGenerator
    - OpenAvatarChat MuseTalk per-frame engine (MuseTalkAlgoV15): generate_idle_frame / extract_whisper_feature / generate_frame

  Run it INSIDE the OpenAvatarChat repo/env (it has the MuseTalk deps + the models via
  `download_models.py --handler musetalk`). The avatar must be prepared once (see the guide).

STATUS: scaffold to iterate on the GPU box — the LiveKit wiring is complete and matches the
  current API; the audio-windowing / fps pacing constants are marked TUNE and want a real run.
"""

from __future__ import annotations

import os
import sys
import asyncio
import logging
import time
from collections import deque
from typing import AsyncIterator, Optional

import numpy as np
import cv2
import librosa

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice.avatar import (
    AvatarRunner,
    AvatarOptions,
    DataStreamAudioReceiver,
    VideoGenerator,
    AudioSegmentEnd,
)

# --- Make the OpenAvatarChat MuseTalk engine importable ---------------------
OAC_ROOT = os.environ.get("OAC_ROOT", "/root/open-avatar-chat")
sys.path.insert(0, os.path.join(OAC_ROOT, "src"))
sys.path.insert(0, os.path.join(OAC_ROOT, "src", "handlers", "avatar", "musetalk", "MuseTalk"))
os.chdir(OAC_ROOT)  # MuseTalk resolves some model paths relative to CWD

from handlers.avatar.musetalk.musetalk_algo import MuseTalkAlgoV15  # noqa: E402

logger = logging.getLogger("musetalk-avatar")

# --- Config (env-overridable) -----------------------------------------------
AVATAR_IDENTITY = os.environ.get("AVATAR_IDENTITY", "avatar_worker")
AVATAR_ID = os.environ.get("AVATAR_ID", "presenter_1")          # a PREPARED avatar id
AVATAR_VIDEO = os.environ.get("AVATAR_VIDEO", "")               # only needed to (re)prepare
FPS = int(os.environ.get("MUSETALK_FPS", "25"))
BATCH = int(os.environ.get("MUSETALK_BATCH", "4"))
OUT_SR = int(os.environ.get("AVATAR_AUDIO_SR", "24000"))        # room audio sample rate (must be divisible by FPS)
ALGO_SR = 16000                                                # whisper feature sample rate (fixed by MuseTalk)
assert OUT_SR % FPS == 0, "AVATAR_AUDIO_SR must be divisible by MUSETALK_FPS"
SAMPLES_PER_FRAME = OUT_SR // FPS                               # audio samples emitted per video frame
WINDOW_SEC = float(os.environ.get("MUSETALK_WINDOW_SEC", "0.6"))  # TUNE: batch this much audio before inferring


def _bgr_to_video_frame(bgr: np.ndarray) -> rtc.VideoFrame:
    """MuseTalk yields HxWx3 BGR uint8 -> LiveKit RGBA VideoFrame."""
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    h, w = rgba.shape[:2]
    return rtc.VideoFrame(width=w, height=h, type=rtc.VideoBufferType.RGBA, data=rgba.tobytes())


def _pcm_to_audio_frame(pcm_i16: np.ndarray) -> rtc.AudioFrame:
    """mono int16 samples -> LiveKit AudioFrame at OUT_SR."""
    return rtc.AudioFrame(
        data=pcm_i16.tobytes(),
        sample_rate=OUT_SR,
        num_channels=1,
        samples_per_channel=len(pcm_i16),
    )


class MuseTalkVideoGenerator(VideoGenerator):
    """Idle-loops the prepared face; lip-syncs to pushed audio.

    Contract (livekit.agents.voice.avatar.VideoGenerator):
      push_audio(frame)  <- agent TTS audio (rtc.AudioFrame) or AudioSegmentEnd
      clear_buffer()     <- barge-in / interruption: drop pending speech, return to idle
      __aiter__()        -> continuously yields VideoFrame / AudioFrame / AudioSegmentEnd
    """

    def __init__(self, algo: MuseTalkAlgoV15):
        self._algo = algo
        self._idx = 0                                   # continuous cycle index (idle & speak share it)
        # inbound TTS audio, resampled to OUT_SR mono float32
        self._in_buf = np.zeros(0, dtype=np.float32)
        self._in_lock = asyncio.Lock()
        # produced speaking output: deque of ("frame", bgr, pcm_i16) or ("segment_end",)
        self._out: deque = deque()
        self._out_evt = asyncio.Event()
        self._closed = False

    # ---- inbound audio from the agent -------------------------------------
    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, AudioSegmentEnd):
            await self._flush_segment(force=True)
            self._out.append(("segment_end",))
            self._out_evt.set()
            return

        # rtc.AudioFrame -> mono float32 @ OUT_SR
        pcm = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        if frame.num_channels > 1:
            pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1)
        if frame.sample_rate != OUT_SR:
            pcm = librosa.resample(pcm, orig_sr=frame.sample_rate, target_sr=OUT_SR)
        async with self._in_lock:
            self._in_buf = np.concatenate([self._in_buf, pcm])
        await self._flush_segment(force=False)

    async def _flush_segment(self, force: bool) -> None:
        """Consume whole WINDOW_SEC chunks (snapped to whole frames) and run MuseTalk."""
        win = int(WINDOW_SEC * OUT_SR)
        while True:
            async with self._in_lock:
                have = len(self._in_buf)
                take = have if force else (win if have >= win else 0)
                take -= take % SAMPLES_PER_FRAME          # whole video frames only
                if take <= 0:
                    return
                seg = self._in_buf[:take]
                self._in_buf = self._in_buf[take:]
            await asyncio.to_thread(self._infer_segment, seg)  # GPU work off the event loop
            self._out_evt.set()
            if not force:
                continue
            return

    def _infer_segment(self, seg_out_sr: np.ndarray) -> None:
        """seg_out_sr: mono float32 @ OUT_SR -> produce (video, audio) pairs into _out."""
        n_frames = len(seg_out_sr) // SAMPLES_PER_FRAME
        if n_frames <= 0:
            return
        seg_16k = librosa.resample(seg_out_sr, orig_sr=OUT_SR, target_sr=ALGO_SR)
        whisper_chunks = self._algo.extract_whisper_feature(seg_16k, ALGO_SR)  # [T,50,384]
        whisper_chunks = whisper_chunks[:n_frames]
        pcm_i16 = (np.clip(seg_out_sr, -1, 1) * 32767).astype(np.int16)
        for i in range(min(n_frames, len(whisper_chunks))):
            bgr = self._algo.generate_frame(whisper_chunks[i:i + 1], self._idx)  # inpainted mouth on cycle[idx]
            audio_slice = pcm_i16[i * SAMPLES_PER_FRAME:(i + 1) * SAMPLES_PER_FRAME]
            self._out.append(("frame", bgr, audio_slice))
            self._idx += 1

    # ---- barge-in ---------------------------------------------------------
    def clear_buffer(self) -> None:
        self._in_buf = np.zeros(0, dtype=np.float32)
        self._out.clear()
        # keep _idx advancing (do NOT reset) so the loop stays continuous

    # ---- outbound stream: constant-fps video, idle when no speech ---------
    async def __aiter__(self) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        frame_period = 1.0 / FPS
        next_t = time.monotonic()
        while not self._closed:
            if self._out:
                item = self._out.popleft()
                if item[0] == "segment_end":
                    yield AudioSegmentEnd()
                    continue
                _, bgr, audio_slice = item
                yield _bgr_to_video_frame(bgr)          # speaking video
                yield _pcm_to_audio_frame(audio_slice)  # + its aligned audio
            else:
                # idle: raw looped frame, NO gpu, NO audio (silence on the track)
                bgr = self._algo.generate_idle_frame(self._idx)
                self._idx += 1
                yield _bgr_to_video_frame(bgr)

            # pace to FPS (AVSynchronizer also syncs; this keeps the idle loop steady)
            next_t += frame_period
            sleep = next_t - time.monotonic()
            if sleep > 0:
                await asyncio.sleep(sleep)
            else:
                next_t = time.monotonic()


# --- MuseTalk engine bootstrap (once per worker process) --------------------
def build_algo() -> MuseTalkAlgoV15:
    algo = MuseTalkAlgoV15(
        avatar_id=AVATAR_ID,
        video_path=AVATAR_VIDEO,            # only used if force_preparation=True
        bbox_shift=0,
        batch_size=BATCH,
        force_preparation=bool(AVATAR_VIDEO),  # prepare if a video path was given, else load cache
        fps=FPS,
        vae_type="sd-vae",
    )
    algo.init()                              # loads models + prepared avatar (masks/latents/frames)
    return algo


# --- LiveKit worker entrypoint ---------------------------------------------
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    logger.info("connected to room %s as %s", ctx.room.name, AVATAR_IDENTITY)

    algo = await asyncio.to_thread(build_algo)
    first = algo.generate_idle_frame(0)
    h, w = first.shape[:2]

    options = AvatarOptions(
        video_width=w, video_height=h, video_fps=FPS,
        audio_sample_rate=OUT_SR, audio_channels=1,
    )
    audio_recv = DataStreamAudioReceiver(ctx.room)   # receives the agent's TTS over DataStream
    video_gen = MuseTalkVideoGenerator(algo)
    runner = AvatarRunner(ctx.room, audio_recv=audio_recv, video_gen=video_gen, options=options)

    await runner.start()
    logger.info("AvatarRunner started; idling until the agent speaks.")
    await runner.wait_for_complete()


if __name__ == "__main__":
    # This worker is dispatched into the SAME room as the voice agent (see the guide / agent example).
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="musetalk-avatar"))
