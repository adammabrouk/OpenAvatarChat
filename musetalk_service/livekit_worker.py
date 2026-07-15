"""
LiveKit avatar worker — the plugin-style integration, built on the SAME `engine`/persona store
the REST API/UI uses. A persona you set up in the UI is directly usable here (AVATAR_ID).

Voice agent (STT→LLM→TTS) --TTS audio over DataStream--> this worker --video+audio--> LiveKit room.
Idle-loops the persona when silent; lip-syncs when the agent speaks.

Env: LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET, OAC_ROOT, AVATAR_ID (a prepared persona).
"""
from __future__ import annotations

import os
import asyncio
import logging
import time
from collections import deque
from typing import AsyncIterator

import numpy as np
import cv2
import librosa

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice.avatar import (
    AvatarRunner, AvatarOptions, DataStreamAudioReceiver, VideoGenerator, AudioSegmentEnd,
)

import engine  # shared MuseTalk engine + persona store

logger = logging.getLogger("musetalk-avatar")

AVATAR_IDENTITY = os.environ.get("AVATAR_IDENTITY", "avatar_worker")
AVATAR_ID = os.environ.get("AVATAR_ID", "presenter_1")   # a persona prepared via the UI/API
FPS = engine.FPS
OUT_SR = int(os.environ.get("AVATAR_AUDIO_SR", "24000"))
assert OUT_SR % FPS == 0, "AVATAR_AUDIO_SR must be divisible by MUSETALK_FPS"
SPF = OUT_SR // FPS                                        # audio samples per video frame
WINDOW_SEC = float(os.environ.get("MUSETALK_WINDOW_SEC", "0.6"))  # TUNE


def _bgr_to_vf(bgr: np.ndarray) -> rtc.VideoFrame:
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    h, w = rgba.shape[:2]
    return rtc.VideoFrame(width=w, height=h, type=rtc.VideoBufferType.RGBA, data=rgba.tobytes())


def _pcm_to_af(pcm_i16: np.ndarray) -> rtc.AudioFrame:
    return rtc.AudioFrame(data=pcm_i16.tobytes(), sample_rate=OUT_SR, num_channels=1,
                          samples_per_channel=len(pcm_i16))


class MuseTalkVideoGenerator(VideoGenerator):
    def __init__(self, algo):
        self._algo = algo
        self._idx = 0
        self._in = np.zeros(0, dtype=np.float32)
        self._lock = asyncio.Lock()
        self._out: deque = deque()
        self._closed = False

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, AudioSegmentEnd):
            await self._flush(force=True)
            self._out.append(("end",))
            return
        pcm = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        if frame.num_channels > 1:
            pcm = pcm.reshape(-1, frame.num_channels).mean(axis=1)
        if frame.sample_rate != OUT_SR:
            pcm = librosa.resample(pcm, orig_sr=frame.sample_rate, target_sr=OUT_SR)
        async with self._lock:
            self._in = np.concatenate([self._in, pcm])
        await self._flush(force=False)

    async def _flush(self, force: bool) -> None:
        win = int(WINDOW_SEC * OUT_SR)
        while True:
            async with self._lock:
                have = len(self._in)
                take = have if force else (win if have >= win else 0)
                take -= take % SPF
                if take <= 0:
                    return
                seg = self._in[:take]
                self._in = self._in[take:]
            await asyncio.to_thread(self._infer, seg)
            if not force:
                continue
            return

    def _infer(self, seg: np.ndarray) -> None:
        n = len(seg) // SPF
        if n <= 0:
            return
        seg16 = librosa.resample(seg, orig_sr=OUT_SR, target_sr=engine.ALGO_SR)
        chunks = self._algo.extract_whisper_feature(seg16, engine.ALGO_SR)[:n]
        pcm_i16 = (np.clip(seg, -1, 1) * 32767).astype(np.int16)
        for i in range(min(n, len(chunks))):
            bgr = self._algo.generate_frame(chunks[i:i + 1], self._idx)
            self._out.append(("frame", bgr, pcm_i16[i * SPF:(i + 1) * SPF]))
            self._idx += 1

    def clear_buffer(self) -> None:
        self._in = np.zeros(0, dtype=np.float32)
        self._out.clear()  # keep _idx advancing -> continuous loop

    async def __aiter__(self) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        period = 1.0 / FPS
        nxt = time.monotonic()
        while not self._closed:
            if self._out:
                item = self._out.popleft()
                if item[0] == "end":
                    yield AudioSegmentEnd()
                    continue
                _, bgr, pcm = item
                yield _bgr_to_vf(bgr)
                yield _pcm_to_af(pcm)
            else:
                yield _bgr_to_vf(self._algo.generate_idle_frame(self._idx))
                self._idx += 1
            nxt += period
            s = nxt - time.monotonic()
            if s > 0:
                await asyncio.sleep(s)
            else:
                nxt = time.monotonic()


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    algo = await asyncio.to_thread(engine.load_persona, AVATAR_ID)
    h, w = algo.generate_idle_frame(0).shape[:2]
    options = AvatarOptions(video_width=w, video_height=h, video_fps=FPS,
                            audio_sample_rate=OUT_SR, audio_channels=1)
    runner = AvatarRunner(
        ctx.room,
        audio_recv=DataStreamAudioReceiver(ctx.room),
        video_gen=MuseTalkVideoGenerator(algo),
        options=options,
    )
    await runner.start()
    logger.info("MuseTalk avatar worker started for persona '%s'", AVATAR_ID)
    await runner.wait_for_complete()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name="musetalk-avatar"))
