"""
Live mic mode — realtime avatar over ONE WebSocket on the same TCP port 8000.

Browser sends 16 kHz mono Int16 PCM chunks; the server slices them into
MUSETALK_WINDOW_SEC windows, runs whisper + batched UNet/VAE per window, and
streams JPEG frames back at MUSETALK_FPS. When no speech is buffered the
avatar plays its idle loop (no GPU work) — one continuous cycle counter keeps
head motion seamless across idle<->speak, same design as the LiveKit worker.

Set MUSETALK_FPS to your measured generation throughput (e.g. 15 on the T4):
if the GPU can't produce frames as fast as audio arrives, backlog grows and
the avatar drifts behind your voice. The per-second stats message shows it.
"""
from __future__ import annotations

import os
import threading
from collections import deque

import numpy as np

import engine
from profiling import logger

WINDOW_SEC = float(os.environ.get("MUSETALK_WINDOW_SEC", "1.0"))
JPEG_QUALITY = int(os.environ.get("MUSETALK_JPEG_QUALITY", "80"))
# Live mode must stay near-realtime: if the GPU can't keep up, DROP the oldest
# buffered audio instead of letting the avatar drift ever further behind.
MAX_LAG_SEC = float(os.environ.get("MUSETALK_MAX_LAG_SEC", "2.0"))
# Silence gate: a window whose loudest 100 ms stays below this RMS (float32, 1.0 = full
# scale) is discarded without inference — the avatar idles instead of lip-fluttering on
# room noise. ~0.01 ≈ -40 dBFS. Raise if the mouth still moves in silence, lower if it
# misses quiet speech. 0 disables the gate.
SILENCE_RMS = float(os.environ.get("MUSETALK_SILENCE_RMS", "0.01"))


class LiveSession:
    """Per-connection state. Audio in (any thread) -> generate_window() (worker
    thread, blocking GPU) -> `pending` frames consumed by the paced sender loop."""

    def __init__(self, algo):
        self.algo = algo
        self.sr = engine.ALGO_SR
        self.fps = engine.FPS
        self.window_samples = int(WINDOW_SEC * self.sr)
        self._audio = np.zeros(0, dtype=np.float32)
        self._audio_lock = threading.Lock()
        self._clock_lock = threading.Lock()
        self._counter = 0  # ONE continuous cycle counter for idle AND speak frames
        self.pending: deque = deque()  # blended BGR frames ready to send (thread-safe ops only)
        self.stopped = False
        self.dropped_sec = 0.0  # audio discarded to keep the avatar near-realtime
        self.silent_windows = 0  # windows skipped by the silence gate
        self.mic_rms = 0.0  # peak 100ms RMS of the last window (helps tune the gate)
        self.silence_rms = SILENCE_RMS  # per-session gate — adjustable live from the UI
        self._hangover = 0  # silent windows still rendered after speech (closes the mouth naturally)
        self._max_buffer = self.window_samples + int(MAX_LAG_SEC * self.sr)

    # ---- audio in (websocket receiver) ----
    def add_audio_pcm16(self, data: bytes):
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        with self._audio_lock:
            self._audio = np.concatenate([self._audio, arr])
            overflow = len(self._audio) - self._max_buffer
            if overflow > 0:  # GPU behind — drop the OLDEST audio, keep the newest
                self._audio = self._audio[overflow:]
                self.dropped_sec += overflow / self.sr

    def audio_backlog_sec(self) -> float:
        return len(self._audio) / self.sr

    def has_window(self) -> bool:
        return len(self._audio) >= self.window_samples

    def _take_window(self):
        with self._audio_lock:
            if len(self._audio) < self.window_samples:
                return None
            seg = self._audio[:self.window_samples]
            self._audio = self._audio[self.window_samples:]
            return seg

    # ---- frame clock ----
    def _reserve_indices(self, n: int) -> int:
        with self._clock_lock:
            start = self._counter
            self._counter += n
            return start

    def next_idle_frame(self) -> np.ndarray:
        with self._clock_lock:
            idx = self._counter
            self._counter += 1
        return self.algo.generate_idle_frame(idx)  # looped source frame, no GPU

    def _is_speech(self, seg: np.ndarray) -> bool:
        """Energy gate: speech if ANY 100 ms sub-chunk exceeds the session gate.
        (Any-chunk, not average, so a word at the edge of a quiet window still passes.)"""
        step = max(1, int(0.1 * self.sr))
        peak = 0.0
        for i in range(0, len(seg), step):
            c = seg[i:i + step]
            if len(c):
                peak = max(peak, float(np.sqrt(np.mean(c * c))))
        self.mic_rms = peak
        return self.silence_rms <= 0 or peak >= self.silence_rms

    # ---- generation (blocking; run via asyncio.to_thread) ----
    def generate_window(self) -> int:
        """One audio window -> whisper -> batched frames appended to `pending`.
        Returns the number of frames generated (0 if no full window buffered,
        or if the window was silence — then the idle loop plays instead)."""
        seg = self._take_window()
        if seg is None:
            return 0
        if self._is_speech(seg):
            self._hangover = 1
        elif self._hangover > 0:
            # Speech just ended: render ONE silent window anyway — inference on
            # silence closes the mouth naturally instead of hard-cutting to the
            # source frame mid-phoneme.
            self._hangover -= 1
        else:
            self.silent_windows += 1
            return 0
        chunks = self.algo.extract_whisper_feature(seg, self.sr)
        n = len(chunks)
        if n == 0:
            return 0
        start = self._reserve_indices(n)
        batch = engine.BATCH
        for i in range(0, n, batch):
            wb = chunks[i:i + batch]
            for recon, idx in self.algo.generate_frames(wb, start + i, len(wb)):
                self.pending.append(self.algo.res2combined(recon, idx))
        return n
