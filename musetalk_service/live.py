"""
Live mic mode — realtime avatar over ONE WebSocket on the same TCP port 8000.

Browser plays the persona's idle loop natively (GET /personas/{id}/idle.mp4) and sends
16 kHz mono Int16 PCM; the server slices it into MUSETALK_WINDOW_SEC windows, gates them
with streaming Silero VAD (the repo's own model; RMS-energy fallback), and ONLY during
speech runs whisper + batched UNet/VAE and streams frames back (4-byte cycle-index prefix
+ JPEG). Silence costs zero GPU and zero bandwidth.

Stability & continuity:
- Each window is extracted WITH the tail of the previous window prepended
  (MUSETALK_CONTEXT_SEC) so whisper sees real audio context at the boundary instead of
  padding — this is what stops the mouth "vibrating" between windows. The context frames
  are dropped after extraction (no extra UNet cost).
- One hangover window after speech renders the mouth closing naturally.
- The client reports its idle-loop position; when a speech segment starts, the cycle
  counter syncs to it so the generated frames continue from (about) where the loop was,
  and the frame-index prefix lets the client seek the loop back to where speech ended.
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
# Audio context prepended to each window before whisper extraction (mouth stability).
CONTEXT_SEC = float(os.environ.get("MUSETALK_CONTEXT_SEC", "0.3"))
# Speech gates: Silero VAD probability if the model loads, else peak-100ms-RMS energy.
VAD_THRESHOLD = float(os.environ.get("MUSETALK_VAD_THRESHOLD", "0.5"))
SILENCE_RMS = float(os.environ.get("MUSETALK_SILENCE_RMS", "0.01"))
# How much gets rendered AFTER the last detected speech (mouth-close tail). Generation
# stops right there instead of playing out the window's silent remainder — this is what
# keeps the mouth from stuttering for seconds after you stop talking.
TAIL_SEC = float(os.environ.get("MUSETALK_TAIL_SEC", "0.25"))


class StreamVad:
    """Streaming Silero VAD on 512-sample chunks, reusing the ONNX model that ships
    with OpenAvatarChat's own VAD handler. State persists across windows, so this is
    true streaming VAD over the continuous mic signal."""

    CHUNK = 512  # 32 ms @ 16 kHz — what the silero model expects

    def __init__(self, sr: int):
        self.sr = sr
        self.session = None
        self._rest = np.zeros(0, dtype=np.float32)
        try:
            import onnxruntime
            path = os.path.join(engine.OAC_ROOT, "src", "handlers", "vad", "silerovad",
                                "silero_vad", "src", "silero_vad", "data", "silero_vad.onnx")
            opts = onnxruntime.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.log_severity_level = 4
            self.session = onnxruntime.InferenceSession(
                path, providers=["CPUExecutionProvider"], sess_options=opts)
            self._state = np.zeros((2, 1, 128), dtype=np.float32)
            self._sr_arr = np.array([sr], dtype=np.int64)
            logger.info(f"live VAD: silero loaded ({path})")
        except Exception as e:
            logger.warning(f"live VAD: silero unavailable ({e}) — falling back to RMS energy gate")

    @property
    def available(self) -> bool:
        return self.session is not None

    def analyze(self, seg: np.ndarray, gate: float) -> tuple[float, int]:
        """Run streaming VAD over the segment's 512-sample chunks.
        Returns (peak probability, end sample of the LAST speech chunk, seg-relative;
        -1 if no chunk crossed the gate)."""
        rest_len = len(self._rest)
        buf = np.concatenate([self._rest, seg]) if rest_len else seg
        n = len(buf) // self.CHUNK
        peak, last_end = 0.0, -1
        for i in range(n):
            clip = buf[i * self.CHUNK:(i + 1) * self.CHUNK][np.newaxis, :]
            prob, self._state = self.session.run(
                None, {"input": clip, "sr": self._sr_arr, "state": self._state})
            p = float(prob[0][0])
            peak = max(peak, p)
            if p >= gate:
                last_end = (i + 1) * self.CHUNK - rest_len
        self._rest = buf[n * self.CHUNK:]
        return peak, max(last_end, -1)


class LiveSession:
    """Per-connection state. Audio in (any thread) -> generate_window() (worker
    thread, blocking GPU) -> `pending` (frame, cycle_idx) consumed by the sender loop."""

    def __init__(self, algo):
        self.algo = algo
        self.sr = engine.ALGO_SR
        self.fps = engine.FPS
        self.window_samples = int(WINDOW_SEC * self.sr)
        self.context_samples = int(CONTEXT_SEC * self.sr)
        self._audio = np.zeros(0, dtype=np.float32)
        self._audio_lock = threading.Lock()
        self._clock_lock = threading.Lock()
        self._counter = 0  # cycle position of generated frames
        self.pending: deque = deque()  # (BGR frame, cycle idx) ready to send
        self.stopped = False
        self.dropped_sec = 0.0   # audio discarded to keep the avatar near-realtime
        self.silent_windows = 0  # windows skipped by the speech gate
        self.vad = StreamVad(self.sr)
        self.gate = VAD_THRESHOLD if self.vad.available else SILENCE_RMS  # UI-adjustable
        self.level = 0.0         # last window's VAD prob (or RMS) — shown in the UI
        self._tail = np.zeros(0, dtype=np.float32)  # previous window's tail (whisper context)
        self._hangover = 0       # silent windows still rendered after speech (mouth closes)
        self._speaking = False   # inside a speech segment?
        self._client_idx: int | None = None  # client's reported idle-loop position
        self._client_idx_at = 0.0
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

    def report_client_idx(self, idx: int, now: float):
        self._client_idx = int(idx)
        self._client_idx_at = now

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
    def _reserve_indices(self, n: int, now: float) -> int:
        with self._clock_lock:
            if not self._speaking and self._client_idx is not None:
                # New speech segment: continue from (about) where the client's idle
                # loop is NOW + the lag until these frames actually show (~1 window).
                elapsed = max(0.0, now - self._client_idx_at)
                self._counter = self._client_idx + int(self.fps * (elapsed + WINDOW_SEC))
            start = self._counter
            self._counter += n
            return start

    def _measure(self, seg: np.ndarray) -> tuple[bool, int]:
        """Speech gate. Returns (is_speech, end sample of last speech in the window).
        Silero VAD prob (streaming) when available, else peak-100ms RMS."""
        if self.vad.available:
            peak, last_end = self.vad.analyze(seg, self.gate)
            self.level = round(peak, 3)
        else:
            step = max(1, int(0.1 * self.sr))
            peak, last_end = 0.0, -1
            for i in range(0, len(seg), step):
                c = seg[i:i + step]
                if len(c):
                    r = float(np.sqrt(np.mean(c * c)))
                    peak = max(peak, r)
                    if r >= self.gate:
                        last_end = i + len(c)
            self.level = round(peak, 4)
        if self.gate <= 0:
            return True, len(seg)
        return self.level >= self.gate, last_end

    # ---- generation (blocking; run via asyncio.to_thread) ----
    def generate_window(self, now: float) -> int:
        """One audio window -> gate -> whisper (with context) -> batched frames.
        Returns frames generated (0 = no full window, or silence -> client shows its loop)."""
        seg = self._take_window()
        if seg is None:
            return 0
        context = self._tail
        self._tail = seg[-self.context_samples:] if self.context_samples > 0 else self._tail

        tail_frames = max(1, int(round(TAIL_SEC * self.fps)))
        speech, last_end = self._measure(seg)
        keep = None  # None = the whole window (mid-utterance)
        if speech:
            self._hangover = 1
            # Speech ended INSIDE this window: render up to the last speech + a short
            # mouth-close tail, not the window's silent remainder (post-speech stutter).
            if 0 <= last_end < len(seg) - int(TAIL_SEC * self.sr):
                keep = int(np.ceil(last_end / self.sr * self.fps)) + tail_frames
                self._hangover = 0  # the close is already rendered
        elif self._hangover > 0:
            self._hangover -= 1
            keep = tail_frames  # speech ended at the window edge: short close, not a full window
        else:
            self._speaking = False
            self.silent_windows += 1
            return 0

        # Whisper sees context + window; the context's frames are dropped afterwards, so
        # the kept frames get real boundary context (stable mouth) at no extra UNet cost.
        full = np.concatenate([context, seg]) if len(context) else seg
        chunks = self.algo.extract_whisper_feature(full, self.sr)
        # keep exactly one window's worth of frames; everything before is context
        frames_per_window = int(round(self.window_samples / self.sr * self.fps))
        n_drop = max(len(chunks) - frames_per_window, 0)
        chunks = chunks[n_drop:]
        if keep is not None:
            chunks = chunks[:max(keep, 1)]
        n = len(chunks)
        if n == 0:
            return 0
        start = self._reserve_indices(n, now)
        self._speaking = True
        batch = engine.BATCH
        for i in range(0, n, batch):
            wb = chunks[i:i + batch]
            for recon, idx in self.algo.generate_frames(wb, start + i, len(wb)):
                self.pending.append((self.algo.res2combined(recon, idx), idx))
        if keep is not None and self._hangover == 0:
            self._speaking = False  # segment closed — next utterance re-syncs to the client loop
        return n
