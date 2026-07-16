"""
Logging + performance profiling helpers for the MuseTalk service.

- One loguru sink for the whole service (the MuseTalk algo already logs via loguru,
  so `docker logs` shows a single coherent stream). Level via MUSETALK_LOG_LEVEL.
- StageTimer: accumulating per-stage wall-clock timer -> STAGE SUMMARY log + dict.
- GpuSampler: background NVML sampler (util/mem/power) around an operation.
"""
from __future__ import annotations

import os
import sys
import json
import time
import threading
from contextlib import contextmanager

from loguru import logger

LOG_LEVEL = os.environ.get("MUSETALK_LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL)

GPU_SAMPLE_SEC = float(os.environ.get("MUSETALK_GPU_SAMPLE_SEC", "1.0"))


class StageTimer:
    """Accumulating stage timer. Stages are re-entrant: entering the same stage
    inside a loop sums time across iterations and counts items for per-item averages.

    with cuda_sync=True, torch.cuda.synchronize() runs at stage boundaries so async
    GPU work is billed to the stage that launched it (costs a little, measures truly).
    """

    def __init__(self, op: str, ref: str = "", cuda_sync: bool = False):
        self.op = op
        self.ref = ref
        self.tag = f"{op}[{ref}]" if ref else op
        self.cuda_sync = cuda_sync
        self.t0 = time.perf_counter()
        self._stages: dict[str, dict] = {}  # name -> {sec, count}
        self._order: list[str] = []

    def _sync(self):
        if not self.cuda_sync:
            return
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            self.cuda_sync = False  # don't retry every stage

    @contextmanager
    def stage(self, name: str, items: int = 0):
        self._sync()
        t = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            dt = time.perf_counter() - t
            s = self._stages.get(name)
            if s is None:
                s = self._stages[name] = {"sec": 0.0, "count": 0}
                self._order.append(name)
            s["sec"] += dt
            s["count"] += items

    def finish(self, audio_sec: float | None = None, **extra) -> dict:
        """Log the STAGE SUMMARY block and return the machine-readable profile dict."""
        total = time.perf_counter() - self.t0
        stages_out: dict = {}
        lines = []
        for name in self._order:
            s = self._stages[name]
            if s["count"] > 1:
                per_ms = s["sec"] / s["count"] * 1000.0
                stages_out[name] = {"sec": round(s["sec"], 3), "count": s["count"],
                                    "per_item_ms": round(per_ms, 1)}
                lines.append(f"  {name:<16}{s['sec']:7.2f}s  ({s['count']} items, {per_ms:.1f}ms/item)")
            else:
                stages_out[name] = round(s["sec"], 3)
                lines.append(f"  {name:<16}{s['sec']:7.2f}s")
        profile = {"op": self.op, "ref": self.ref, "stages": stages_out,
                   "total_sec": round(total, 3)}
        summary = f"  TOTAL           {total:7.2f}s"
        if audio_sec:
            rtf = total / audio_sec
            profile["audio_sec"] = round(audio_sec, 3)
            profile["realtime_factor"] = round(rtf, 3)
            summary += f"  | audio {audio_sec:.2f}s | realtime_factor {rtf:.2f}x"
            frames = extra.get("frames")
            if frames:
                profile["fps"] = round(frames / total, 2)
                summary += f" | effective {frames / total:.1f} fps"
        profile.update(extra)
        logger.info(f"{self.tag} STAGE SUMMARY")
        for line in lines:
            logger.info(line)
        logger.info(summary)
        gpu = extra.get("gpu")
        if gpu:
            logger.info(
                f"  GPU util avg/max {gpu.get('util_avg', '?')}%/{gpu.get('util_max', '?')}% "
                f"| mem peak {gpu.get('mem_peak_mb', '?')}/{gpu.get('mem_total_mb', '?')} MB "
                f"| power avg {gpu.get('power_avg_w', '?')}W"
            )
        return profile


def write_profile(path: str, profile: dict) -> None:
    try:
        with open(path, "w") as f:
            json.dump(profile, f, indent=2)
        logger.debug(f"profile written: {path}")
    except Exception as e:
        logger.warning(f"could not write profile {path}: {e}")


# ---------------------------------------------------------------- GPU (NVML) ---

_nvml_lock = threading.Lock()
_nvml_state = {"tried": False, "handle": None, "pynvml": None}


def _nvml_handle():
    """Lazy NVML init; returns (pynvml, handle) or (None, None). Never raises."""
    with _nvml_lock:
        if not _nvml_state["tried"]:
            _nvml_state["tried"] = True
            try:
                import pynvml
                pynvml.nvmlInit()
                _nvml_state["pynvml"] = pynvml
                _nvml_state["handle"] = pynvml.nvmlDeviceGetHandleByIndex(
                    int(os.environ.get("MUSETALK_GPU_INDEX", "0")))
            except Exception as e:
                logger.warning(f"GPU profiling disabled (NVML unavailable: {e}) — "
                               f"install nvidia-ml-py and run with GPU access to enable")
        return _nvml_state["pynvml"], _nvml_state["handle"]


def gpu_snapshot() -> dict:
    """One-shot GPU reading (for GET /gpu). Empty dict if NVML is unavailable."""
    pynvml, h = _nvml_handle()
    if h is None:
        return {}
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        snap = {
            "name": pynvml.nvmlDeviceGetName(h),
            "util_pct": util.gpu,
            "mem_util_pct": util.memory,
            "mem_used_mb": mem.used // (1024 * 1024),
            "mem_total_mb": mem.total // (1024 * 1024),
        }
        try:
            snap["power_w"] = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
            snap["temp_c"] = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            pass
        return snap
    except Exception as e:
        return {"error": str(e)}


class GpuSampler:
    """Samples GPU util/mem/power every GPU_SAMPLE_SEC while an operation runs.
    start() -> ... -> stop() returns aggregated stats (or {} if NVML unavailable)."""

    def __init__(self, interval: float | None = None):
        self.interval = GPU_SAMPLE_SEC if interval is None else interval
        self._samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self):
        while not self._stop.is_set():
            snap = gpu_snapshot()
            if snap and "error" not in snap:
                self._samples.append(snap)
            self._stop.wait(self.interval)

    def start(self):
        if self.interval <= 0 or _nvml_handle()[1] is None:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True, name="gpu-sampler")
        self._thread.start()
        return self

    def stop(self) -> dict:
        if self._thread is None:
            return {}
        self._stop.set()
        self._thread.join(timeout=2 * self.interval + 1)
        # final sample so short ops still get at least one reading
        snap = gpu_snapshot()
        if snap and "error" not in snap:
            self._samples.append(snap)
        if not self._samples:
            return {}
        utils = [s["util_pct"] for s in self._samples]
        mems = [s["mem_used_mb"] for s in self._samples]
        agg = {
            "util_avg": round(sum(utils) / len(utils)),
            "util_max": max(utils),
            "mem_peak_mb": max(mems),
            "mem_total_mb": self._samples[-1]["mem_total_mb"],
            "samples": len(self._samples),
        }
        powers = [s["power_w"] for s in self._samples if "power_w" in s]
        if powers:
            agg["power_avg_w"] = round(sum(powers) / len(powers), 1)
        return agg
