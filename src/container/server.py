#!/usr/bin/env python3
"""server.py - ONNX Runtime inference server (one model, one process, GPU only).

Started by supervisord next to guard.py. Loads the model at MODEL_PATH with the
CUDA execution provider and serves it over HTTP:

  GET  /health   liveness, provider in use, model inputs
  GET  /metrics  the container's own GPU figures (SM limit, memory, utilisation)
  POST /infer    {"inputs": {"<input name>": <nested list>, ...}}
                 -> {"outputs": {"<output name>": <nested list>, ...},
                     "metrics": {GPU time, wall time, utilisation of this inference}}

The GPU figures come from libvitroobs.so (vitro_observer.c), which supervisord
preloads into this process. The server only marks where each inference starts
and ends. Inferences are not serialised: several may run at once, and the
library attributes kernels to an inference by the thread that launched them.
Without the library the server works the same and reports no metrics.

Environment:
  MODEL_PATH        model file                       (default /models/model.onnx)
  PORT              listen port                      (default 8000)
  GPU_MEM_LIMIT_MB  cap on ONNX Runtime's GPU arena  (default: no cap)
                    Keep it below the MPS memory limit of the container: the
                    CUDA context and the CUDA/cuDNN libraries also use memory.
"""
import ctypes, os, sys

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import vitro_stats

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.onnx")
PORT = int(os.environ.get("PORT", "8000"))
GPU_MEM_LIMIT_MB = int(os.environ.get("GPU_MEM_LIMIT_MB", "0"))

DTYPES = {"tensor(float)": np.float32, "tensor(double)": np.float64, "tensor(float16)": np.float16,
          "tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(uint8)": np.uint8,
          "tensor(bool)": np.bool_}


class Inference(ctypes.Structure):
    """Mirrors `struct vitro_inference` in vitro_observer.c."""
    _fields_ = [("gpu_ms", ctypes.c_double), ("wall_ms", ctypes.c_double),
                ("smw_ms", ctypes.c_double), ("kernels", ctypes.c_uint64),
                ("unresolved", ctypes.c_uint64)]


def load_observer():
    """The preloaded library's two entry points, or None when it is not loaded."""
    try:
        lib = ctypes.CDLL(None)                # symbols of the process, incl. LD_PRELOAD
        return lib.vitro_inference_begin, lib.vitro_inference_end
    except (OSError, AttributeError):
        return None


def load_session():
    ort.preload_dlls()                         # CUDA / cuDNN from the nvidia-* pip packages
    cuda = {"arena_extend_strategy": "kSameAsRequested"}   # grow by what is needed, not by x2
    if GPU_MEM_LIMIT_MB:
        cuda["gpu_mem_limit"] = GPU_MEM_LIMIT_MB * 2**20
    sess = ort.InferenceSession(MODEL_PATH, providers=[("CUDAExecutionProvider", cuda)])
    if sess.get_providers()[0] != "CUDAExecutionProvider":
        # ONNX Runtime falls back to the CPU without an error; here that is a failure.
        sys.exit(f"CUDA execution provider not available (got {sess.get_providers()})")
    return sess


observer = load_observer()
session = load_session()
inputs = {i.name: i for i in session.get_inputs()}
output_names = [o.name for o in session.get_outputs()]
app = FastAPI(title="ONNX inference server")


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_PATH, "providers": session.get_providers(),
            "observer": observer is not None,
            "inputs": {n: {"shape": i.shape, "type": i.type} for n, i in inputs.items()}}


@app.get("/metrics")
def metrics():
    stats = vitro_stats.read()
    if stats is None:
        return JSONResponse({"error": "no observer stats"}, status_code=503)
    return stats


def inference_metrics(m):
    """Per-inference figures. util is time-based; smw_util is an estimate."""
    wall = m.wall_ms or 1e-9
    return {"gpu_ms": round(m.gpu_ms, 3), "wall_ms": round(m.wall_ms, 3),
            "util_pct": round(100 * m.gpu_ms / wall, 1),
            "smw_util_pct": round(100 * m.smw_ms / wall, 1),
            "kernels": m.kernels, "unresolved": m.unresolved}


@app.post("/infer")
def infer(payload: dict):
    given = payload.get("inputs", {})
    missing = [n for n in inputs if n not in given]
    if missing:
        return JSONResponse({"error": f"missing inputs: {missing}"}, status_code=422)
    try:
        feed = {n: np.asarray(given[n], dtype=DTYPES.get(i.type)) for n, i in inputs.items()}
    except (TypeError, ValueError) as e:
        return JSONResponse({"error": f"bad input: {e}"}, status_code=422)
    measured = Inference()
    if observer:
        observer[0]()                          # inference starts on this thread
    try:
        results = session.run(output_names, feed)
    except Exception as e:                     # ONNX Runtime raises its own exception types
        if observer:
            observer[1](ctypes.byref(measured))
        text = str(e)
        print(f"inference failed: {text}", file=sys.stderr, flush=True)
        if "out of memory" in text.lower() or "failed to allocate" in text.lower():
            return JSONResponse({"error": "gpu_oom", "retry": True}, status_code=503,
                                headers={"Retry-After": "1"})
        return JSONResponse({"error": text}, status_code=500)
    if observer:
        observer[1](ctypes.byref(measured))    # waits for this inference's kernels
    out = {"outputs": {n: r.tolist() for n, r in zip(output_names, results)}}
    if observer:
        out["metrics"] = inference_metrics(measured)
    return out


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
