#!/usr/bin/env python3
"""server.py - ONNX Runtime inference server (one model, one process, GPU only).

Started by supervisord next to guard.py. Loads the model at MODEL_PATH with the
CUDA execution provider and serves it over HTTP:

  GET  /health   liveness, provider in use, model inputs
  POST /infer    {"inputs": {"<input name>": <nested list>, ...}}
                 -> {"outputs": {"<output name>": <nested list>, ...}}

Environment:
  MODEL_PATH        model file                       (default /models/model.onnx)
  PORT              listen port                      (default 8000)
  GPU_MEM_LIMIT_MB  cap on ONNX Runtime's GPU arena  (default: no cap)
                    Keep it below the MPS memory limit of the container: the
                    CUDA context and the CUDA/cuDNN libraries also use memory.
"""
import os, sys

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.onnx")
PORT = int(os.environ.get("PORT", "8000"))
GPU_MEM_LIMIT_MB = int(os.environ.get("GPU_MEM_LIMIT_MB", "0"))

DTYPES = {"tensor(float)": np.float32, "tensor(double)": np.float64, "tensor(float16)": np.float16,
          "tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(uint8)": np.uint8,
          "tensor(bool)": np.bool_}


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


session = load_session()
inputs = {i.name: i for i in session.get_inputs()}
output_names = [o.name for o in session.get_outputs()]
app = FastAPI(title="ONNX inference server")


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_PATH, "providers": session.get_providers(),
            "inputs": {n: {"shape": i.shape, "type": i.type} for n, i in inputs.items()}}


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
    try:
        results = session.run(output_names, feed)
    except Exception as e:                     # ONNX Runtime raises its own exception types
        text = str(e)
        print(f"inference failed: {text}", file=sys.stderr, flush=True)
        if "out of memory" in text.lower() or "failed to allocate" in text.lower():
            return JSONResponse({"error": "gpu_oom", "retry": True}, status_code=503,
                                headers={"Retry-After": "1"})
        return JSONResponse({"error": text}, status_code=500)
    return {"outputs": {n: r.tolist() for n, r in zip(output_names, results)}}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
