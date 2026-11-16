#!/usr/bin/env python3
"""make_test_model.py - write a small dummy ONNX model for testing the container.

The model is LAYERS x (MatMul 1024x1024 + Relu) on a float input "x" of shape
[batch, 1024]; output "y" has the same shape. Weights are random (about 4 MiB
per layer), so it loads the GPU without needing a real model.

Needs the `onnx` package. Without installing it on the host (no bind mount, so
it also works with SELinux enforcing):
  mkdir -p models
  docker run --rm -i python:3.12-slim sh -c "pip install -q onnx numpy >&2 && \
    python - /tmp/m.onnx >&2 && cat /tmp/m.onnx" < tests/make_test_model.py > models/model.onnx
"""
import os, sys

import numpy as np
from onnx import TensorProto, helper, numpy_helper, save

LAYERS, WIDTH = 8, 1024

out = sys.argv[1] if len(sys.argv) > 1 else "model.onnx"
rng = np.random.default_rng(0)
nodes, weights, prev = [], [], "x"
for i in range(LAYERS):
    w = (rng.standard_normal((WIDTH, WIDTH)) / np.sqrt(WIDTH)).astype(np.float32)
    weights.append(numpy_helper.from_array(w, f"w{i}"))
    last = i == LAYERS - 1
    nodes.append(helper.make_node("MatMul", [prev, f"w{i}"], [f"m{i}"]))
    nodes.append(helper.make_node("Relu", [f"m{i}"], ["y" if last else f"r{i}"]))
    prev = f"r{i}"

graph = helper.make_graph(
    nodes, "dummy",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", WIDTH])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", WIDTH])],
    weights)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 9                       # readable by older ONNX Runtime releases too
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
save(model, out)
print(f"wrote {out} ({os.path.getsize(out) / 2**20:.1f} MiB)")
