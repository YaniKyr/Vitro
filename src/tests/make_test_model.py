#!/usr/bin/env python3
"""make_test_model.py - write a small dummy ONNX model for testing the container.

The model is LAYERS x (MatMul 1024x1024 + Relu) on a float input "x" of shape
[batch, 1024]; output "y" has the same shape. Weights are random (about 4 MiB
per layer), so it loads the GPU without needing a real model.

  make_test_model.py [out.onnx] [rows]

With `rows` (e.g. 4096) the model first repeats each input row that many times
and averages them again at the end. The output shape is unchanged, but one
small request then keeps the GPU busy for a long time; iwmr/calibrate.py needs
that to compare the measured utilisation with the driver's.

Needs the `onnx` package. Without installing it on the host (no bind mount, so
it also works with SELinux enforcing):
  mkdir -p models
  docker run --rm -i python:3.12-slim sh -c "pip install -q onnx numpy >&2 && \
    python - /tmp/m.onnx >&2 && cat /tmp/m.onnx" < tests/make_test_model.py > models/model.onnx
(for the heavy variant use `python - /tmp/m.onnx 4096`)
"""
import os, sys

import numpy as np
from onnx import TensorProto, helper, numpy_helper, save

LAYERS, WIDTH = 8, 1024

out = sys.argv[1] if len(sys.argv) > 1 else "model.onnx"
rows = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rng = np.random.default_rng(0)
nodes, weights, prev = [], [], "x"
if rows:                                   # [batch, W] -> [batch * rows, W]
    weights.append(numpy_helper.from_array(np.array([rows, 1], dtype=np.int64), "repeats"))
    weights.append(numpy_helper.from_array(np.array([-1, rows, WIDTH], dtype=np.int64), "grouped"))
    nodes.append(helper.make_node("Tile", ["x", "repeats"], ["tiled"]))
    prev = "tiled"
for i in range(LAYERS):
    w = (rng.standard_normal((WIDTH, WIDTH)) / np.sqrt(WIDTH)).astype(np.float32)
    weights.append(numpy_helper.from_array(w, f"w{i}"))
    last = i == LAYERS - 1
    nodes.append(helper.make_node("MatMul", [prev, f"w{i}"], [f"m{i}"]))
    nodes.append(helper.make_node("Relu", [f"m{i}"], ["y" if last and not rows else f"r{i}"]))
    prev = f"r{i}"
if rows:                                   # back to [batch, W]: mean over each group of rows
    nodes.append(helper.make_node("Reshape", [prev, "grouped"], ["g"]))
    nodes.append(helper.make_node("ReduceMean", ["g"], ["y"], axes=[1], keepdims=0))

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
