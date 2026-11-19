#!/usr/bin/env python3
"""make_workload_models.py - one dummy ONNX model per workload type.

Every model takes the same request (float "x" of shape [batch, 1024]) and
returns "y" of the same shape, so the same test can drive all of them and only
the kind of GPU work differs:

  dense        8 x (MatMul 1024x1024 + Relu) on 4096 rows      cuBLAS GEMM
  conv         8 x (Conv 3x3, 64 channels + Relu) at 128x128,  cuDNN convolution
               batch 8
  attention    4 x self-attention over 4096 tokens of width 64 batched MatMul + Softmax
  elementwise  32 x (Mul + Add + Tanh) on 4096x1024 values     memory-bound kernels

  make_workload_models.py [out dir]      writes <type>.onnx for every type

Needs the `onnx` package; without installing it on the host:
  docker run --rm -i python:3.12-slim sh -c "pip install -q onnx numpy >&2 && \
    python - /tmp/m >&2 && tar -C /tmp/m -c ." < tests/make_workload_models.py | tar -C models -x
"""
import os, sys

import numpy as np
from onnx import TensorProto, helper, numpy_helper, save

W = 1024
rng = np.random.default_rng(0)


class Graph:
    def __init__(self):
        self.nodes, self.weights, self.n = [], [], 0

    def const(self, array, dtype=np.float32):
        self.n += 1
        self.weights.append(numpy_helper.from_array(np.asarray(array, dtype=dtype), f"c{self.n}"))
        return f"c{self.n}"

    def op(self, kind, inputs, out=None, **attrs):
        self.n += 1
        out = out or f"t{self.n}"
        self.nodes.append(helper.make_node(kind, inputs, [out], **attrs))
        return out

    def rand(self, *shape, scale=1.0):
        return self.const(rng.standard_normal(shape) * scale)

    def save(self, path):
        graph = helper.make_graph(
            self.nodes, "dummy",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", W])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", W])], self.weights)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 9
        save(model, path)
        print(f"wrote {path} ({os.path.getsize(path) / 2**20:.1f} MiB)")


def tiled(g, rows):
    """[batch, W] -> [batch * rows, W]"""
    return g.op("Tile", ["x", g.const([rows, 1], np.int64)])


def averaged(g, t, rows):
    """[batch * rows, W] -> [batch, W], written to the output "y"."""
    grouped = g.op("Reshape", [t, g.const([-1, rows, W], np.int64)])
    g.op("ReduceMean", [grouped], "y", axes=[1], keepdims=0)


def dense(g, rows=4096, layers=8):
    t = tiled(g, rows)
    for _ in range(layers):
        t = g.op("Relu", [g.op("MatMul", [t, g.rand(W, W, scale=W ** -0.5)])])
    averaged(g, t, rows)


def conv(g, images=8, channels=64, layers=8):
    t = g.op("Reshape", [tiled(g, images), g.const([-1, 1, 32, 32], np.int64)])
    t = g.op("Resize", [t, "", g.const([1, 1, 4, 4])], mode="nearest")          # 128 x 128
    t = g.op("Conv", [t, g.rand(channels, 1, 3, 3, scale=1 / 3)], pads=[1, 1, 1, 1])
    for _ in range(layers):
        k = g.rand(channels, channels, 3, 3, scale=(channels * 9) ** -0.5)
        t = g.op("Relu", [g.op("Conv", [t, k], pads=[1, 1, 1, 1])])
    t = g.op("Conv", [t, g.rand(1, channels, 1, 1, scale=channels ** -0.5)])
    t = g.op("AveragePool", [t], kernel_shape=[4, 4], strides=[4, 4])           # 32 x 32
    averaged(g, g.op("Reshape", [t, g.const([-1, W], np.int64)]), images)


def attention(g, rows=256, d=64, layers=4):
    t = g.op("Reshape", [tiled(g, rows), g.const([-1, rows * W // d, d], np.int64)])   # 4096 tokens
    for _ in range(layers):
        q = g.op("MatMul", [t, g.rand(d, d, scale=1 / d)])                      # 1/sqrt(d) twice
        k = g.op("Transpose", [g.op("MatMul", [t, g.rand(d, d, scale=d ** -0.5)])], perm=[0, 2, 1])
        v = g.op("MatMul", [t, g.rand(d, d, scale=d ** -0.5)])
        a = g.op("Softmax", [g.op("MatMul", [q, k])], axis=-1)
        t = g.op("Add", [t, g.op("MatMul", [a, v])])
    averaged(g, g.op("Reshape", [t, g.const([-1, W], np.int64)]), rows)


def elementwise(g, rows=4096, layers=32):
    t = tiled(g, rows)
    for _ in range(layers):
        t = g.op("Tanh", [g.op("Add", [g.op("Mul", [t, g.rand(W)]), g.rand(W, scale=0.1)])])
    averaged(g, t, rows)


TYPES = {"dense": dense, "conv": conv, "attention": attention, "elementwise": elementwise}

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "models"
    os.makedirs(out, exist_ok=True)
    for name, build in TYPES.items():
        g = Graph()
        build(g)
        g.save(os.path.join(out, f"{name}.onnx"))
