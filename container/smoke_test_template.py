#!/usr/bin/env python3
"""smoke_test_template.py - copy into the image as /app/smoke_test.py.

The sanitizer only checks code paths that actually run, so exercise the shapes
production will see: smallest and largest batch, shortest and longest sequence,
and any special inputs (padding, empty prompts, max tokens).
"""
import os
import torch

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.pt")
CASES = [(1, 16), (1, 4096), (32, 128), (64, 512)]    # (batch, sequence) - adapt per model

model = torch.load(MODEL_PATH, weights_only=False).eval().to("cuda")
with torch.inference_mode():
    for batch, seq in CASES:
        x = torch.randint(0, 1000, (batch, seq), device="cuda")
        model(x)
        torch.cuda.synchronize()          # surface asynchronous faults per case
        print(f"ok batch={batch} seq={seq}")
print("SMOKE TEST PASSED")
