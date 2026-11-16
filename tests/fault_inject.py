#!/usr/bin/env python3
"""fault_inject.py - deliberately trigger an illegal GPU memory access.

Use only on the testbed. Start several burn clients first, then run this as an
extra client on the same GPU and record which processes die and with which
error. That measures the fault blast radius of each backend.

  python3 fault_inject.py nvidia     (run with the same MPS pipe dir as the victims)
  python3 fault_inject.py amd
"""
import ctypes, sys, time
from gpu_lowlevel import open_gpu

PTX = rb"""
.version 7.0
.target sm_52
.address_size 64
.visible .entry boom()
{
  .reg .u64 %rd1; .reg .f32 %f1;
  mov.u64 %rd1, 16;                 // address 0x10 is never mapped
  mov.f32 %f1, 0f3F800000;
  st.global.f32 [%rd1], %f1;
  ret;
}
"""
HIP_SRC = r"""
extern "C" __global__ void boom() { *(volatile int*)0x10 = 1; }
"""

vendor = sys.argv[1] if len(sys.argv) > 1 else "nvidia"
gpu = open_gpu(vendor)
fn = gpu.kernel(PTX if vendor == "nvidia" else HIP_SRC, "boom")
print(f"[{time.strftime('%H:%M:%S')}] launching faulting kernel", flush=True)
try:
    gpu.launch(fn, 1, 1)
    print("no fault raised (unexpected)")
except RuntimeError as e:
    print(f"fault reported to the culprit: {e}", flush=True)
    sys.exit(1)
