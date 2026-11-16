#!/usr/bin/env python3
"""Print the SM count visible to this process (reflects the MPS active thread percentage).

Usage:
  python mps_sm_count.py                 # SMs visible now
  python mps_sm_count.py 25 50 100       # SMs at each CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
"""
import ctypes, os, subprocess, sys

def sm_count(device=0):
    cu = ctypes.CDLL("libcuda.so.1")
    cu.cuInit(0)
    dev, ctx, sms = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_int()
    cu.cuDeviceGet(ctypes.byref(dev), device)
    cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)  # connect as MPS client
    cu.cuCtxSetCurrent(ctx)
    cu.cuDeviceGetAttribute(ctypes.byref(sms), 16, dev)  # 16 = MULTIPROCESSOR_COUNT
    return sms.value

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(sm_count())
    else:
        for pct in sys.argv[1:]:
            env = dict(os.environ, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=pct)
            out = subprocess.run([sys.executable, __file__], env=env,
                                 capture_output=True, text=True).stdout.strip()
            print(f"{pct:>3}%  {out} SMs")
