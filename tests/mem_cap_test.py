#!/usr/bin/env python3
"""mem_cap_test.py - verify the per-workload memory cap is enforced.

Allocates 256 MiB blocks until an allocation fails and prints the total reached.
Expect a total close to the cap (MPS: CUDA_MPS_PINNED_DEVICE_MEM_LIMIT;
AMD: HIPMEM_LIMIT_BYTES with LD_PRELOAD=libhipmemcap.so).

  NVIDIA: CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=2G python3 mem_cap_test.py nvidia
  AMD:    LD_PRELOAD=/opt/gpushare/lib/libhipmemcap.so HIPMEM_LIMIT_BYTES=2147483648 \
          python3 mem_cap_test.py amd
"""
import sys
from gpu_lowlevel import open_gpu

BLOCK = 256 * 2**20
vendor = sys.argv[1] if len(sys.argv) > 1 else "nvidia"
gpu = open_gpu(vendor)
total, ptrs = 0, []
while total < 512 * 2**30:
    p = gpu.malloc(BLOCK)
    if p is None:
        break
    ptrs.append(p)
    total += BLOCK
print(f"allocated {total / 2**30:.2f} GiB before the first failure ({len(ptrs)} blocks)")
