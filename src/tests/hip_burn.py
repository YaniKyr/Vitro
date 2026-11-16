#!/usr/bin/env python3
"""hip_burn.py - AMD counterpart of mps_burn.py (same two-function structure).

Each client gets a disjoint CU range via HSA_CU_MASK instead of an MPS percentage.

  python3 hip_burn.py 25 50 25     # 3 concurrent clients on 25 %, 50 %, 25 % of the CUs
"""
import ctypes, json, math, os, subprocess, sys, time
from gpu_lowlevel import Hip, amd_cu_count

SRC = r"""
extern "C" __global__ void burn(float *out, unsigned iters) {
  float x = threadIdx.x, c = 1.0f;
  for (unsigned i = 0; i < iters; ++i) x = x * c + c;
  if (x == -1.0f) *out = x;          // never true; keeps the loop alive
}
"""
DURATION, ITERS, BLOCKS, THREADS = 10, 20000, 1024, 256


def worker():
    gpu = Hip()
    fn = gpu.kernel(SRC, "burn")
    p = ctypes.c_void_p()
    gpu.call("hipMalloc", ctypes.byref(p), ctypes.c_size_t(4))
    out, iters = ctypes.c_void_p(p.value), ctypes.c_uint32(ITERS)
    launches, t0 = 0, time.time()
    while time.time() - t0 < DURATION:
        gpu.launch(fn, BLOCKS, THREADS, out, iters)
        launches += 1
    ms = 1000 * (time.time() - t0) / max(launches, 1)
    print(json.dumps({"mask": os.environ.get("HSA_CU_MASK", "all"),
                      "launches": launches, "ms": round(ms, 2)}))


def launch(percentages):
    total, start, procs = amd_cu_count(), 0, []
    for pct in percentages:
        n = max(2, math.ceil(int(pct) / 100 * total))
        n += n % 2
        if start + n > total:
            sys.exit(f"shares exceed the {total} CUs of the GPU")
        env = dict(os.environ, HSA_CU_MASK=f"0:{start}-{start + n - 1}")
        start += n                                    # disjoint ranges
        procs.append(subprocess.Popen([sys.executable, __file__, "--worker"],
                                      env=env, stdout=subprocess.PIPE, text=True))
    print(f"GPU has {total} CUs\n mask          launches  ms/launch")
    for p in procs:
        lines = p.communicate()[0].strip().splitlines()
        if not lines:
            print("client failed")
            continue
        r = json.loads(lines[-1])
        print(f"{r['mask']:<13} {r['launches']:>8}  {r['ms']:>9}")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker()
    else:
        launch(sys.argv[1:] or ["100"])
