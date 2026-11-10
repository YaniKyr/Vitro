"""gpu_lowlevel.py - minimal ctypes access to the CUDA driver and HIP runtime.

Used by the test scripts so they run with no PyTorch and no toolkit:
NVIDIA kernels are PTX (JIT-compiled by the driver); AMD kernels are HIP C++
compiled at runtime with hiprtc (part of every ROCm install).
"""
import ctypes, glob


class Cuda:
    def __init__(self, device=0):
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.call("cuInit", 0)
        self.dev, ctx = ctypes.c_int(), ctypes.c_void_p()
        self.call("cuDeviceGet", ctypes.byref(self.dev), device)
        self.call("cuDevicePrimaryCtxRetain", ctypes.byref(ctx), self.dev)   # MPS client here
        self.call("cuCtxSetCurrent", ctx)

    def call(self, name, *args, check=True):
        rc = getattr(self.lib, name)(*args)
        if rc and check:
            s = ctypes.c_char_p()
            self.lib.cuGetErrorString(rc, ctypes.byref(s))
            raise RuntimeError(f"{name}: {s.value.decode() if s.value else rc}")
        return rc

    def kernel(self, ptx: bytes, name: str):
        mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
        self.call("cuModuleLoadData", ctypes.byref(mod), ctypes.c_char_p(ptx))
        self.call("cuModuleGetFunction", ctypes.byref(fn), mod, name.encode())
        return fn

    def launch(self, fn, blocks, threads, *args):
        params = (ctypes.c_void_p * max(len(args), 1))(*[ctypes.addressof(a) for a in args])
        self.call("cuLaunchKernel", fn, blocks, 1, 1, threads, 1, 1, 0, None, params, None)
        self.call("cuCtxSynchronize")

    def malloc(self, n):
        p = ctypes.c_uint64()
        rc = self.call("cuMemAlloc_v2", ctypes.byref(p), ctypes.c_size_t(n), check=False)
        return p.value if rc == 0 else None


class Hip:
    def __init__(self, device=0):
        self.lib = ctypes.CDLL("libamdhip64.so")
        self.rtc = None
        self.call("hipSetDevice", device)

    def call(self, name, *args, check=True):
        rc = getattr(self.lib, name)(*args)
        if rc and check:
            self.lib.hipGetErrorString.restype = ctypes.c_char_p
            raise RuntimeError(f"{name}: {self.lib.hipGetErrorString(rc).decode()}")
        return rc

    def kernel(self, src: str, name: str):
        if self.rtc is None:
            self.rtc = ctypes.CDLL("libhiprtc.so")
        prog, size = ctypes.c_void_p(), ctypes.c_size_t()
        self.rtc.hiprtcCreateProgram(ctypes.byref(prog), src.encode(), b"k.cpp", 0, None, None)
        if self.rtc.hiprtcCompileProgram(prog, 0, None):
            raise RuntimeError("hiprtc compilation failed")
        self.rtc.hiprtcGetCodeSize(prog, ctypes.byref(size))
        code = ctypes.create_string_buffer(size.value)
        self.rtc.hiprtcGetCode(prog, code)
        mod, fn = ctypes.c_void_p(), ctypes.c_void_p()
        self.call("hipModuleLoadData", ctypes.byref(mod), code)
        self.call("hipModuleGetFunction", ctypes.byref(fn), mod, name.encode())
        return fn

    def launch(self, fn, blocks, threads, *args):
        params = (ctypes.c_void_p * max(len(args), 1))(*[ctypes.addressof(a) for a in args])
        self.call("hipModuleLaunchKernel", fn, blocks, 1, 1, threads, 1, 1, 0, None, params, None)
        self.call("hipDeviceSynchronize")

    def malloc(self, n):
        p = ctypes.c_void_p()
        rc = self.call("hipMalloc", ctypes.byref(p), ctypes.c_size_t(n), check=False)
        return p.value if rc == 0 else None


def amd_cu_count(index=0):
    """Total CUs of the index-th GPU from KFD topology (no HIP enum dependencies)."""
    gpus = []
    for node in sorted(glob.glob("/sys/class/kfd/kfd/topology/nodes/*"),
                       key=lambda p: int(p.rsplit("/", 1)[1])):
        props = dict(l.split() for l in open(f"{node}/properties") if len(l.split()) == 2)
        if int(props.get("simd_count", 0)):
            gpus.append(int(props["simd_count"]) // int(props["simd_per_cu"]))
    return gpus[index]


def open_gpu(vendor):
    return Cuda() if vendor == "nvidia" else Hip()
