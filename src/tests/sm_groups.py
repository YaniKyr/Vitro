#!/usr/bin/env python3
"""Print how many SM groups (partitions) the GPU can be split into.

Usage:  python sm_groups.py [device index]

Asks the driver for the SM partition limits (cuDeviceGetDevResource). If the
driver does not give them, the group size is taken from the compute capability.
"""
import ctypes, sys

CU_DEV_RESOURCE_TYPE_SM = 1
FALLBACK = {6: 2, 7: 2, 8: 4}          # compute capability major -> group size; 9.0+ -> 8


class SmResource(ctypes.Structure):    # CUdevSmResource in cuda.h
    _fields_ = [("smCount", ctypes.c_uint),
                ("minSmPartitionSize", ctypes.c_uint),
                ("smCoscheduledAlignment", ctypes.c_uint)]


class DevResource(ctypes.Structure):   # CUdevResource in cuda.h (144 bytes in ABI v1)
    _fields_ = [("type", ctypes.c_int),
                ("_internal_padding", ctypes.c_ubyte * 92),
                ("sm", SmResource),
                ("_spare", ctypes.c_ubyte * 1024)]   # room in case a newer driver writes more


def main(index=0):
    cu = ctypes.CDLL("libcuda.so.1")

    def call(name, *args):
        rc = getattr(cu, name)(*args)
        if rc:
            sys.exit(f"{name} failed with error {rc}")

    dev, major, minor, total = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    name = ctypes.create_string_buffer(256)
    call("cuInit", 0)
    call("cuDeviceGet", ctypes.byref(dev), index)
    call("cuDeviceGetName", name, len(name), dev)
    call("cuDeviceGetAttribute", ctypes.byref(major), 75, dev)   # compute capability major
    call("cuDeviceGetAttribute", ctypes.byref(minor), 76, dev)   # compute capability minor
    call("cuDeviceGetAttribute", ctypes.byref(total), 16, dev)   # SM count
    total = total.value

    res, minimum, align, source = DevResource(), 0, 0, ""
    if hasattr(cu, "cuDeviceGetDevResource"):
        rc = cu.cuDeviceGetDevResource(dev, ctypes.byref(res), CU_DEV_RESOURCE_TYPE_SM)
        if rc == 0:
            total = res.sm.smCount or total
            minimum, align = res.sm.minSmPartitionSize, res.sm.smCoscheduledAlignment
        else:
            source = f"driver query failed with error {rc}; "
    else:
        source = "driver has no cuDeviceGetDevResource; "

    if minimum and align:
        group = -(-minimum // align) * align                     # minimum rounded up to alignment
        source = f"driver: minimum {minimum}, alignment {align}"
    else:
        group = FALLBACK.get(major.value, 8 if major.value >= 9 else 2)
        source += "from compute capability"

    groups = total // group
    print(f"GPU:                {name.value.decode()}")
    print(f"Compute capability: {major.value}.{minor.value}")
    print(f"Total SMs:          {total}")
    print(f"Group size:         {group} SMs  ({source})")
    print(f"Max groups:         {groups}")
    print(f"Leftover SMs:       {total - groups * group}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
