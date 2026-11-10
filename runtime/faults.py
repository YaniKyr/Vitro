"""faults.py - kernel-log GPU fault monitor (thread inside the runtime).

Reads /dev/kmsg (runtime started with --device /dev/kmsg --cap-add SYSLOG) and
attributes GPU fault records to containers through /proc/<pid>/cgroup (needs
--pid host). NVIDIA Xid records may name the MPS server rather than the client,
so they corroborate guard.py; amdgpu page-fault records name the process.
"""
import re, threading

XID = re.compile(r"NVRM: Xid \(PCI:(?P<pci>[0-9a-fA-F:.]+)\): (?P<xid>\d+)"
                 r"(?:, pid=(?P<pid>\d+|'<unknown>'), name=(?P<name>[^,]+))?")
AMD_PF = re.compile(r"amdgpu (?P<pci>[0-9a-fA-F:.]+): \[\w+\] (?:\w+ )?page fault.*?"
                    r"process (?P<name>\S+) pid (?P<pid>\d+)")
AMD_RESET = re.compile(r"amdgpu (?P<pci>[0-9a-fA-F:.]+): .*GPU reset")
FATAL_XIDS = {13, 31, 43, 45, 48, 62, 63, 64, 74, 79, 94, 95}
DEVICE_XIDS = {48, 74, 79, 95}


def bus_dev(pci):
    """'0000:3b:00.0' / 'PCI:0000:3b:00' -> '3b:00' (comparable across sources)."""
    m = re.search(r"([0-9a-fA-F]{2}):([0-9a-fA-F]{2})(?:\.\d)?$", pci or "")
    return f"{m.group(1)}:{m.group(2)}".lower() if m else ""


def container_of(pid):
    try:
        m = re.search(r"([0-9a-f]{64})", open(f"/proc/{pid}/cgroup").read())
        return m.group(1) if m else None
    except OSError:
        return None


def parse(line):
    if m := XID.search(line):
        xid = int(m["xid"])
        if xid not in FATAL_XIDS:
            return None
        pid = m["pid"] if m["pid"] and m["pid"].isdigit() else None
        return {"source": "nvidia_xid", "xid": xid, "pci": bus_dev(m["pci"]), "pid": pid,
                "device_level": xid in DEVICE_XIDS}
    if m := AMD_PF.search(line):
        return {"source": "amdgpu_pagefault", "pci": bus_dev(m["pci"]), "pid": m["pid"],
                "device_level": False}
    if m := AMD_RESET.search(line):
        return {"source": "amdgpu_reset", "pci": bus_dev(m["pci"]), "pid": None,
                "device_level": True}
    return None


class FaultMonitor(threading.Thread):
    def __init__(self, on_fault, path="/dev/kmsg"):
        super().__init__(daemon=True)
        self.on_fault, self.path = on_fault, path

    def run(self):
        try:
            f = open(self.path, "r", errors="replace")
            f.seek(0, 2)                                   # only new records
        except OSError as e:
            print(f"[faults] kernel log unavailable ({e}); relying on guard.py only")
            return
        while True:
            try:
                line = f.readline()
            except OSError:                                # kmsg overrun: skip ahead
                continue
            evt = parse(line.split(";", 1)[-1])
            if evt:
                evt["container"] = container_of(evt["pid"]) if evt["pid"] else None
                evt["raw"] = line.strip()[:400]
                self.on_fault(evt)
