#!/usr/bin/env python3
"""vitro_stats.py - read the stats file written by libvitroobs.so.

The observer library (vitro_observer.c) publishes the container's own GPU
figures to a memory-mapped file about ten times a second. This module decodes
it; the field list below mirrors `struct vitro_stats`.

  from vitro_stats import read
  read("/run/vitro/stats")      -> dict, or None if there is no valid file yet

  python3 vitro_stats.py [path] -> prints the figures as JSON
"""
import json, os, struct, sys, time

MAGIC = 0x53424F4F52544956                     # "VITROOBS"
DEFAULT_PATH = os.environ.get("VITRO_STATS_FILE", "/run/vitro/stats")

HEADER = [("magic", "Q"), ("version", "I"), ("seq", "I")]
FIELDS = HEADER + [(n, "Q") for n in (
    "pid", "updated_ns", "sm_count", "mem_cap_bytes", "mem_model_bytes", "mem_model_peak_bytes",
    "mem_footprint_bytes", "mem_overhead_bytes", "mem_free_bytes", "mem_total_bytes",
    "kernels", "kernels_unattributed", "kernels_unobserved", "pending", "inferences", "inflight",
)] + [(n, "d") for n in (
    "gpu_ms", "smw_ms", "busy_ms", "util_1s", "util_10s", "smw_util_1s", "smw_util_10s",
    "inf_gpu_ms", "inf_wall_ms", "last_gpu_ms", "last_wall_ms", "last_smw_ms",
)] + [("last_kernels", "Q")]
LAYOUT = struct.Struct("<" + "".join(f for _, f in FIELDS))


def read(path=DEFAULT_PATH):
    """Return the current figures, or None. Retries while the writer is mid-update."""
    for _ in range(50):
        try:
            with open(path, "rb") as f:
                raw = f.read(LAYOUT.size)
        except OSError:
            return None
        if len(raw) < LAYOUT.size:
            return None
        s = dict(zip((n for n, _ in FIELDS), LAYOUT.unpack(raw)))
        if s["magic"] != MAGIC:
            return None
        if s["seq"] % 2 == 0:                  # even: not being written
            s["age_s"] = round(time.time() - s["updated_ns"] / 1e9, 3)
            del s["magic"], s["seq"]
            return s
        time.sleep(0.001)
    return None


if __name__ == "__main__":
    stats = read(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH)
    if stats is None:
        sys.exit("no valid stats file")
    print(json.dumps(stats, indent=2))
