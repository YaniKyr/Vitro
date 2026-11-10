#!/usr/bin/env bash
# setup_amd_node.sh - prepare an AMD (ROCm) GPU node for CU-mask based sharing.
# Builds the hipMalloc interposer and prints the GPU inventory from KFD sysfs.
# Usage: sudo ./setup_amd_node.sh
set -euo pipefail

LIB_DIR=/opt/gpushare/lib
SRC_DIR="$(cd "$(dirname "$0")/../container" && pwd)"

[ -e /dev/kfd ] || { echo "/dev/kfd missing: amdgpu/ROCm driver not loaded"; exit 1; }
command -v amd-smi >/dev/null || echo "WARN: amd-smi not found (monitoring/partitioning unavailable)"

mkdir -p "$LIB_DIR"
gcc -O2 -fPIC -shared -o "$LIB_DIR/libhipmemcap.so" "$SRC_DIR/hipmemcap.c" -ldl -lpthread
echo "built $LIB_DIR/libhipmemcap.so"

echo "== GPU inventory (KFD topology) =="
for n in /sys/class/kfd/kfd/topology/nodes/*; do
  p="$n/properties"
  simd=$(awk '/^simd_count/{print $2}' "$p")
  [ "${simd:-0}" -gt 0 ] || continue            # skip CPU nodes
  per_cu=$(awk '/^simd_per_cu/{print $2}' "$p")
  minor=$(awk '/^drm_render_minor/{print $2}' "$p")
  xcc=$(awk '/^num_xcc/{print $2}' "$p"); xcc=${xcc:-1}
  vram=$(cat "/sys/class/drm/renderD$minor/device/mem_info_vram_total" 2>/dev/null || echo 0)
  echo "renderD$minor  CUs=$((simd / per_cu))  XCCs=$xcc  VRAM=$((vram / 1024 / 1024)) MiB"
done

if command -v amd-smi >/dev/null; then
  amd-smi static --partition 2>/dev/null || true
fi
