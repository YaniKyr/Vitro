#!/usr/bin/env bash
# setup_nvidia_node.sh - prepare an NVIDIA GPU node for MPS-based sharing.
# Installs the systemd unit and starts one MPS control daemon PER GPU, so each
# GPU is its own fault domain. Everything printed (including errors) is also
# appended to /var/log/nvidia-mps/setup.log.
# The MPS version is checked first (see mps_version.sh): with protocol 3 the
# daemons offer namespaces and SM partitions, with an older driver only the
# legacy per-client limits. MPS_REQUIRE_V3=1 makes the legacy case an error.
# Usage: sudo ./setup_nvidia_node.sh                     (all GPUs)
#        sudo ./setup_nvidia_node.sh 0 2                 (selected GPUs)
#        sudo VALIDATION_GPU=3 ./setup_nvidia_node.sh    (GPU 3 kept without MPS)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MPS_ROOT=/var/run/nvidia-mps
LOG_ROOT=/var/log/nvidia-mps
LOG="$LOG_ROOT/setup.log"
UNIT=/etc/systemd/system/nvidia-mps@.service
SM_HELPER="$HERE/../tests/mps_sm_count.py"

[ "$(id -u)" -eq 0 ] || { echo "run as root (sudo)"; exit 1; }
mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "ERROR: line $LINENO: \"$BASH_COMMAND\" exited with $?"' ERR
echo "===== $(date '+%F %T') setup_nvidia_node.sh $* ====="

command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found"; exit 1; }
command -v nvidia-cuda-mps-control >/dev/null || { echo "ERROR: MPS control binary not found"; exit 1; }

# MPS version: what the driver's control binary can do decides how limits are set.
. "$HERE/mps_version.sh"
echo "driver $(mps_driver_version); MPS control binary offers protocol $(mps_binary_protocol)"
if mps_supports_namespaces; then
  echo "MPS namespaces and SM partitions are available"
else
  echo "WARN: MPS namespaces are NOT available (need protocol 3 and driver >= $MPS_MIN_DRIVER_NAMESPACES);" \
       "only the legacy per-client limits can be used. Upgrade the driver to change that."
  [ "${MPS_REQUIRE_V3:-0}" != 1 ] || { echo "ERROR: MPS_REQUIRE_V3=1"; exit 1; }
fi
docker info 2>/dev/null | grep -qi nvidia || echo "WARN: NVIDIA container runtime not detected (install nvidia-container-toolkit)"

# Install (or refresh) the systemd template unit that runs the daemons.
if ! cmp -s "$HERE/nvidia-mps@.service" "$UNIT"; then
  install -m 644 "$HERE/nvidia-mps@.service" "$UNIT"
  command -v restorecon >/dev/null && restorecon "$UNIT"
  systemctl daemon-reload
  echo "installed $UNIT"
fi

GPUS=("$@")
if [ ${#GPUS[@]} -eq 0 ]; then
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi

for g in "${GPUS[@]}"; do
  echo "== GPU $g =="
  nvidia-smi -i "$g" --format=csv,noheader \
    --query-gpu=name,memory.total,driver_version,pci.bus_id,uuid | sed 's/^/info: /'
  if [ "$g" = "${VALIDATION_GPU:-}" ]; then
    nvidia-smi -i "$g" -c DEFAULT              # validation GPU: no MPS, checker needs it
    echo "GPU $g reserved for image validation"; continue
  fi
  nvidia-smi -i "$g" -pm 1                    # persistence mode
  nvidia-smi -i "$g" -c EXCLUSIVE_PROCESS     # only the MPS server may own the GPU
  mkdir -p "$MPS_ROOT/gpu$g" "$LOG_ROOT/gpu$g"
  if ! systemctl enable --now "nvidia-mps@$g.service"; then
    echo "ERROR: nvidia-mps@$g.service did not start"
    systemctl status "nvidia-mps@$g.service" --no-pager -n 20 || true
    exit 1
  fi
done

# Verify each daemon and record what an MPS client sees (SM count per percentage).
for g in "${GPUS[@]}"; do
  [ "$g" = "${VALIDATION_GPU:-}" ] && continue
  export CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/gpu$g"
  proto=$(mps_daemon_protocol "$CUDA_MPS_PIPE_DIRECTORY")
  if [ "$proto" = none ]; then
    echo "ERROR: GPU $g: MPS daemon not reachable"; exit 1
  fi
  echo "GPU $g: MPS daemon reachable, protocol $proto (pipe dir $CUDA_MPS_PIPE_DIRECTORY)"
  if [ "$proto" = 3 ]; then
    # SMs and chunks: an SM partition is a whole number of chunks.
    CUDA_MPS_PROTOCOL_VERSION=3 nvidia-cuda-mps-control device list | sed 's/^/  /'
  elif mps_supports_namespaces; then
    # The binary offers 3 but this daemon was started before the unit asked for it.
    echo "WARN: GPU $g: daemon still speaks protocol 2; restart nvidia-mps@$g.service"
    [ "${MPS_REQUIRE_V3:-0}" != 1 ] || exit 1
  fi
  if command -v python3 >/dev/null && [ -f "$SM_HELPER" ]; then
    echo "GPU $g: SMs visible to an MPS client:"
    python3 "$SM_HELPER" 100 50 25 | sed 's/^/  /'
  else
    echo "WARN: GPU $g: SM count skipped (python3 or $SM_HELPER missing)"
  fi
done
echo "done; log: $LOG"
