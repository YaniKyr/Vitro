#!/usr/bin/env bash
# setup_nvidia_node.sh - prepare an NVIDIA GPU node for MPS-based sharing.
# Installs the systemd unit and starts one MPS control daemon PER GPU, so each
# GPU is its own fault domain. Everything printed (including errors) is also
# appended to /var/log/nvidia-mps/setup.log.
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
  if echo get_default_active_thread_percentage | nvidia-cuda-mps-control >/dev/null; then
    echo "GPU $g: MPS daemon reachable (pipe dir $CUDA_MPS_PIPE_DIRECTORY)"
  else
    echo "ERROR: GPU $g: MPS daemon not reachable"; exit 1
  fi
  if command -v python3 >/dev/null && [ -f "$SM_HELPER" ]; then
    echo "GPU $g: SMs visible to an MPS client:"
    python3 "$SM_HELPER" 100 50 25 | sed 's/^/  /'
  else
    echo "WARN: GPU $g: SM count skipped (python3 or $SM_HELPER missing)"
  fi
done
echo "done; log: $LOG"
