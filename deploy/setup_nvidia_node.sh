#!/usr/bin/env bash
# setup_nvidia_node.sh - prepare an NVIDIA GPU node for MPS-based sharing.
# Starts one MPS control daemon PER GPU, so each GPU is its own fault domain.
# Usage: sudo ./setup_nvidia_node.sh                     (all GPUs)
#        sudo ./setup_nvidia_node.sh 0 2                 (selected GPUs)
#        sudo VALIDATION_GPU=3 ./setup_nvidia_node.sh    (GPU 3 kept without MPS)
set -euo pipefail

MPS_ROOT=/var/run/nvidia-mps
LOG_ROOT=/var/log/nvidia-mps

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found"; exit 1; }
command -v nvidia-cuda-mps-control >/dev/null || { echo "MPS control binary not found"; exit 1; }
docker info 2>/dev/null | grep -qi nvidia || echo "WARN: NVIDIA container runtime not detected (install nvidia-container-toolkit)"

GPUS=("$@")
if [ ${#GPUS[@]} -eq 0 ]; then
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi

for g in "${GPUS[@]}"; do
  if [ "$g" = "${VALIDATION_GPU:-}" ]; then
    nvidia-smi -i "$g" -c DEFAULT              # validation GPU: no MPS, checker needs it
    echo "GPU $g reserved for image validation"; continue
  fi
  echo "== GPU $g =="
  nvidia-smi -i "$g" -pm 1                    # persistence mode
  nvidia-smi -i "$g" -c EXCLUSIVE_PROCESS     # only the MPS server may own the GPU
  mkdir -p "$MPS_ROOT/gpu$g" "$LOG_ROOT/gpu$g"
  systemctl enable --now "nvidia-mps@$g.service"
done

for g in "${GPUS[@]}"; do
  [ "$g" = "${VALIDATION_GPU:-}" ] && continue
  CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/gpu$g" \
    bash -c 'echo get_default_active_thread_percentage | nvidia-cuda-mps-control' \
    && echo "GPU $g: MPS daemon reachable"
done
