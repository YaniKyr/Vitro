#!/usr/bin/env bash
# mps_split.sh - divide one GPU evenly into N MPS namespaces.
#
# Each namespace gets its own SM partition (the GPU's chunks divided by N,
# rounded down; chunks left over stay unallocated) and the same memory limit.
# Existing namespaces and SM partitions of the server are replaced, so no
# client may be attached. Needs MPS protocol 3 (see mps_version.sh) and root.
#
#   sudo ./mps_split.sh N [memory per namespace, e.g. 400M]
#   SERVER=vitro GPU=0 PREFIX=ns   server name, GPU index, namespace name prefix
#
# Without a memory size the GPU's currently free memory is divided by N, less
# 16 MiB each. Namespaces are named <PREFIX>1 .. <PREFIX>N; a container joins
# one through its pipe directory (see container/entrypoint.sh).
set -euo pipefail

N=${1:?usage: mps_split.sh N [memory per namespace]}
MEM=${2:-}
SERVER=${SERVER:-vitro}; GPU=${GPU:-0}; PREFIX=${PREFIX:-ns}
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/var/run/nvidia-mps/gpu$GPU}
export CUDA_MPS_PROTOCOL_VERSION=3
ctl() { nvidia-cuda-mps-control "$@"; }

ctl device list >/dev/null 2>&1 || { echo "ERROR: no protocol 3 MPS daemon in $CUDA_MPS_PIPE_DIRECTORY"; exit 1; }
IFS=, read -r _ _ _ UUID _ SMS CHUNKS _ < <(ctl device list --format=csv,noheader | awk -F, '$2 == "-"' | head -n1)
PER=$((CHUNKS / N))
[ "$PER" -ge 1 ] || { echo "ERROR: $CHUNKS chunks cannot be divided into $N partitions"; exit 1; }
if [ -z "$MEM" ]; then
  free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  MEM="$((free / N - 16))M"
fi

ctl server list --format=csv,noheader | grep -q "^$SERVER," || ctl server create "$SERVER" --uid=0
if ctl client list --format=csv,noheader | awk -F, -v s="$SERVER" '$2 == s' | grep -q .; then
  echo "ERROR: server $SERVER has clients attached; stop them first"; exit 1
fi
# Remove what is there: a partition cannot go while a namespace refers to it.
ctl namespace list --server="$SERVER" --format=csv,noheader | cut -d, -f1 | sort -u | while read -r ns; do
  [ "$ns" = default ] && continue
  ctl namespace set "$ns" --server="$SERVER" --device="$UUID" --sm-partition=none >/dev/null 2>&1 || true
  ctl namespace delete "$ns" --server="$SERVER" >/dev/null
done
ctl sm-partition list --server="$SERVER" --format=csv,noheader | awk -F, '$3 != "-" {print $3}' | while read -r part; do
  ctl sm-partition delete "$part" --server="$SERVER" --device="$UUID" >/dev/null
done

for i in $(seq 1 "$N"); do
  ns="$PREFIX$i"
  ctl sm-partition create "$ns" --server="$SERVER" --device="$UUID" --chunks="$PER" >/dev/null
  ctl namespace create "$ns" --server="$SERVER" >/dev/null
  ctl namespace set "$ns" --server="$SERVER" --device="$UUID" --sm-partition="$ns" >/dev/null
  ctl namespace set "$ns" --server="$SERVER" --pinned-memory-limit="$MEM" >/dev/null
done
echo "GPU $GPU ($SMS SMs, $CHUNKS chunks): $N namespaces of $((PER * SMS / CHUNKS)) SMs and $MEM on server $SERVER," \
     "$(((CHUNKS - PER * N) * SMS / CHUNKS)) SMs unallocated"
ctl namespace list --server="$SERVER" --format=csv,noheader | awk -F, '$1 != "default" {print "  " $2 "/" $1, $4, $11, $9}'
