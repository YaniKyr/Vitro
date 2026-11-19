# Inference container: supervisord runs the ONNX Runtime server and guard.py.
# The server runs with libvitroobs.so preloaded (per-container GPU metrics).
# Build from src/:
#   docker build -f container/Dockerfile -t onnx-infer:latest .
# The model is not part of the image; mount it read-only at /models/model.onnx.
FROM python:3.12-slim AS observer
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
COPY container/vitro_observer.c /src/
RUN gcc -O2 -fPIC -shared -o /libvitroobs.so /src/vitro_observer.c -lpthread -ldl -lm

FROM python:3.12-slim

# onnxruntime-gpu[cuda,cudnn] brings the CUDA and cuDNN libraries as pip packages,
# so no CUDA base image is needed. The GPU driver library (libcuda) is injected at
# run time by the NVIDIA container toolkit.
RUN pip install --no-cache-dir "onnxruntime-gpu[cuda,cudnn]" numpy fastapi uvicorn supervisor

COPY container/server.py container/guard.py container/vitro_stats.py /app/
COPY --from=observer /libvitroobs.so /opt/vitro/libvitroobs.so
COPY container/supervisord.conf /etc/supervisord.conf

ENV PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MODEL_PATH=/models/model.onnx \
    VITRO_STATS_FILE=/run/vitro/stats \
    PORT=8000
RUN mkdir -p /run/vitro
EXPOSE 8000
CMD ["supervisord", "-c", "/etc/supervisord.conf"]
