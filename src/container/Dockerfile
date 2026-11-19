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

COPY container/server.py container/guard.py container/vitro_stats.py container/entrypoint.sh /app/
COPY --from=observer /libvitroobs.so /opt/vitro/libvitroobs.so
COPY container/supervisord.conf /etc/supervisord.conf

ENV PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MODEL_PATH=/models/model.onnx \
    VITRO_STATS_FILE=/run/vitro/stats \
    PORT=8000
RUN mkdir -p /run/vitro
EXPOSE 8000
# entrypoint.sh connects the container to its MPS namespace (see the file).
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["supervisord", "-c", "/etc/supervisord.conf"]
