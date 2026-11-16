# Inference container: supervisord runs the ONNX Runtime server and guard.py.
# Build from the repository root:
#   docker build -f container/Dockerfile -t onnx-infer:latest .
# The model is not part of the image; mount it read-only at /models/model.onnx.
FROM python:3.12-slim

# onnxruntime-gpu[cuda,cudnn] brings the CUDA and cuDNN libraries as pip packages,
# so no CUDA base image is needed. The GPU driver library (libcuda) is injected at
# run time by the NVIDIA container toolkit.
RUN pip install --no-cache-dir "onnxruntime-gpu[cuda,cudnn]" numpy fastapi uvicorn supervisor

COPY container/server.py container/guard.py /app/
COPY container/supervisord.conf /etc/supervisord.conf

ENV PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MODEL_PATH=/models/model.onnx \
    PORT=8000
EXPOSE 8000
CMD ["supervisord", "-c", "/etc/supervisord.conf"]
