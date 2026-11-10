# Inference Workload Management Runtime (IWMR) image. Vendor-neutral: the vendor
# is selected at run time with GPUSHARE_VENDOR. Build from the repository root:
#   docker build -f runtime/Dockerfile -t iwmr:latest .
FROM python:3.12-slim

RUN pip install --no-cache-dir fastapi uvicorn docker requests anyio
COPY runtime/ /opt/iwmr/
COPY tests/mps_sm_count.py /opt/iwmr/tools/mps_sm_count.py

WORKDIR /opt/iwmr
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
CMD ["uvicorn", "runtime:app", "--host", "0.0.0.0", "--port", "8080"]
