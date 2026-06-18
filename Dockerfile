FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /workspace/sla-ssl
COPY . .
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir ".[eval]"

ENV PYTHONUNBUFFERED=1
CMD ["bash"]
