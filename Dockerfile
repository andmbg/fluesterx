FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH


RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libavdevice-dev \
    libswresample-dev \
    pkg-config \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

COPY whisperx_service.py /app/
COPY src/ /app/src/

ENV PORT=19000
ENV HOST=0.0.0.0
ENV DEVICE=cuda

EXPOSE 19000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD [ "gunicorn", \
      "--bind", "0.0.0.0:19000", \
      "--workers", "1", \
      "--threads", "1", \
      "--timeout", "600", \
      "whisperx_service:app" \
    ]
