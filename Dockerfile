FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
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

# Copy requirements and install dependencies
COPY requirements-frozen.txt /app/
RUN pip3 install --no-cache-dir -r requirements-frozen.txt

# Copy application code
COPY whisperx_service.py /app/
COPY src/ /app/src/

# Environment variables
ENV PORT=19000
ENV HOST=0.0.0.0

EXPOSE 19000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run with Gunicorn
CMD [ "gunicorn", \
      "--bind", "0.0.0.0:19000", \
      "--workers", "1", \
      "--threads", "2", \
      "--timeout", "600", \
      "whisperx_service:app" \
    ]
