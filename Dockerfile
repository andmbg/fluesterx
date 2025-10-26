FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Make clear we need 2 env vars: API_TOKEN and HF_TOKEN
LABEL org.opencontainers.image.title="fluesterx"
LABEL org.opencontainers.image.description="Fluesterx ASR and embedder service (requires API_TOKEN and HF_TOKEN at runtime)"
LABEL com.fluesterx.required_env="API_TOKEN,HF_TOKEN"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install cuDNN explicitly
RUN apt-get update && apt-get install -y \
    libcudnn8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies with compatible PyTorch
RUN pip3 install --no-cache-dir \
    torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install WhisperX and all required dependencies
RUN pip3 install --no-cache-dir \
    whisperx \
    flask \
    gunicorn \
    pyannote.audio \
    python-dotenv \
    pandas \
    transformers \
    speechbrain \
    sentence-transformers

# Create app directory
WORKDIR /app

# Copy service files
COPY whisperx_service.py /app/
COPY src/ /app/src/

# Create log directory
RUN mkdir -p /var/log

# Set default environment variables (can be overridden)
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV FLASK_ENV=production
ENV HOST=0.0.0.0
ENV PORT=19000
ENV ASR_MODEL=distil-large-v3
ENV WORKERS=1
ENV THREADS=2
ENV TIMEOUT=300
ENV API_TOKEN=""
ENV HF_TOKEN=""

# Set CUDA and cuDNN library paths
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Expose port
EXPOSE 19000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/ || exit 1

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Start service
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python3", "whisperx_service.py"]
