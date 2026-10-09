FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Expose NVIDIA GPU capabilities to the container
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONUNBUFFERED=1


FROM python:3.11-slim

# Install dependencies needed by Ollama installer
RUN apt-get update && apt-get install -y \
    curl \
    pciutils \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Download and run Ollama install script reliably
RUN curl -fsSL https://ollama.com/install.sh -o install.sh \
    && chmod +x install.sh \
    && ./install.sh \
    && rm install.sh

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the Python FastAPI/Uvicorn port
EXPOSE 8100
ENV PORT=8100

# Script to start Ollama, pull model, then start Uvicorn
RUN echo '#!/bin/bash\n\
ollama serve &\n\
#sleep 5\n\
until curl -s http://localhost:11434/ > /dev/null; do sleep 1; done\n\
ollama pull qwen3-vl:4b-instruct-q4_K_M\n\
uvicorn main:app --host 0.0.0.0 --port $PORT\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
