FROM python:3.11-slim

# Install dependencies needed by Ollama installer
RUN apt-get update && apt-get install -y \
    curl \
    pciutils \
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
sleep 5\n\
ollama pull qwen3-vl:4b-instruct-q4_K_M\n\
uvicorn main:app --host 0.0.0.0 --port $PORT\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
