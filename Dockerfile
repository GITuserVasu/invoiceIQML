FROM python:3.11-slim

# Install Ollama and curl
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://ollama.com | sh

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the Python FastAPI/Uvicorn port
EXPOSE 8000
ENV PORT=8000

# Script to start Ollama, pull model, then start Uvicorn
RUN echo '#!/bin/bash\n\
ollama serve &\n\
sleep 5\n\
ollama pull llama3\n\
uvicorn main:app --host 0.0.0.0 --port $PORT\n\
' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"]
