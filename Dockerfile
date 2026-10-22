# TwinSim: backend + prebuilt frontend in one container.
# Build the frontend first (npm run build) so dist/ exists, then: docker build -t twinsim .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TWINSIM_HOME=/data \
    TWINSIM_PUBLIC=1

WORKDIR /app
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn>=0.29" "pydantic>=2.6" "numpy>=1.26" "psutil>=5.9"

COPY backend /app/backend
COPY dist /app/dist

# Trained models and the research database live in /data (mount a volume there).
VOLUME ["/data"]
EXPOSE 8000
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
