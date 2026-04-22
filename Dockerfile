FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data cache directory
RUN mkdir -p data_cache

# Environment defaults (override via Cloud Run env vars)
ENV APP_ENV=production
ENV LLM_PROVIDER=gemini
ENV GEMINI_MODEL=gemini-flash-latest
ENV PYTHONUNBUFFERED=1

# Expose port (Cloud Run sets $PORT automatically)
EXPOSE 8080

# Start with gunicorn
CMD exec gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
