FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install .

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p ml_pipeline/data/raw ml_pipeline/data/processed ml_pipeline/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "app.model_serving:app", "--host", "0.0.0.0", "--port", "8000"]