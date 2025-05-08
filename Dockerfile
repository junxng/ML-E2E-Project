FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Set up virtual environment with uv
RUN uv venv --python 3.11 /app/.venv

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy project files for dependency installation
COPY pyproject.toml uv.lock ./

# Install dependencies with uv
RUN uv pip install -e .

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p pdfs logs qdrant_storage

# Expose port
EXPOSE 8000

# Command to run the application with uv
CMD ["uv", "run", "fastapi", "run", "app/main.py"]
