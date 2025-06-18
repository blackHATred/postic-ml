ARG PYTHON="3.12.10"

# Build stage
FROM python:${PYTHON}-slim AS builder

WORKDIR /app

# Install build dependencies and clean up in one layer
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy only requirements first for better caching
COPY requirements.txt .

# Create wheels with cache mount and install in one step
RUN --mount=type=cache,target=/root/.cache/pip \
    pip wheel --no-cache-dir --no-deps \
    --wheel-dir /app/wheels -r requirements.txt

# Model download stage
FROM python:${PYTHON}-slim AS model-downloader

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

WORKDIR /app

# Install only necessary packages for model downloading
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy download script
COPY download_models.py .

# Download models and cache them - this will prevent download at runtime
RUN mkdir -p /app/models_cache && \
    python download_models.py

# Final stage
FROM python:${PYTHON}-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV FORCE_CPU=true

# Install runtime dependencies in one layer
RUN apt-get update && apt-get install -y \
    curl \
    netcat-traditional \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app

# Create non-root user early
RUN useradd --create-home --shell /bin/bash --uid 1000 app

# Install Python packages from wheels (more efficient than force-reinstall)
COPY --from=builder /app/wheels /wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps /wheels/* \
    && rm -rf /wheels

# Copy pre-downloaded models from model-downloader stage
COPY --from=model-downloader --chown=app:app /app/models_cache /app/models_cache

# Set environment variable to use cached models
ENV TRANSFORMERS_CACHE=/app/models_cache
ENV HF_HOME=/app/models_cache

# Copy model weights first (changes rarely, better caching)
COPY --chown=app:app synt_ticket_model_weights.pth .

# Copy application code in fewer layers
COPY --chown=app:app api/ ./api/
COPY --chown=app:app config/ ./config/
COPY --chown=app:app core/ ./core/
COPY --chown=app:app models/ ./models/
COPY --chown=app:app services/ ./services/
COPY --chown=app:app utils/ ./utils/
COPY --chown=app:app static/ ./static/
COPY --chown=app:app main.py .

# Switch to non-root user
USER app

# Health check with better error handling
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/docs || exit 1

EXPOSE 8000

CMD ["python", "main.py"]
