# syntax=docker/dockerfile:1
# Stage 1: builder — install all Python dependencies
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build tools needed by some packages (e.g. faiss-cpu, scipy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: runtime — lean image, non-root user
FROM python:3.10-slim AS runtime

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application source and config only — models are mounted as a volume
COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml .

# models/ directory placeholder so volume mount works without error
RUN mkdir -p models/stage_a_onnx models/baseline data/faiss_index

# Own the working directory
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# TOGETHER_API_KEY must be passed at runtime via --env or --env-file
# Never bake secrets into the image
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
