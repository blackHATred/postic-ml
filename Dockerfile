ARG PYTHON="3.12.10"

# tmp stage
FROM python:${PYTHON}-slim AS python_image

# tmp stage
FROM python_image AS builder

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/app/wheels \
    pip wheel --no-cache-dir --no-deps \
    --wheel-dir /app/wheels -r requirements.txt

# final stage
FROM python_image

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

COPY synt_ticket_model_weights.pth .
COPY main.py .
COPY gcs.py .
ENV MAX_TOKENS=4096
CMD ["python", "main.py"]
