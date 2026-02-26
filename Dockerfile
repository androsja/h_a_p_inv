# ══════════════════════════════════════════════════════════════
#  Stage 1: Builder
# ══════════════════════════════════════════════════════════════
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ══════════════════════════════════════════════════════════════
#  Stage 2: Runtime
# ══════════════════════════════════════════════════════════════
FROM python:3.12-slim AS runtime
LABEL maintainer="Hapi Trading Bot"
LABEL description="Bot de scalping algorítmico para Hapi Trade - Colombia"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
RUN mkdir -p logs data/cache

CMD ["python", "main.py", "--mode", "SIMULATED"]

