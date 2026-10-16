FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# libgomp1 dibutuhkan XGBoost; install dependensi dulu supaya layer ini ter-cache
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sepsis_ew/ ./sepsis_ew/
COPY api/ ./api/
COPY artifacts/ ./artifacts/

# Jalankan sebagai user biasa, bukan root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
