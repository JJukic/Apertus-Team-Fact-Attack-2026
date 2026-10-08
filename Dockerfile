FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY track_2a/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and assets from track_2a
COPY track_2a/ .

# Ensure entrypoint is executable
RUN chmod +x entrypoint.sh

# Ensure official voting booklets are downloaded and available offline
RUN python -m src.download_data || true

# Pre-parse the booklets so the first claim per booklet does not pay for PDF parsing
# Also fetch all 60 booklets of the OST dataset so known booklets are pre-parsed (falls back silently offline)
RUN python -m src.hf_dataset || true
# No "|| true" here: a broken booklet only warns, but a code error must fail the build
RUN BOOKLET_CACHE_DIR=/app/.cache/booklets python -m src warm-cache

EXPOSE 8501

ENV PORT=8501
ENV PYTHONUNBUFFERED=1
# Evaluation contract: /data is read-only and caches belong in /tmp; the build-time cache is only read
ENV BOOKLET_CACHE_DIR=/tmp/fact-attack/booklets
ENV BOOKLET_CACHE_PREBUILT=/app/.cache/booklets

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["benchmark"]
