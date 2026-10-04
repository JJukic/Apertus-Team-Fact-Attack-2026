FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and data
COPY . .

# Ensure entrypoint is executable
RUN chmod +x entrypoint.sh

# Ensure data and official booklets are present (offline safe)
RUN python -m src.download_data || true

# Pre-parse the booklets so the first claim per booklet does not pay for PDF parsing
# Also fetch all 60 booklets of the OST dataset so known booklets are pre-parsed (falls back silently offline)
RUN python -m src.hf_dataset || true
RUN python -m src warm-cache || true

EXPOSE 8501

ENV PORT=8501
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["benchmark"]
