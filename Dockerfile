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

EXPOSE 8501

ENV PORT=8501
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["benchmark"]
