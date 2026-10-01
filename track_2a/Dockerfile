FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and data
COPY src/ ./src/
COPY data/ ./data/

# Default entrypoint runs the benchmark
ENTRYPOINT ["python", "-m", "src"]
CMD ["benchmark"]
