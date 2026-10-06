FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ARG INSTALL_HYBRID=false
COPY requirements-hybrid.txt .
# Opt-in CPU PyTorch; never download model weights during image build.
RUN if [ "$INSTALL_HYBRID" = "true" ]; then \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-hybrid.txt; fi

# Copy code and data
COPY . .

# Ensure entrypoint is executable
RUN chmod +x entrypoint.sh

# Use the supplied data snapshot. Download missing PDFs explicitly with the CLI.

EXPOSE 8501

ENV PORT=8501
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["benchmark"]
