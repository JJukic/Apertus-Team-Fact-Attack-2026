FROM python:3.12.6-slim@sha256:ad48727987b259854d52241fac3bc633574364867b8e20aec305e6e7f4028b26
WORKDIR /app
# Exact, tested dependency versions (no PyMuPDF: AGPL, and not needed by the submission)
COPY requirements-lock.txt /app/requirements-lock.txt
RUN pip install --no-cache-dir -r /app/requirements-lock.txt
# Explicit copies exclude evaluation gold, private journals, local caches and secrets.
COPY src /app/src
COPY vendor /app/vendor
COPY tests /app/tests
COPY scripts /app/scripts
COPY data/booklets /app/data/booklets
COPY data/demo_dataset.jsonl /app/data/demo_dataset.jsonl
COPY app.py entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh
# Build time only (network allowed): fetch the 60 booklets of the OST dataset and pre-parse them, so a known booklet
# costs no parsing at evaluation. No "|| true" on warm-cache: a broken booklet only warns, a code error fails the build
RUN python -m src.hf_dataset || true
RUN BOOKLET_CACHE_DIR=/app/.cache/booklets python -m src warm-cache
ARG GIT_COMMIT=unknown
# Evaluation contract: /data is read-only and caches belong in /tmp; the build-time cache is only read
ENV GIT_COMMIT=${GIT_COMMIT} PYTHONUNBUFFERED=1 BOOKLET_CACHE_DIR=/tmp/fact-attack/booklets     BOOKLET_CACHE_PREBUILT=/app/.cache/booklets
EXPOSE 8501
ENV PORT=8501
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
