FROM python:3.12.6-slim@sha256:ad48727987b259854d52241fac3bc633574364867b8e20aec305e6e7f4028b26
WORKDIR /app
# Exact, tested dependency versions (includes PyMuPDF, AGPL-3.0: see the license section of the README)
COPY track_2a/requirements-lock.txt /app/requirements-lock.txt
RUN pip install --no-cache-dir -r /app/requirements-lock.txt
# Explicit copies exclude evaluation gold, private journals, local caches and secrets.
COPY track_2a/src /app/src
COPY track_2a/vendor /app/vendor
COPY track_2a/tests /app/tests
COPY track_2a/scripts /app/scripts
COPY track_2a/data/booklets /app/data/booklets
# Source-only demo inputs: the development dataset's gold labels stay outside
# every image layer while the dashboard's existing demo path remains usable.
COPY track_2a/data/demo_cases.jsonl /app/data/demo_dataset.jsonl
COPY track_2a/app.py track_2a/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh
# Build time only (network allowed): fetch the 60 booklets of the OST dataset and pre-parse them, so a known booklet
# costs no parsing at evaluation. No "|| true" on warm-cache: a broken booklet only warns, a code error fails the build
# One layer: the downloaded dataset (with its labels) is deleted before the layer is written, so the image holds
# only sources (booklets and their parse cache)
RUN (python -m src.hf_dataset || true)     && BOOKLET_CACHE_DIR=/app/.cache/booklets python -m src warm-cache     && rm -rf /app/data/hf
ARG GIT_COMMIT=unknown
# Evaluation contract: /data is read-only and caches belong in /tmp; the build-time cache is only read
ENV GIT_COMMIT=${GIT_COMMIT} PYTHONUNBUFFERED=1 BOOKLET_CACHE_DIR=/tmp/fact-attack/booklets     BOOKLET_CACHE_PREBUILT=/app/.cache/booklets
EXPOSE 8501
ENV PORT=8501
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
