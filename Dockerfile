FROM python:3.12.6-slim@sha256:ad48727987b259854d52241fac3bc633574364867b8e20aec305e6e7f4028b26
WORKDIR /app
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
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT} PYTHONUNBUFFERED=1 BOOKLET_CACHE_DIR=/tmp/fact-attack/booklets
EXPOSE 8501
ENV PORT=8501
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
