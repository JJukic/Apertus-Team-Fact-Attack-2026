#!/bin/sh
set -e

# Hack Apertus Track 2A - Universal Container Entrypoint
# Supports:
#   docker run <image> benchmark [args]  -> runs benchmark
#   docker run <image> compare [args]    -> runs strategy comparison
#   docker run <image> predict [args]    -> runs single claim verification
#   docker run <image> web               -> launches Streamlit dashboard
#   docker run <image> test              -> runs test suite
#   docker run <image>                   -> displays CLI help

if [ "$1" = "streamlit" ] || [ "$1" = "web" ]; then
    exec streamlit run app.py --server.port="${PORT:-8501}" --server.address=0.0.0.0
elif [ "$1" = "benchmark" ]; then
    shift
    exec python -m src benchmark "$@"
elif [ "$1" = "compare" ]; then
    shift
    exec python -m src compare "$@"
elif [ "$1" = "predict" ]; then
    shift
    exec python -m src predict "$@"
elif [ "$1" = "download" ]; then
    shift
    exec python -m src download "$@"
elif [ "$1" = "test" ]; then
    shift
    exec python -m unittest discover -s tests -p "test_*.py"
elif [ "$1" = "python" ] || [ "$1" = "sh" ] || [ "$1" = "bash" ]; then
    exec "$@"
elif [ "$#" -eq 0 ]; then
    exec python -m src --help
else
    exec python -m src "$@"
fi
