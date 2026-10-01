.PHONY: run build test benchmark compare download install

IMAGE_NAME ?= hackapertus-track2a

# Judges run `make run` from the root of the project.
# It builds and executes the Docker container, passing required environment variables.
run:
	docker build -t $(IMAGE_NAME) .
	docker run --rm \
		-e LLM_NAME="$${LLM_NAME:-swiss-ai/Apertus-8B-Instruct}" \
		-e LLM_BASE_URL="$${LLM_BASE_URL:-https://api.cscs.ch/v1}" \
		-e LLM_API_KEY="$${LLM_API_KEY}" \
		-e NLI_STRATEGY="$${NLI_STRATEGY:-retrieval}" \
		$(IMAGE_NAME) benchmark

# Local helper targets (without Docker)
install:
	pip install -r requirements.txt

download:
	python src/download_data.py

benchmark:
	python -m src benchmark

compare:
	python -m src compare
