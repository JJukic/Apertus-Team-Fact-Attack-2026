.PHONY: run build test benchmark compare download install web

IMAGE_NAME ?= hackapertus-track2a

# Judges run `make run` from the root of the project.
# It builds and executes the Docker container, passing required environment variables.
# The run's JSON (metrics and per-sample predictions) is written to track_2a/results/ on the host.
run:
	docker build --build-arg GIT_COMMIT="$$(git rev-parse --short HEAD 2>/dev/null || echo unknown)" -t $(IMAGE_NAME) .
	mkdir -p results
	docker run --rm \
		-v "$(CURDIR)/results:/app/results" \
		-e LLM_NAME="$${LLM_NAME:-swiss-ai/Apertus-v1.5-70B-thinking}" \
		-e LLM_BASE_URL="$${LLM_BASE_URL:-https://api.inference.cscs.ch/v1}" \
		-e LLM_API_KEY="$${LLM_API_KEY}" \
		-e NLI_STRATEGY="$${NLI_STRATEGY:-hybrid}" \
		-e NLI_TOP_K="$${NLI_TOP_K:-12}" \
		-e PROMPT_MODE="$${PROMPT_MODE:-ids}" \
		$(IMAGE_NAME) benchmark

test:
	python -m unittest discover -s tests -p "test_*.py"

# Local helper targets (without Docker)
install:
	pip install -r requirements.txt

download:
	python src/download_data.py

benchmark:
	python -m src benchmark

compare:
	python -m src compare

web:
	streamlit run app.py
