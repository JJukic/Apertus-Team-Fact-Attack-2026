#!/bin/bash
#SBATCH --job-name=apertus-adaptive-prepare
#SBATCH --account=g234
#SBATCH --qos=large-sc-qos
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

# Submit from the private staged directory with CSCS_WORK_DIR and
# CSCS_BASE_COMMIT exported. The job prepares sources only; no API key needed.
set -euo pipefail
ulimit -c 0
cd "${CSCS_WORK_DIR:?Set CSCS_WORK_DIR to the staged directory}"
export HF_HOME="$CSCS_WORK_DIR/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LLM_NAME=swiss-ai/Apertus-v1.5-70B-thinking
export EMBEDDING_BATCH_SIZE=4
export OMP_NUM_THREADS=8
export CSCS_RESULT_DIR="${CSCS_RESULT_DIR:-results}"
unset LLM_API_KEY

srun --unbuffered uenv --repo="$CSCS_WORK_DIR/uenv" run c05f143d5fbf0927 --view=default -- bash -c '
set -euo pipefail
venv/bin/python - <<"PY"
import json, socket, torch
from pathlib import Path
assert torch.cuda.is_available(), "GPU runtime is unavailable"
props = torch.cuda.get_device_properties(0)
data = {"hostname":socket.gethostname(), "torch":torch.__version__,
        "cuda":torch.version.cuda, "gpu":props.name, "gpu_memory_bytes":props.total_memory}
Path("logs/gpu_runtime.json").write_text(json.dumps(data, indent=2)+"\n")
print(json.dumps(data), flush=True)
PY
venv/bin/python -u repo/track_2a/scripts/run_adaptive_comparison.py prepare \
  --output "$CSCS_RESULT_DIR" --device cuda --source-commit "$CSCS_BASE_COMMIT"
'
