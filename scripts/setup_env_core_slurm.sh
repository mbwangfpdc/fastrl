#!/bin/bash
#
# Oscar CPU-only job: build just the main fastrl/.venv (sglang + verl +
# flash-attn, plus skyrl-gym) -- steps 1-2 of setup_env_slurm.sh, without the
# eagle-train venv or the SQL data regeneration. Enough to run examples/grpo_7B.sh.
#
#   sbatch scripts/setup_env_core_slurm.sh
#
#SBATCH --partition=batch
#SBATCH --job-name=fastrl-env-core
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail
REPO=/oscar/data/deeptir/mborjigi/fastrl
cd "$REPO"
echo "node=$(hostname)"

unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT || true
export UV_CACHE_DIR="$REPO/.cache/uv"
export HF_HOME=${HF_HOME:-/users/mborjigi/data/mborjigi/hf}
mkdir -p "$UV_CACHE_DIR"

echo "=== 1/2: fastrl/.venv (sglang + verl + flash-attn) ==="
bash "$REPO/install_uv.sh"

echo "=== 2/2: skyrl-gym ==="
VIRTUAL_ENV="$REPO/.venv" uv pip install --python "$REPO/.venv/bin/python" skyrl-gym==0.3.0

"$REPO/.venv/bin/python" -c "import verl, sglang, torch; print('ok: torch', torch.__version__, 'sglang', sglang.__version__)"
