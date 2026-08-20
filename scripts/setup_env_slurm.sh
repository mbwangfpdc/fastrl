#!/bin/bash
#
# Oscar CPU-only job: build both venvs this checkout needs (fastrl/.venv,
# eagle-train/.venv) and regenerate the SkyRL-SQL parquet data in FastRL's
# format. No GPU required -- installs/imports only; see
# RESUME_TLT_DRAFTER.md for the GPU runs that come after this.
#
# This node (Oscar) has no /local_nvme*; INSTALL-uv.md and
# eagle-train/install_uv.sh were written on a different box, so cache/HF dirs
# are overridden below to live under /oscar/scratch.
#
#   sbatch scripts/setup_env_slurm.sh
#
# Rerunning is safe: install_uv.sh / eagle-train/install_uv.sh both skip
# venv creation if the directory already exists, and re-resolve on top.

#SBATCH --partition=batch
#SBATCH --job-name=fastrl-env-setup
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mbwang@brown.edu

set -euo pipefail

REPO=/oscar/scratch/mborjigi/fastrl
cd "$REPO"

echo "node=$(hostname) nproc_all=$(nproc --all)"

# Neither install script must ever land in an inherited $VIRTUAL_ENV -- shells
# on this box often have granular-cais-rl/.venv activated, and an unqualified
# `uv pip install` there would downgrade its pinned torch under a live run.
# Both install_uv.sh scripts already guard against this by setting
# VIRTUAL_ENV explicitly rather than falling back to it, but unset it here too
# so nothing in between (this script, uv itself) can pick it up by accident.
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT || true

# /local_nvme1 (the box INSTALL-uv.md was written on) does not exist on
# Oscar -- point the uv cache and HF cache at scratch instead.
export UV_CACHE_DIR="$REPO/.cache/uv"
export HF_HOME=${HF_HOME:-/users/mborjigi/data/mborjigi/hf}
mkdir -p "$UV_CACHE_DIR"

echo "=== 1/4: fastrl/.venv (sglang + verl + flash-attn) ==="
bash "$REPO/install_uv.sh"

echo "=== 2/4: skyrl-gym (one-time dep for the SQL interaction) ==="
VIRTUAL_ENV="$REPO/.venv" uv pip install --python "$REPO/.venv/bin/python" skyrl-gym==0.3.0

echo "=== 3/4: eagle-train/.venv (isolated: torch 2.6 / transformers 4.51) ==="
bash "$REPO/eagle-train/install_uv.sh"

echo "=== 4/4: regenerate SkyRL-SQL parquet in FastRL/verl format ==="
DATA_ROOT=/users/mborjigi/data/datasets/skyrl_sql
[ -d "$DATA_ROOT" ] || { echo "DATA_ROOT not found: $DATA_ROOT" >&2; exit 1; }
mkdir -p data/skyrl_sql
"$REPO/.venv/bin/python" examples/sql/prepare_skyrl_sql_data.py \
    --input    "$DATA_ROOT/train.parquet" \
    --output   data/skyrl_sql/train.parquet \
    --db-path  "$DATA_ROOT/data" \
    --split    train
"$REPO/.venv/bin/python" examples/sql/prepare_skyrl_sql_data.py \
    --input    "$DATA_ROOT/validation.parquet" \
    --output   data/skyrl_sql/validation.parquet \
    --db-path  "$DATA_ROOT/data" \
    --split    validation

echo "=== sanity: CPU-only interaction test (no GPU/model) ==="
"$REPO/.venv/bin/python" examples/sql/test_sql_interaction.py \
    --dataset "$DATA_ROOT/train.parquet" \
    --db-path "$DATA_ROOT/data"

echo "=== done ==="
echo "fastrl venv:      $REPO/.venv"
echo "eagle-train venv: $REPO/eagle-train/.venv"
echo "data:             $REPO/data/skyrl_sql/{train,validation}.parquet"
