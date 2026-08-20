#!/bin/bash
# Cheap diagnostic: does a gpu-he node have outbound internet? The e2e smoke
# job (5105294) hung silently for a full hour right after Ray came up, with
# zero further output -- a classic signature of a network call that hangs
# instead of failing fast (e.g. HF Hub's revision/etag check), rather than a
# crash. Check before burning another GPU-hour blind.

#SBATCH --qos=gpu-he+
#SBATCH --job-name=fastrl-net-check
#SBATCH --partition=gpu-he
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --mem=8g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=%x-%j.out

echo "node=$(hostname)"
for url in https://huggingface.co https://pypi.org https://github.com; do
  echo "--- $url ---"
  curl -m 8 -sI "$url" | head -1 || echo "FAILED/TIMEOUT: $url"
done
echo "=== done ==="
