#!/usr/bin/env bash
set -e

# Repo root derived from this script's location (scripts/ is one level down).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Activate a virtualenv unless one is already active. Override with PAROQUANT_VENV.
PAROQUANT_VENV="${PAROQUANT_VENV:-$HOME/.venv}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$PAROQUANT_VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$PAROQUANT_VENV/bin/activate"
fi

# nix-ld LD_LIBRARY_PATH: auto-detect the share/nix-ld/lib dir, allow override.
if [ -n "${PAROQUANT_LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$PAROQUANT_LD_LIBRARY_PATH"
else
  for _ldp in /run/host/nix/store/*-ld-library-path/share/nix-ld/lib \
              /nix/store/*-ld-library-path/share/nix-ld/lib; do
    if [ -d "$_ldp" ]; then export LD_LIBRARY_PATH="$_ldp"; break; fi
  done
fi

export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_TRUST_REMOTE_CODE=1
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HOME/.cache/huggingface/modules_new}"
mkdir -p "$HF_MODULES_CACHE"

# Model, spill dir and output dir are configurable. Model is required.
MODEL="${1:-${PAROQUANT_MODEL:?set PAROQUANT_MODEL or pass the model path as the first argument}}"
SPILL_DIR="${PAROQUANT_SPILL_DIR:-$HOME/paroquant-spill}"
OUTPUT_DIR="${PAROQUANT_OUTPUT_DIR:-$REPO_ROOT/output/MiniMax-M2.7}"
PARO_OUTPUT_DIR="${PAROQUANT_PARO_OUTPUT_DIR:-${OUTPUT_DIR}-PARO}"
mkdir -p "$SPILL_DIR"

pip install -q -e "$REPO_ROOT" 2>/dev/null || true
pip install -q simple-parsing 2>/dev/null || true

echo "=== ParoQuant Optimize: MiniMax-M2.7 ==="
echo "Started at $(date)"
echo "venv: $(which python3) — $(python3 -c 'import transformers; print(transformers.__version__)')"
python3 -m paroquant.cli.optimize \
  --model "$MODEL" \
  --params "channel_scales:0.05,angles:0.05" "weight:1e-5,quantizer:1e-6" \
  --epochs 5 5 \
  --group-size 128 \
  --n-bit 4 \
  --num-rotations 8 \
  --datasets wikitext2 c4 redpajama \
  --val-dataset pileval \
  --train-size 2048 \
  --validation-size 64 \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --seqlen 2048 \
  --cache-shards 4 \
  --activation-spill-dir "$SPILL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --stream-from-disk \
  --resume \
  --seed 0
echo "Optimize exited at $(date) with code $?"

echo "=== Convert to safetensors ==="
python3 -m paroquant.cli.convert \
  --model "$MODEL" \
  --result-dir "$OUTPUT_DIR" \
  --output-path "$PARO_OUTPUT_DIR" \
  --mode real \
  --stream-from-disk
echo "Convert exited at $(date) with code $?"

echo "=== Final size ==="
du -sh "$PARO_OUTPUT_DIR/"
echo "DONE at $(date)"
