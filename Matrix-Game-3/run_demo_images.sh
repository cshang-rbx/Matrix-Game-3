#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

CKPT_DIR="${CKPT_DIR:-Matrix-Game-3.0}"
DEMO_DIR="${DEMO_DIR:-demo_images}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_demo_images_ori}"

GPUS="${GPUS:-6,7}"
NPROC="$(awk -F',' '{print NF}' <<< "$GPUS")"

NUM_ITERATIONS="${NUM_ITERATIONS:-13}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-3}"
SEED="${SEED:-42}"
SIZE="${SIZE:-704*1280}"
MASTER_PORT="${MASTER_PORT:-29500}"

mkdir -p "$OUTPUT_DIR"

for subdir in "$DEMO_DIR"/*/; do
  name="$(basename "$subdir")"
  image="$subdir/image.png"
  prompt_file="$subdir/prompt.txt"

  if [[ ! -f "$image" ]]; then
    echo "[skip] $name: no image.png"; continue
  fi
  if [[ ! -f "$prompt_file" ]]; then
    echo "[skip] $name: no prompt.txt"; continue
  fi

  prompt="$(< "$prompt_file")"

  if [[ -f "$OUTPUT_DIR/${name}.mp4" ]]; then
    echo "[skip] $name: already generated"; continue
  fi

  echo "=============================================="
  echo "[run ] $name on GPUs $GPUS"
  echo "  image : $image"
  echo "  prompt: $prompt"
  echo "=============================================="

  CUDA_VISIBLE_DEVICES="$GPUS" \
  torchrun \
    --nproc_per_node="$NPROC" \
    --master_port="$MASTER_PORT" \
    generate.py \
      --size "$SIZE" \
      --ulysses_size "$NPROC" \
      --dit_fsdp \
      --t5_fsdp \
      --ckpt_dir "$CKPT_DIR" \
      --fa_version 3 \
      --use_int8 \
      --num_iterations "$NUM_ITERATIONS" \
      --num_inference_steps "$NUM_INFERENCE_STEPS" \
      --image "$image" \
      --prompt "$prompt" \
      --save_name "$name" \
      --seed "$SEED" \
      --compile_vae \
      --lightvae_pruning_rate 0.5 \
      --vae_type mg_lightvae \
      --output_dir "$OUTPUT_DIR"
done

echo "All done. Videos saved to: $OUTPUT_DIR"
