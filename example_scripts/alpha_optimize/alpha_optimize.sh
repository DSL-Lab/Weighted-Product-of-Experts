#!/bin/bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: alpha_optimize_vallina_transformers.sh --dim <int> --model_type <name> --dataset <name> [options]

Required arguments:
  --dim <int>           Embedding dimension (e.g., 64, 128, 256)
  --model_type <name>   Model type passed to weighted_product_of_experts.py
  --dataset <name>      Dataset name (e.g., enwik8, enwik9, math, code, shakespeare)

Optional overrides:
  --iteration <int>     Number of training iterations (default: 1000)
  --batch-size <int>    Batch size (default: 4)
  --learning-rate <lr>  Learning rate (default: 5e-3)
  --load-model <path>   Model checkpoint path (default: ckpt/pretrained_enwik8_dim<DIM>.pth)
  -h, --help            Show this help message
EOF
}

dim="64"
model_type=""
dataset=""
iteration="1000"
batch_size="4"
learning_rate="5e-3"
load_model=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dim)
      dim="$2"
      shift 2
      ;;
    --model_type)
      model_type="$2"
      shift 2
      ;;
    --dataset)
      dataset="$2"
      shift 2
      ;;
    --iteration)
      iteration="$2"
      shift 2
      ;;
    --batch-size)
      batch_size="$2"
      shift 2
      ;;
    --learning-rate)
      learning_rate="$2"
      shift 2
      ;;
    --load-model)
      load_model="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$dim" || -z "$model_type" || -z "$dataset" ]]; then
  echo "Error: --dim, --model_type, and --dataset are required."
  usage
  exit 1
fi

if [[ -z "$load_model" ]]; then
  load_model="ckpt/pretrained_enwik8_dim${dim}.pth"
fi

uv run weighted_product_of_experts.py \
  --load_model "$load_model" \
  --embedding_dim "$dim" \
  --model_type "$model_type" \
  --dataset "$dataset" \
  --iteration "$iteration" \
  --batch_size "$batch_size" \
  --learning_rate "$learning_rate"
