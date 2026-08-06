#!/usr/bin/env bash
set -euo pipefail

# Fair Base-vs-GRPO-LoRA evaluation for the Qwen-Image-Edit-2509 checkpoint.
#
# Usage:
#   bash inference/run_qwen_grpo_checkpoint20_test.sh /path/to/moire.png
#   bash inference/run_qwen_grpo_checkpoint20_test.sh /path/to/test.jsonl /path/to/output
#
# Optional overrides:
#   MODEL_PATH=/path/to/base/model
#   LORA_DIR=/path/to/checkpoint
#   NUM_CANDIDATES=4
#   PRESERVE_RESOLUTION=1
#   IMAGE_ROOT=/path/to/images

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 INPUT_IMAGE_OR_JSONL [OUTPUT_DIR]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_SOURCE="$1"
OUTPUT_DIR="${2:-${REPO_ROOT}/outputs/grpo_checkpoint20_base_vs_lora}"
MODEL_PATH="${MODEL_PATH:-/data/ckpts/Qwen/Qwen-Image-Edit-2509/}"
LORA_DIR="${LORA_DIR:-/data_mount_162/zhaoqianqian/code/Flow-Factory/saves/qwen-image-edit-plus_lora_grpo_20260803_154554/checkpoints/checkpoint-20}"
NUM_CANDIDATES="${NUM_CANDIDATES:-4}"
PRESERVE_RESOLUTION="${PRESERVE_RESOLUTION:-0}"

if [[ ! -f "${INPUT_SOURCE}" ]]; then
  echo "Input image or JSONL does not exist: ${INPUT_SOURCE}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Base model directory does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -d "${LORA_DIR}" ]]; then
  echo "LoRA checkpoint directory does not exist: ${LORA_DIR}" >&2
  exit 1
fi

if [[ ! -f "${LORA_DIR}/adapter_model.safetensors" && ! -f "${LORA_DIR}/transformer/adapter_model.safetensors" ]]; then
  echo "Missing adapter_model.safetensors under ${LORA_DIR} or ${LORA_DIR}/transformer" >&2
  exit 1
fi

if [[ ! -f "${LORA_DIR}/adapter_config.json" && ! -f "${LORA_DIR}/transformer/adapter_config.json" ]]; then
  echo "Missing adapter_config.json under ${LORA_DIR} or ${LORA_DIR}/transformer" >&2
  echo "Please list the checkpoint directory before testing." >&2
  exit 1
fi

SIZE_ARGS=(--resolution 512 --auto_resize)
if [[ "${PRESERVE_RESOLUTION}" == "1" ]]; then
  SIZE_ARGS=(--preserve_resolution)
fi

SOURCE_ARGS=(--input_image "${INPUT_SOURCE}")
if [[ "${INPUT_SOURCE}" == *.jsonl ]]; then
  SOURCE_ARGS=(--jsonl "${INPUT_SOURCE}")
  if [[ -n "${IMAGE_ROOT:-}" ]]; then
    SOURCE_ARGS+=(--image_root "${IMAGE_ROOT}")
  fi
fi

cd "${REPO_ROOT}"

python tools/infer_qwen_demoire_candidates.py \
  --mode both \
  --model_path "${MODEL_PATH}" \
  --lora_path "${LORA_DIR}" \
  "${SOURCE_ARGS[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_candidates "${NUM_CANDIDATES}" \
  --bank_version "grpo-checkpoint-20" \
  --seed 42 \
  --increment_seed \
  "${SIZE_ARGS[@]}" \
  --steps 20 \
  --true_cfg_scale 4.0

echo
echo "Evaluation finished."
echo "Base results:       ${OUTPUT_DIR}/base"
echo "GRPO-LoRA results:  ${OUTPUT_DIR}/lora"
echo "Comparison images:  ${OUTPUT_DIR}/comparison"
echo "Metadata:           ${OUTPUT_DIR}/metadata.jsonl"
