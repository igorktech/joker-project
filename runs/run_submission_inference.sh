#!/bin/bash
set -e

TEST_DATA="data/task2/joker_pun_translation_2025_test.json"

MODEL_LIST=(
  "/workspace/models/skommarkhos_lucie7binstructv1_1_sft_arpo_a19"
#  "/workspace/models/skommarkhos_lucie7binstructv1_1_sft_v4"
#  "/workspace/models/skommarkhos_lucie7binstructv1_1_sft_v8"
#  "igorktech/skommarkhos-lucie7binstructv1-1-sft-arpo-a1"
#  "igorktech/skommarkhos-lucie7binstructv1-1-sft-arpo-a5"
#  "igorktech/skommarkhos-lucie7binstructv1-1-sft-arpo-a7"
#  "igorktech/skommarkhos-lucie7binstructv1-1-sft-arpo-a11"
)

for model_path in "${MODEL_LIST[@]}"; do
  echo "🚀 Processing model: $model_path"
  python src/run_submission_inference.py run --test_data="$TEST_DATA" --model_path="$model_path"
done
