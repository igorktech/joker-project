#!/bin/bash
set -e
CONFIG_LIST=(
#"configs/generated/skommarkhos_lucie7binstructv1_1_sft_arpo_a1.json"
#"configs/generated/skommarkhos_lucie7binstructv1_1_sft_arpo_a2.json"
#"configs/generated/skommarkhos_lucie7binstructv1_1_sft_arpo_a3.json"
#"configs/generated/skommarkhos_lucie7binstructv1_1_sft_arpo_a4.json"
#"configs/generated/skommarkhos_lucie7binstructv1_1_sft_arpo_a5.json"
#"configs/generated/skommarkhos_lucie7binstructv1_1_sft_arpo_a6.json"
#"configs/generated/skommarkhos_lucie7binstructv1_1_sft_arpo_a7.json"
#"configs/generated/skommarkhos_lucie7binstructv1_1_sft_arpo_a8.json"
#"configs/generated/skommarkhos_lucie7binstructv1_1_sft_arpo_a9.json"
#"configs/generated/skommarkhos_lucie7binstructv1_1_sft_arpo_a10.json"
#"configs/generated/skommarkhos_lucie7binstructv1_1_sft_arpo_a11.json"
#"configs/generated/skommarkhos_lucie7binstructv1_1_sft_arpo_a12.json"
configs/generated/gpt/skommarkhos_lucie7binstructv1_1_sft_arpo_a13.json
configs/generated/gpt/skommarkhos_lucie7binstructv1_1_sft_arpo_a14.json
configs/generated/gpt/skommarkhos_lucie7binstructv1_1_sft_arpo_a15.json
configs/generated/gpt/skommarkhos_lucie7binstructv1_1_sft_arpo_a19.json
configs/generated/gpt/skommarkhos_lucie7binstructv1_1_sft_arpo_a20.json
configs/generated/gpt/skommarkhos_lucie7binstructv1_1_sft_arpo_a21.json
)

for config in "${CONFIG_LIST[@]}"; do
  echo "🚀 Training ARPO model with config: $config"
  python src/arpo.py --config-file "$config"
done
