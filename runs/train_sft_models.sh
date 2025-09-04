#!/bin/bash
set -e
CONFIG_LIST=(
#"configs/generated/skommarkhos_croissantllmchatv0_1_sft_v1.json"
#"configs/generated/skommarkhos_croissantllmchatv0_1_sft_v2.json"
#"configs/generated/skommarkhos_croissantllmchatv0_1_sft_v3.json"
#"configs/generated/skommarkhos_croissantllmchatv0_1_sft_v4.json"
#"configs/generated/skommarkhos_croissantllmchatv0_1_sft_v5.json"
#"configs/generated/skommarkhos_croissantllmchatv0_1_sft_v6.json"
#"configs/generated/skommarkhos_croissantllmchatv0_1_sft_v7.json"
#"configs/generated/skommarkhos_croissantllmchatv0_1_sft_v8.json"
"configs/generated/skommarkhos_lucie7binstructv1_1_sft_v1.json"
"configs/generated/skommarkhos_lucie7binstructv1_1_sft_v2.json"
"configs/generated/skommarkhos_lucie7binstructv1_1_sft_v3.json"
"configs/generated/skommarkhos_lucie7binstructv1_1_sft_v4.json"
"configs/generated/skommarkhos_lucie7binstructv1_1_sft_v5.json"
"configs/generated/skommarkhos_lucie7binstructv1_1_sft_v6.json"
"configs/generated/skommarkhos_lucie7binstructv1_1_sft_v7.json"
"configs/generated/skommarkhos_lucie7binstructv1_1_sft_v8.json"
)

for config in "${CONFIG_LIST[@]}"; do
  echo "🚀 Training SFT model with config: $config"
  python src/sft.py --config-file "$config"
done