#!/usr/bin/env bash
# 从 ModelScope 下载 SAM3 权重
# 用法: bash 01_download_model.sh [facebook/sam3|facebook/sam3.1] [目标目录]
#   facebook/sam3   —— HF 格式 model.safetensors，可直接 from_pretrained（推荐）
#   facebook/sam3.1 —— Meta 原始格式 sam3.1_multiplex.pt（键名与 HF 不同，见 docs/tutorial.md）
set -euo pipefail

REPO=${1:-facebook/sam3}
NAME=${2:-SAM3}
DEST=${MODEL_DEST:-/data1/models}/$NAME
mkdir -p "$DEST"

modelscope download --repo-type model "$REPO" \
    --include model.safetensors config.json processor_config.json \
             tokenizer.json tokenizer_config.json special_tokens_map.json \
             vocab.json merges.txt assets \
    --local_dir "$DEST"

echo "=== done, files: ==="
ls -lh "$DEST"
