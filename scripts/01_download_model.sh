#!/usr/bin/env bash
# 下载 SAM 家族权重
#
# 用法: bash 01_download_model.sh <sam3|sam3.1|sam2.1|sam|medsam> [目标根目录]
# 目标根目录默认 $MODEL_DEST (默认 /data1/models)
#
# 说明:
#   sam3/sam3.1 走 ModelScope; 其余 HF 权重走 hf-mirror.com (curl, 支持断点续传)
#   sam3        -> HF safetensors, from_pretrained 直接可用 (推荐)
#   sam3.1      -> Meta 原始格式 multiplex.pt, 需先转换 (scripts/08_convert_sam31.py)
set -euo pipefail

KIND=${1:-sam3}
ROOT=${2:-${MODEL_DEST:-/data1/models}}

HF=https://hf-mirror.com

fetch_hf() {  # fetch_hf <repo> <dest_dir> <file...>
    local repo=$1 dest=$2; shift 2
    mkdir -p "$dest"
    for f in "$@"; do
        for i in 1 2 3; do
            curl -fSL --max-time 3600 --retry 3 --retry-delay 5 -C - \
                "$HF/$repo/resolve/main/$f" -o "$dest/$f" && break || sleep 5
        done
    done
}

case "$KIND" in
  sam3)
    DEST=$ROOT/SAM3
    mkdir -p "$DEST"
    modelscope download --repo-type model facebook/sam3 \
        --include model.safetensors config.json processor_config.json \
                 tokenizer.json tokenizer_config.json special_tokens_map.json \
                 vocab.json merges.txt assets \
        --local_dir "$DEST"
    ;;
  sam3.1)
    DEST=$ROOT/SAM3.1
    mkdir -p "$DEST"
    modelscope download --repo-type model facebook/sam3.1 \
        --local_dir "$DEST"
    echo "原始格式需转换:"
    echo "  python scripts/08_convert_sam31.py --checkpoint $DEST/sam3.1_multiplex.pt \\"
    echo "      --config-dir $DEST --output $ROOT/SAM3.1-hf"
    ;;
  sam2.1)
    fetch_hf facebook/sam2.1-hiera-large "$ROOT/SAM2.1" \
        config.json preprocessor_config.json processor_config.json model.safetensors
    ;;
  sam)
    fetch_hf facebook/sam-vit-huge "$ROOT/SAM-vit-h" \
        config.json preprocessor_config.json model.safetensors
    ;;
  medsam)
    # HF 格式镜像 (官方 bowang-lab/MedSAM 只发原始 .pt)
    fetch_hf flaviagiammarino/medsam-vit-base "$ROOT/MedSAM" \
        config.json preprocessor_config.json pytorch_model.bin
    ;;
  *)
    echo "unknown kind: $KIND (support: sam3|sam3.1|sam2.1|sam|medsam)"; exit 1
    ;;
esac

echo "=== $KIND done ==="
ls -lh "$DEST" 2>/dev/null || true
