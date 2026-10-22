#!/usr/bin/env python3
"""将 SAM3.1 的 Meta 原始格式权重(sam3.1_multiplex.pt)转换为 HF transformers 格式。

基于 HuggingFace 官方转换脚本改编(Apache-2.0):
  transformers/src/transformers/models/sam3/convert_sam3_to_hf.py
差异: multiplex 权重的顶层前缀是 `detector.`/`tracker.`(官方脚本面向 `sam3_model.`),
本脚本剥离 `detector.` 前缀只转换 detector 部分(Sam3Model), tracker 部分暂不处理。

用法:
  python scripts/08_convert_sam31.py \
      --checkpoint /data1/models/SAM3.1/sam3.1_multiplex.pt \
      --config-dir /data1/models/SAM3.1 \
      --output /data1/models/SAM3.1-hf
"""
import argparse
import gc
import json
import os
import re
import shutil

import torch

# ---------------------------------------------------------------------------
# 官方键名映射 (来自 convert_sam3_to_hf.py, Apache-2.0)
# ---------------------------------------------------------------------------
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"^sam3_model\.": r"",
    r"^backbone\.vision_backbone\.trunk\.":                                 r"vision_encoder.backbone.",
    r"^vision_encoder\.backbone\.pos_embed":                                r"vision_encoder.backbone.embeddings.position_embeddings",
    r"^vision_encoder\.backbone\.patch_embed\.proj\.":                      r"vision_encoder.backbone.embeddings.patch_embeddings.projection.",
    r"^vision_encoder\.backbone\.ln_pre\.":                                 r"vision_encoder.backbone.layer_norm.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.norm1\.":                   r"vision_encoder.backbone.layers.\1.layer_norm1.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.norm2\.":                   r"vision_encoder.backbone.layers.\1.layer_norm2.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.attn\.qkv\.":               r"vision_encoder.backbone.layers.\1.attention.qkv.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.attn\.proj\.":              r"vision_encoder.backbone.layers.\1.attention.o_proj.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.attn\.freqs_cis":           r"vision_encoder.backbone.layers.\1.rotary_emb.rope_embeddings",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.mlp\.fc1\.":                r"vision_encoder.backbone.layers.\1.mlp.fc1.",
    r"^vision_encoder\.backbone\.blocks\.(\d+)\.mlp\.fc2\.":                r"vision_encoder.backbone.layers.\1.mlp.fc2.",
    r"^backbone\.vision_backbone\.neck\.fpn\.(\d+)\.":                      r"vision_encoder.neck.fpn_layers.\1.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2_0\.":             r"vision_encoder.neck.fpn_layers.\1.scale_layers.0.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2_1\.":             r"vision_encoder.neck.fpn_layers.\1.scale_layers.2.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.dconv_2x2\.":               r"vision_encoder.neck.fpn_layers.\1.scale_layers.0.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.maxpool_2x2\.":             r"vision_encoder.neck.fpn_layers.\1.scale_layers.0.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.conv_1x1\.":                r"vision_encoder.neck.fpn_layers.\1.proj1.",
    r"^backbone\.vision_backbone\.convs\.(\d+)\.conv_3x3\.":                r"vision_encoder.neck.fpn_layers.\1.proj2.",
    r"^backbone\.language_backbone\.encoder\.":                             r"text_encoder.",
    r"^text_encoder\.token_embedding\.":                                    r"text_encoder.text_model.embeddings.token_embedding.",
    r"^text_encoder\.positional_embedding":                                 r"text_encoder.text_model.embeddings.position_embedding.weight",
    r"^text_encoder\.ln_final\.":                                           r"text_encoder.text_model.final_layer_norm.",
    r"^text_encoder\.text_projection":                                      r"text_encoder.text_projection.weight",
    r"^text_encoder\.transformer\.resblocks\.(\d+)\.attn\.in_proj_":       r"text_encoder.text_model.encoder.layers.\1.self_attn.in_proj_",
    r"^text_encoder\.transformer\.resblocks\.(\d+)\.attn\.out_proj\.":     r"text_encoder.text_model.encoder.layers.\1.self_attn.out_proj.",
    r"^text_encoder\.transformer\.resblocks\.(\d+)\.ln_1\.":               r"text_encoder.text_model.encoder.layers.\1.layer_norm1.",
    r"^text_encoder\.transformer\.resblocks\.(\d+)\.ln_2\.":               r"text_encoder.text_model.encoder.layers.\1.layer_norm2.",
    r"^text_encoder\.transformer\.resblocks\.(\d+)\.mlp\.c_fc\.":          r"text_encoder.text_model.encoder.layers.\1.mlp.fc1.",
    r"^text_encoder\.transformer\.resblocks\.(\d+)\.mlp\.c_proj\.":        r"text_encoder.text_model.encoder.layers.\1.mlp.fc2.",
    r"^backbone\.language_backbone\.resizer\.":                             r"text_projection.",
    r"^geometry_encoder\.encode\.(\d+)\.cross_attn_image\.out_proj\.":     r"geometry_encoder.layers.\1.cross_attn.o_proj.",
    r"^geometry_encoder\.encode\.(\d+)\.cross_attn_image\.":               r"geometry_encoder.layers.\1.cross_attn.",
    r"^geometry_encoder\.encode\.(\d+)\.self_attn\.out_proj\.":            r"geometry_encoder.layers.\1.self_attn.o_proj.",
    r"^geometry_encoder\.encode\.(\d+)\.self_attn\.":                      r"geometry_encoder.layers.\1.self_attn.",
    r"^geometry_encoder\.encode\.(\d+)\.linear1\.":                        r"geometry_encoder.layers.\1.mlp.fc1.",
    r"^geometry_encoder\.encode\.(\d+)\.linear2\.":                        r"geometry_encoder.layers.\1.mlp.fc2.",
    r"^geometry_encoder\.encode\.(\d+)\.norm1\.":                          r"geometry_encoder.layers.\1.layer_norm1.",
    r"^geometry_encoder\.encode\.(\d+)\.norm2\.":                          r"geometry_encoder.layers.\1.layer_norm2.",
    r"^geometry_encoder\.encode\.(\d+)\.norm3\.":                          r"geometry_encoder.layers.\1.layer_norm3.",
    r"^geometry_encoder\.img_pre_norm\.":                                   r"geometry_encoder.vision_layer_norm.",
    r"^geometry_encoder\.norm\.":                                           r"geometry_encoder.prompt_layer_norm.",
    r"^geometry_encoder\.encode_norm\.":                                    r"geometry_encoder.output_layer_norm.",
    r"^transformer\.encoder\.layers\.(\d+)\.cross_attn_image\.out_proj\.":  r"detr_encoder.layers.\1.cross_attn.o_proj.",
    r"^transformer\.encoder\.layers\.(\d+)\.cross_attn_image\.":            r"detr_encoder.layers.\1.cross_attn.",
    r"^transformer\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.":         r"detr_encoder.layers.\1.self_attn.o_proj.",
    r"^transformer\.encoder\.layers\.(\d+)\.self_attn\.":                   r"detr_encoder.layers.\1.self_attn.",
    r"^transformer\.encoder\.layers\.(\d+)\.cross_attn\.out_proj\.":        r"detr_encoder.layers.\1.cross_attn.o_proj.",
    r"^transformer\.encoder\.layers\.(\d+)\.cross_attn\.":                  r"detr_encoder.layers.\1.cross_attn.",
    r"^transformer\.encoder\.layers\.(\d+)\.linear1\.":                     r"detr_encoder.layers.\1.mlp.fc1.",
    r"^transformer\.encoder\.layers\.(\d+)\.linear2\.":                     r"detr_encoder.layers.\1.mlp.fc2.",
    r"^transformer\.encoder\.layers\.(\d+)\.norm1\.":                       r"detr_encoder.layers.\1.layer_norm1.",
    r"^transformer\.encoder\.layers\.(\d+)\.norm2\.":                       r"detr_encoder.layers.\1.layer_norm2.",
    r"^transformer\.encoder\.layers\.(\d+)\.norm3\.":                       r"detr_encoder.layers.\1.layer_norm3.",
    r"^transformer\.decoder\.query_embed\.":                                r"detr_decoder.query_embed.",
    r"^transformer\.decoder\.reference_points\.":                           r"detr_decoder.reference_points.",
    r"^transformer\.decoder\.instance_query_embed\.":                       r"detr_decoder.instance_query_embed.",
    r"^transformer\.decoder\.instance_reference_points\.":                  r"detr_decoder.instance_reference_points.",
    r"^transformer\.decoder\.presence_token\.":                             r"detr_decoder.presence_token.",
    r"^transformer\.decoder\.presence_token_head\.layers\.0\.":             r"detr_decoder.presence_head.layer1.",
    r"^transformer\.decoder\.presence_token_head\.layers\.1\.":             r"detr_decoder.presence_head.layer2.",
    r"^transformer\.decoder\.presence_token_head\.layers\.2\.":             r"detr_decoder.presence_head.layer3.",
    r"^transformer\.decoder\.presence_token_out_norm\.":                    r"detr_decoder.presence_layer_norm.",
    r"^transformer\.decoder\.norm\.":                                       r"detr_decoder.output_layer_norm.",
    r"^transformer\.decoder\.bbox_embed\.layers\.0\.":                      r"detr_decoder.box_head.layer1.",
    r"^transformer\.decoder\.bbox_embed\.layers\.1\.":                      r"detr_decoder.box_head.layer2.",
    r"^transformer\.decoder\.bbox_embed\.layers\.2\.":                      r"detr_decoder.box_head.layer3.",
    r"^transformer\.decoder\.instance_bbox_embed\.layers\.0\.":             r"detr_decoder.instance_box_head.layer1.",
    r"^transformer\.decoder\.instance_bbox_embed\.layers\.1\.":             r"detr_decoder.instance_box_head.layer2.",
    r"^transformer\.decoder\.instance_bbox_embed\.layers\.2\.":             r"detr_decoder.instance_box_head.layer3.",
    r"^transformer\.decoder\.ref_point_head\.layers\.0\.":                  r"detr_decoder.ref_point_head.layer1.",
    r"^transformer\.decoder\.ref_point_head\.layers\.1\.":                  r"detr_decoder.ref_point_head.layer2.",
    r"^transformer\.decoder\.boxRPB_embed_x\.layers\.0\.":                  r"detr_decoder.box_rpb_embed_x.layer1.",
    r"^transformer\.decoder\.boxRPB_embed_x\.layers\.1\.":                  r"detr_decoder.box_rpb_embed_x.layer2.",
    r"^transformer\.decoder\.boxRPB_embed_y\.layers\.0\.":                  r"detr_decoder.box_rpb_embed_y.layer1.",
    r"^transformer\.decoder\.boxRPB_embed_y\.layers\.1\.":                  r"detr_decoder.box_rpb_embed_y.layer2.",
    r"^transformer\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.":         r"detr_decoder.layers.\1.self_attn.o_proj.",
    r"^transformer\.decoder\.layers\.(\d+)\.self_attn\.":                   r"detr_decoder.layers.\1.self_attn.",
    r"^transformer\.decoder\.layers\.(\d+)\.ca_text\.out_proj\.":           r"detr_decoder.layers.\1.text_cross_attn.o_proj.",
    r"^transformer\.decoder\.layers\.(\d+)\.ca_text\.":                     r"detr_decoder.layers.\1.text_cross_attn.",
    r"^transformer\.decoder\.layers\.(\d+)\.cross_attn\.out_proj\.":        r"detr_decoder.layers.\1.vision_cross_attn.o_proj.",
    r"^transformer\.decoder\.layers\.(\d+)\.cross_attn\.":                  r"detr_decoder.layers.\1.vision_cross_attn.",
    r"^transformer\.decoder\.layers\.(\d+)\.linear1\.":                     r"detr_decoder.layers.\1.mlp.fc1.",
    r"^transformer\.decoder\.layers\.(\d+)\.linear2\.":                     r"detr_decoder.layers.\1.mlp.fc2.",
    r"^transformer\.decoder\.layers\.(\d+)\.norm1\.":                       r"detr_decoder.layers.\1.vision_cross_attn_layer_norm.",
    r"^transformer\.decoder\.layers\.(\d+)\.catext_norm\.":                 r"detr_decoder.layers.\1.text_cross_attn_layer_norm.",
    r"^transformer\.decoder\.layers\.(\d+)\.norm2\.":                       r"detr_decoder.layers.\1.self_attn_layer_norm.",
    r"^transformer\.decoder\.layers\.(\d+)\.norm3\.":                       r"detr_decoder.layers.\1.mlp_layer_norm.",
    r"^dot_prod_scoring\.prompt_mlp\.layers\.0\.":                          r"dot_product_scoring.text_mlp.layer1.",
    r"^dot_prod_scoring\.prompt_mlp\.layers\.1\.":                          r"dot_product_scoring.text_mlp.layer2.",
    r"^dot_prod_scoring\.prompt_mlp\.out_norm\.":                           r"dot_product_scoring.text_mlp_out_norm.",
    r"^dot_prod_scoring\.prompt_proj\.":                                    r"dot_product_scoring.text_proj.",
    r"^dot_prod_scoring\.hs_proj\.":                                        r"dot_product_scoring.query_proj.",
    r"^segmentation_head\.pixel_decoder\.conv_layers\.(\d+)\.":             r"mask_decoder.pixel_decoder.conv_layers.\1.",
    r"^segmentation_head\.pixel_decoder\.norms\.(\d+)\.":                   r"mask_decoder.pixel_decoder.norms.\1.",
    r"^segmentation_head\.mask_embed\.layers\.(\d+)\.":                     r"mask_decoder.mask_embedder.layers.\1.",
    r"^segmentation_head\.mask_predictor\.mask_embed\.layers\.(\d+)\.":     r"mask_decoder.mask_embedder.layers.\1.",
    r"^segmentation_head\.instance_seg_head\.":                             r"mask_decoder.instance_projection.",
    r"^segmentation_head\.semantic_seg_head\.":                             r"mask_decoder.semantic_projection.",
    r"^segmentation_head\.cross_attend_prompt\.out_proj\.":                 r"mask_decoder.prompt_cross_attn.o_proj.",
    r"^segmentation_head\.cross_attend_prompt\.":                           r"mask_decoder.prompt_cross_attn.",
    r"^segmentation_head\.cross_attn_norm\.":                               r"mask_decoder.prompt_cross_attn_norm.",
    # --- sam3.1 新增: geometry 的 points_* 改名为 boxes_* ---
    r"^geometry_encoder\.points_direct_project\.":                         r"geometry_encoder.boxes_direct_project.",
    r"^geometry_encoder\.points_pool_project\.":                          r"geometry_encoder.boxes_pool_project.",
}

# 官方注明的预期差异(非错误):
# - patch_embeddings.projection.bias: 原始 patch projection 无 bias
# - rotary_emb.rope_embeddings: 原始预计算, HF 在线计算
# 官方注明的预期差异(非错误):
# - patch_embeddings.projection.bias: 原始 patch projection 无 bias
# - rotary_emb.rope_embeddings: 原始预计算, HF 在线计算
EXPECTED_MISSING = ("rotary_emb", "patch_embeddings.projection.bias")
# 预期的多余键(非错误):
# - interactive_convs/propagation_convs: sam3.1 TriNeck 对 convs 的训练副本,
#   detector 主路径(HF Sam3Model)只消费 convs
# - freqs_cis/rope_embeddings: 同上, HF 在线计算
EXPECTED_UNEXPECTED = ("interactive_convs", "propagation_convs", "freqs_cis", "rope_embeddings")


def strip_prefix(sd, prefixes=("detector.",)):
    """剥离 multiplex 顶层前缀, 只保留 detector 部分。"""
    out = {}
    dropped = 0
    for k, v in sd.items():
        for p in prefixes:
            if k.startswith(p):
                out[k[len(p):]] = v
                break
        else:
            dropped += 1
    print(f"stripped prefix: kept {len(out)} detector keys, dropped {dropped} others (tracker 等)")
    return out


def rename_keys(sd):
    old_text = "\n".join(sd.keys())
    new_text = old_text
    for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
        new_text = re.sub(pattern, replacement, new_text, flags=re.MULTILINE)
    mapping = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return {mapping.get(k, k): v for k, v in sd.items()}


def split_qkv(sd):
    """合并的 QKV 投影拆分为独立 Q/K/V(vision qkv 与 text in_proj_*)。"""
    for key in [k for k in list(sd.keys()) if ".attention.qkv." in k]:
        qkv = sd.pop(key)
        q, k, v = torch.chunk(qkv, 3, dim=0)
        sd[key.replace(".qkv.", ".q_proj.")] = q
        sd[key.replace(".qkv.", ".k_proj.")] = k
        sd[key.replace(".qkv.", ".v_proj.")] = v
    for key in [k for k in list(sd.keys()) if ".in_proj_" in k]:
        proj = sd.pop(key)
        q, k, v = torch.chunk(proj, 3, dim=0)
        if key.endswith("in_proj_weight"):
            base = key.replace("in_proj_weight", "")
        else:
            base = key.replace("in_proj_bias", "")
        suffix = "weight" if key.endswith("weight") else "bias"
        for name, t in (("q", q), ("k", k), ("v", v)):
            sd[f"{base}{name}_proj.{suffix}"] = t
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config-dir", required=True, help="含 config.json 的模型目录")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import AutoConfig
    from transformers.models.sam3.modeling_sam3 import Sam3Model

    print(f"[1/5] loading {args.checkpoint} ...")
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = strip_prefix(sd)

    print("[2/5] renaming keys (official mapping) ...")
    sd = rename_keys(sd)
    print("[3/5] splitting QKV ...")
    sd = split_qkv(sd)

    # cls token 剥离 + CLIP text_projection 转置
    pe = "vision_encoder.backbone.embeddings.position_embeddings"
    if pe in sd and sd[pe].shape[1] == 577:
        sd[pe] = sd[pe][:, 1:, :]
        print(f"  pos_embed: stripped cls token -> {tuple(sd[pe].shape)}")
    tp = "text_encoder.text_projection.weight"
    if tp in sd:
        sd[tp] = sd[tp].T
        print("  text_projection: transposed")

    print("[4/5] loading into Sam3Model ...")
    cfg = AutoConfig.from_pretrained(args.config_dir)
    det_cfg = getattr(cfg, "detector_config", cfg)
    det_cfg.architectures = ["Sam3Model"]
    # sam3.1 的 TriNeck 是 3 级 (convs 由 scale_factors=[4,2,1] 构建);
    # ModelScope config.json 里 [4,2,1,0.5] 是旧版默认值, 与实际权重不符
    det_cfg.vision_config.scale_factors = [4.0, 2.0, 1.0]
    print("  vision_config.scale_factors -> [4.0, 2.0, 1.0] (sam3.1 TriNeck)")
    model = Sam3Model(det_cfg)
    msg = model.load_state_dict(sd, strict=False)
    unexpected = [k for k in msg.unexpected_keys]
    real_missing = [k for k in msg.missing_keys if not any(p in k for p in EXPECTED_MISSING)]
    print(f"  missing: {len(msg.missing_keys)} (expected {len(msg.missing_keys)-len(real_missing)})")
    print(f"  unexpected: {len(unexpected)}")
    if real_missing:
        print("  REAL missing sample:", real_missing[:8])
    if unexpected:
        print("  unexpected sample:", unexpected[:8])

    os.makedirs(args.output, exist_ok=True)
    print(f"[5/5] saving to {args.output} ...")
    model.save_pretrained(args.output)
    det_cfg.save_pretrained(args.output)
    # tokenizer/processor 文件直接复用模型目录
    for f in ["vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "processor_config.json"]:
        src = os.path.join(args.config_dir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.output, f))

    del sd, model
    gc.collect()
    print(f"\nDONE. converted model: {args.output}")
    real_unexpected = [k for k in msg.unexpected_keys
                       if not any(p in k for p in EXPECTED_UNEXPECTED)]
    print(f"  unexpected: {len(msg.unexpected_keys)} (expected {len(msg.unexpected_keys)-len(real_unexpected)})")
    if real_unexpected:
        print("  REAL unexpected sample:", real_unexpected[:8])
    n_real = len(real_missing) + len(real_unexpected)
    print("conversion:", "OK ✅" if n_real == 0 else f"CHECK NEEDED ({n_real} unexpected/missing)")


if __name__ == "__main__":
    main()
