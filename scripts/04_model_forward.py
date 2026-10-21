#!/usr/bin/env python3
"""SAM3.1 适配 310P — 整模型(detector Sam3Model)前向验证。

步骤:
  1. 加载 sam3.1_multiplex.pt，检查键名结构与 Sam3Model 的匹配（自动剥前缀）
  2. CPU vs NPU 数值对比 @336：image_size=336 的 config 克隆 + 随机权重
     （SAM3 视觉 RoPE 表按 image_size 固定生成，模型尺寸锁定，不能直接缩小输入）
  3. 真实权重 NPU fp32 前向 @1008x1008（原始分辨率）+ 计时
  4. 结果写 $RESULTS_DIR/model_forward_results.json

用法(容器内): python /workspace/scripts/04_model_forward.py
"""
import copy
import json
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ascend310p_compat import enable_manual_sdpa as patch_sdpa  # noqa: E402

MODEL_DIR = os.environ.get("SAM31_DIR", "/data1/models/SAM3")  # HF 格式权重目录
CKPT = os.path.join(MODEL_DIR, "sam3.1_multiplex.pt")
OUT = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")) + "/model_forward_results.json"
RESULTS = {}

PROMPT = "person"
# 前缀候选：multiplex 权重可能含 detector/tracker 两部分
CANDIDATE_PREFIXES = ["detector_model.", "detector.", "sam3_model.", "model.", ""]


def load_state_dict():
    print(f"loading {CKPT} ...")
    try:
        sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict) or "state_dict" in sd:
        sd = sd.get("state_dict", sd)
    keys = list(sd.keys())
    RESULTS["ckpt"] = {"n_keys": len(keys), "first_keys": keys[:8],
                       "total_params_M": round(sum(v.numel() for v in sd.values() if hasattr(v, "numel")) / 1e6, 1)}
    print(f"  {len(keys)} keys, top-level prefixes:",
          sorted({k.split(".")[0] for k in keys})[:10])

    # 自动选前缀: 能匹配到 Sam3Model 顶层子模块(vision_encoder/text_encoder/...)最多的那个
    SUBMODULES = ("vision_encoder", "text_encoder", "text_projection", "geometry_encoder",
                  "detr_encoder", "detr_decoder", "mask_decoder", "dot_product_scoring")
    best, best_n = "", -1
    for p in CANDIDATE_PREFIXES:
        n = sum(1 for k in keys if k.startswith(p) and k[len(p):].split(".")[0] in SUBMODULES)
        if n > best_n:
            best, best_n = p, n
    stripped = {k[len(best):]: v for k, v in sd.items() if k.startswith(best)}
    RESULTS["ckpt"]["chosen_prefix"] = repr(best)
    RESULTS["ckpt"]["matched_keys"] = best_n
    print(f"  chosen prefix={best!r} -> {best_n} detector keys")
    return stripped


def build_model(config_dir=None, state_dict=None):
    """优先 from_pretrained(HF safetensors)；否则 config 构建后手动 load。
    safetensors 为 Sam3VideoModel 布局(detector_model./tracker_model./tracker_neck.)，
    取 detector_model(Sam3Model) 做验证。"""
    import struct
    from transformers import AutoConfig
    from transformers.models.sam3.modeling_sam3 import Sam3Model

    d = config_dir or MODEL_DIR
    st = os.path.join(d, "model.safetensors")
    if state_dict is None and os.path.exists(st):
        with open(st, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        if any(k.startswith("detector_model.") for k in hdr):
            from transformers.models.sam3_video.modeling_sam3_video import Sam3VideoModel
            vm = Sam3VideoModel.from_pretrained(d, attn_implementation="eager")
            print("loaded Sam3VideoModel; using .detector_model for forward test")
            det = vm.detector_model
            return det, det.config
        model = Sam3Model.from_pretrained(d, attn_implementation="eager")
        return model, model.config
    cfg_full = AutoConfig.from_pretrained(d)
    cfg = getattr(cfg_full, "detector_config", cfg_full)
    cfg._attn_implementation = "eager"
    model = Sam3Model(cfg)
    return model, cfg


def encode_text(config_dir=None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config_dir or MODEL_DIR)
    enc = tok([PROMPT], padding="max_length", max_length=32, truncation=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def forward_once(model, pixel_values, ids, mask):
    with torch.no_grad():
        return model(pixel_values=pixel_values, input_ids=ids, attention_mask=mask)



def compare_336_random(dev):
    """step2: 随机权重 + image_size=336 config，CPU vs NPU 输出对比。"""
    from transformers.models.sam3.modeling_sam3 import Sam3Model

    cfg_full = AutoConfigForSam(MODEL_DIR)
    det = getattr(cfg_full, "detector_config", cfg_full)  # 必须传 detector 配置本身
    cfg336 = copy.deepcopy(det)
    cfg336._attn_implementation = "eager"
    cfg336.vision_config.backbone_config.image_size = 336
    cfg336.vision_config.pretrain_image_size = 336

    torch.manual_seed(42)
    m_a = Sam3Model(cfg336)
    torch.manual_seed(42)
    m_b = Sam3Model(cfg336)
    m_a.load_state_dict(m_b.state_dict())
    del m_b
    m_a.eval()

    torch.manual_seed(1)
    img = torch.rand(1, 3, 336, 336) * 2 - 1
    ids, mask = encode_text(MODEL_DIR) if os.path.isdir(MODEL_DIR) and os.path.exists(
        os.path.join(MODEL_DIR, "tokenizer_config.json")) else (None, None)
    if ids is None:  # 容器里没有权重目录时用 /tmp/sam31
        ids, mask = encode_text("/tmp/sam31")

    with torch.no_grad():
        t0 = time.time()
        out_cpu = m_a(pixel_values=img, input_ids=ids, attention_mask=mask)
        print(f"  CPU forward @336 in {time.time()-t0:.1f}s")
        m_npu = m_a.to(dev)
        out_npu = m_npu(pixel_values=img.to(dev), input_ids=ids.to(dev), attention_mask=mask.to(dev))
        torch.npu.synchronize()

    diff = {}
    for k in out_cpu.keys():
        a, b = out_npu[k].float().cpu(), out_cpu[k].float()
        if a.shape == b.shape and a.numel():
            diff[k] = round((a - b).abs().max().item(), 6)
    RESULTS["compare_336_random_weights"] = {"max_abs_diff": diff}
    print("  NPU vs CPU @336 max_abs_diff:", diff)
    del m_a, m_npu, out_cpu, out_npu
    torch.npu.empty_cache()


def AutoConfigForSam(model_dir):
    from transformers import AutoConfig
    cfg_full = AutoConfig.from_pretrained(model_dir)
    return cfg_full


def main():
    dev = "npu"
    import torch_npu  # noqa: F401

    patch_sdpa()

    cfg_dir = MODEL_DIR if os.path.exists(os.path.join(MODEL_DIR, "config.json")) else "/tmp/sam31"

    # --- 1. 数值一致性(随机权重, 336)
    print("=" * 70)
    print("[step1] CPU vs NPU @336, random weights")
    if os.environ.get("SKIP_STEP1"):
        print("[skip] SKIP_STEP1=1")
    else:
        try:
            compare_336_random(dev)
        except Exception as e:  # noqa: BLE001
            RESULTS["compare_336_random_weights"] = {"error": repr(e), "trace": traceback.format_exc(limit=8)}
            print("[ERROR] compare@336:", repr(e))

    # --- 2. 真实权重加载
    if os.environ.get("SKIP_WEIGHTS"):
        print("[skip] SKIP_WEIGHTS=1")
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(RESULTS, f, indent=2, ensure_ascii=False)
        return
    print("=" * 70)
    print("[step2] load real weights")
    has_st = os.path.exists(os.path.join(cfg_dir, "model.safetensors"))
    if not has_st and not os.path.exists(CKPT):
        print(f"[skip] {CKPT} 不存在，跳过真实权重验证")
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(RESULTS, f, indent=2, ensure_ascii=False)
        return

    if os.path.exists(os.path.join(cfg_dir, "model.safetensors")):
        sd = None
        model, cfg = build_model(cfg_dir)
        RESULTS["load"] = {"source": "model.safetensors (HF format, from_pretrained)"}
        print("loaded HF safetensors directly")
        msg = None
    else:
        sd = load_state_dict()
        model, cfg = build_model(cfg_dir)
        msg = model.load_state_dict(sd, strict=False)
    if msg is not None:
        RESULTS["load"].update({
            "missing_n": len(msg.missing_keys),
            "missing": msg.missing_keys[:15],
            "unexpected_n": len(msg.unexpected_keys),
            "unexpected": msg.unexpected_keys[:15],
        })
        print(f"load_state_dict: missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
        if msg.missing_keys:
            print("  sample missing:", msg.missing_keys[:5])
    model.eval()

    ids, mask = encode_text(cfg_dir)
    RESULTS["text"] = {"prompt": PROMPT, "input_ids": ids[0][:8].tolist()}

    # --- 3. NPU @1008 + 计时
    print("=" * 70)
    print("[step3] real weights, NPU fp32 forward @1008")
    try:
        model = model.to(dev)
        img = (torch.rand(1, 3, 1008, 1008, device=dev) * 2 - 1)
        ids_d, mask_d = ids.to(dev), mask.to(dev)
        t0 = time.time()
        out = forward_once(model, img, ids_d, mask_d)
        torch.npu.synchronize()
        print(f"NPU forward @1008 OK ({time.time()-t0:.1f}s); outputs:",
              {k: tuple(v.shape) for k, v in out.items()})
        for k, v in out.items():
            print(f"  {k}: nan={torch.isnan(v.float()).any().item()}")

        for _ in range(2):  # warmup
            forward_once(model, img, ids_d, mask_d)
        torch.npu.synchronize()
        t0 = time.time()
        n = 3
        for _ in range(n):
            forward_once(model, img, ids_d, mask_d)
        torch.npu.synchronize()
        avg = (time.time() - t0) / n
        RESULTS["perf_1008"] = {
            "output_shapes": {k: tuple(v.shape) for k, v in out.items()},
            "avg_latency_s": round(avg, 3),
            "mem_reserved_GB": round(torch.npu.memory_reserved() / 2**30, 2),
        }
        print(f"avg latency @1008 fp32: {avg:.3f}s, mem {torch.npu.memory_reserved()/2**30:.2f} GB")
    except Exception as e:  # noqa: BLE001
        RESULTS["perf_1008"] = {"error": repr(e), "trace": traceback.format_exc(limit=8)}
        print("[ERROR] full-size forward:", repr(e))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    print("results ->", OUT)


if __name__ == "__main__":
    main()
