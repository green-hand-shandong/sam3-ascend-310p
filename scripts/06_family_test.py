#!/usr/bin/env python3
"""SAM 家族模型 310P 适配 — 通用整模型测试（SamModel / Sam2Model / Sam3Model）。

对每个模型:
  1. from_pretrained 加载（HF 格式权重）
  2. NPU fp32 前向（随机或真实输入），与 CPU 对比（数值一致性）
  3. 计时 + 显存

用法(容器内):
  python 06_family_test.py --model medsam  --dir /data1/models/MedSAM
  python 06_family_test.py --model sam     --dir /data1/models/SAM-vit-h
  python 06_family_test.py --model sam2    --dir /data1/models/SAM2.1
  python 06_family_test.py --model sam3    --dir /data1/models/SAM3
"""
import argparse
import json
import os
import sys
import time
import traceback

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ascend310p_compat import enable_manual_sdpa, patch_activations, patch_interpolate  # noqa: E402

enable_manual_sdpa(verbose=False)
patch_interpolate(verbose=False)
patch_activations(verbose=False)

OUT = os.environ.get("RESULTS_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))


def load_model(kind, d):
    from transformers import SamModel
    if kind == "sam3":
        from transformers.models.sam3_video.modeling_sam3_video import Sam3VideoModel
        vm = Sam3VideoModel.from_pretrained(d, attn_implementation="eager")
        return vm.detector_model, vm.detector_model.config
    if kind == "sam2":
        from transformers.models.sam2.modeling_sam2 import Sam2Model
        m = Sam2Model.from_pretrained(d, attn_implementation="eager")
        return m, m.config
    m = SamModel.from_pretrained(d, attn_implementation="eager")
    return m, m.config


def make_inputs(kind, d):
    """各家族的输入。MedSAM/SAM 用 box 提示；sam2/sam3 点+文本。"""
    from transformers import AutoImageProcessor, AutoTokenizer

    proc = AutoImageProcessor.from_pretrained(d)
    import numpy as np
    img = (np.random.rand(1024, 1024, 3) * 255).astype("uint8")
    pv = proc(images=img, return_tensors="pt")["pixel_values"]

    if kind in ("medsam", "sam"):  # box 提示 (x1, y1, x2, y2) 归一化, 3D: [batch, nb_boxes, 4]
        input_boxes = torch.tensor([[[256.0, 256.0, 768.0, 768.0]]]) / 1024.0
        return {"pixel_values": pv, "input_boxes": input_boxes}
    if kind == "sam2":
        return {"pixel_values": pv,
                "input_points": torch.tensor([[[[512.0, 512.0]]]]) / 1024.0,
                "input_labels": torch.tensor([[1]])}
    # sam3: 文本提示
    tok = AutoTokenizer.from_pretrained(d)
    t = tok(["object"], return_tensors="pt", padding="max_length", max_length=32)
    return {"pixel_values": pv, "input_ids": t["input_ids"], "attention_mask": t["attention_mask"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["medsam", "sam", "sam2", "sam3"])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cpu-compare", action="store_true", help="额外做 CPU vs NPU 对比(慢)")
    args = ap.parse_args()

    dev = "npu"
    RES = {"model": args.model, "dir": args.dir}

    print(f"[{args.model}] loading from {args.dir}")
    model, cfg = load_model(args.model, args.dir)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    RES["params_M"] = round(n_params, 1)
    print(f"  params: {n_params:.0f}M")

    inputs = make_inputs(args.model, args.dir)
    RES["inputs"] = {k: list(v.shape) for k, v in inputs.items()}

    def fwd(m, ip):
        with torch.no_grad():
            return m(**ip)

    # --- NPU 前向（先编译，后计时）
    print(f"[{args.model}] NPU fp32 forward")
    try:
        model = model.to(dev)
        ip_npu = {k: v.to(dev) for k, v in inputs.items()}
        t0 = time.time()
        out = fwd(model, ip_npu)
        torch.npu.synchronize()
        first = time.time() - t0
        outs = {k: tuple(v.shape) for k, v in out.items() if hasattr(v, "shape")}
        nans = {k: bool(torch.isnan(v.float()).any().item()) for k, v in out.items() if hasattr(v, "shape") and v.is_floating_point()}
        print(f"  first(compile) {first:.1f}s; outputs: {outs}")
        print(f"  nan: {nans}")

        for _ in range(2):
            fwd(model, ip_npu)
        torch.npu.synchronize()
        t0 = time.time()
        n = 3
        for _ in range(n):
            fwd(model, ip_npu)
        torch.npu.synchronize()
        avg = (time.time() - t0) / n
        RES.update({"first_forward_s": round(first, 1), "avg_latency_s": round(avg, 3),
                    "output_shapes": outs, "nan": nans,
                    "mem_reserved_GB": round(torch.npu.memory_reserved() / 2**30, 2)})
        print(f"  avg {avg:.3f}s, mem {torch.npu.memory_reserved()/2**30:.2f} GB")
    except Exception as e:  # noqa: BLE001
        RES["npu_error"] = repr(e)
        RES["trace"] = traceback.format_exc(limit=6)
        print("  [ERROR]", repr(e)[:200])

    # --- CPU 对比（可选）
    if args.cpu_compare and "npu_error" not in RES:
        try:
            model = model.to("cpu")
            out_cpu = fwd(model, inputs)
            model.to(dev)
            ip_npu = {k: v.to(dev) for k, v in inputs.items()}
            out_npu = fwd(model, ip_npu)
            torch.npu.synchronize()
            diff = {}
            for k in out_cpu.keys():
                a, b = out_npu[k].float().cpu(), out_cpu[k].float()
                if hasattr(a, "shape") and a.shape == b.shape and a.numel():
                    diff[k] = round((a - b).abs().max().item(), 6)
            RES["cpu_vs_npu_max_diff"] = diff
            print(f"  CPU vs NPU diff: {diff}")
        except Exception as e:  # noqa: BLE001
            RES["cpu_error"] = repr(e)
            print("  [ERROR cpu]", repr(e)[:200])

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"family_{args.model}.json")
    with open(path, "w") as f:
        json.dump(RES, f, indent=2, ensure_ascii=False)
    print("results ->", path)


if __name__ == "__main__":
    main()
