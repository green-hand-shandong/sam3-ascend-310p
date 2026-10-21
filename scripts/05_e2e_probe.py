#!/usr/bin/env python3
"""SAM3.1 适配 310P — 端到端探针：真实图片 + 文本提示，NPU fp32 前向。

用法(容器内): python /tmp/05_e2e_probe.py <image_path> [prompt]
"""
import os
import sys
import time

import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ascend310p_compat import enable_manual_sdpa  # noqa: E402

enable_manual_sdpa(verbose=False)

from transformers import AutoImageProcessor, AutoTokenizer, Sam3VideoModel  # noqa: E402

MODEL_DIR = os.environ.get("SAM31_DIR", sys.argv[1] if len(sys.argv) > 1 else "/data1/models/SAM3")
IMAGE = sys.argv[2] if len(sys.argv) > 2 else os.path.join(MODEL_DIR, "assets", "sam3.1_diagram.png")
PROMPT = sys.argv[3] if len(sys.argv) > 3 else "diagram"

image_processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
vm = Sam3VideoModel.from_pretrained(MODEL_DIR, attn_implementation="eager").eval()
det = vm.detector_model  # Sam3Model: ViT-H + CLIP text + DETR

from PIL import Image  # noqa: E402
img = Image.open(IMAGE).convert("RGB")
pixel_values = image_processor(images=img, return_tensors="pt")["pixel_values"]
# 与 Sam3Processor 一致: padding="max_length", max_length=32
text_inputs = tokenizer([PROMPT], return_tensors="pt", padding="max_length", max_length=32)
print("inputs: pixel_values", tuple(pixel_values.shape))

det = det.npu()
t0 = time.time()
with torch.no_grad():
    out = det(pixel_values=pixel_values.npu(),
              input_ids=text_inputs["input_ids"].npu(),
              attention_mask=text_inputs["attention_mask"].npu())
torch.npu.synchronize()
print(f"NPU forward done in {time.time()-t0:.1f}s")

scores = out["pred_logits"][0].sigmoid()
boxes = out["pred_boxes"][0]
top = scores.topk(5)
print(f"prompt={PROMPT!r} top5 scores:", [round(s, 4) for s in top.values.tolist()])
for s, i in zip(top.values, top.indices):
    cxcywh = boxes[i].tolist()
    xyxy = [round(cxcywh[0]*1008 - cxcywh[2]*1008/2), round(cxcywh[1]*1008 - cxcywh[3]*1008/2),
            round(cxcywh[0]*1008 + cxcywh[2]*1008/2), round(cxcywh[1]*1008 + cxcywh[3]*1008/2)]
    print(f"  score={s:.4f} box(xyxy@1008)={xyxy}")
print("presence logit:", out["presence_logits"].item())
