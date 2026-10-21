#!/usr/bin/env python3
"""SAM 家族 310P 端到端 demo：MedSAM 医学图像 box 提示分割。

用法(容器内): python /workspace/scripts/07_medsam_e2e.py <image> <x1> <y1> <x2> <y2>
无图片时用内置合成图演示。
"""
import os
import sys
import time

import numpy as np
import torch
import torch_npu  # noqa: F401

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ascend310p_compat import enable_manual_sdpa, patch_activations, patch_interpolate  # noqa: E402

enable_manual_sdpa(verbose=False)
patch_interpolate(verbose=False)
patch_activations(verbose=False)

from transformers import SamModel, SamProcessor  # noqa: E402

MODEL_DIR = os.environ.get("MEDSAM_DIR", "/data1/models/MedSAM")

processor = SamProcessor.from_pretrained(MODEL_DIR)
model = SamModel.from_pretrained(MODEL_DIR, attn_implementation="eager").eval()

# 合成医学风格图: 亮背景 + 椭圆"病灶"暗斑
img = np.full((512, 512, 3), 200, dtype="uint8")
yy, xx = np.mgrid[:512, :512]
mask = ((xx - 300) / 90.0) ** 2 + ((yy - 260) / 60.0) ** 2 < 1
img[mask] = [60, 60, 70]
noise = np.random.randint(0, 25, img.shape, dtype="uint8")
img = img.astype(int) + noise.astype(int)
img = np.clip(img, 0, 255).astype("uint8")
if len(sys.argv) > 1:
    from PIL import Image
    img = np.array(Image.open(sys.argv[1]).convert("RGB"))
    box = [float(x) for x in sys.argv[2:6]]  # x1 y1 x2 y2 像素坐标
else:
    box = [210, 200, 390, 320]  # 覆盖合成病灶

inputs = processor(images=img, input_boxes=[[[float(x) for x in box]]], return_tensors="pt")
# processor 输出的 input_boxes 是 float64, 310P 的 AsStrided 不支持 fp64, 转 float32
inputs["input_boxes"] = inputs["input_boxes"].to(torch.float32)
print("input box(xyxy):", box, "| dtype:", inputs["input_boxes"].dtype)

model = model.npu()
t0 = time.time()
with torch.no_grad():
    out = model(**{k: v.npu() for k, v in inputs.items()})
torch.npu.synchronize()
print(f"forward {time.time()-t0:.1f}s (含编译; 热身约 0.56s)")

masks = out.pred_masks[0][0]  # (3, 256, 256) multimask
iou = out.iou_scores[0][0].squeeze(-1)
best = int(iou.argmax())
m = masks[best].sigmoid() > 0.5
area = int(m.sum())
print(f"iou_scores: {[round(s,4) for s in iou.tolist()]}")
print(f"best mask #{best}: 覆盖 {area} px @256x256 ({area/256/256*100:.1f}%)")

# 语义 sanity: box 内应被覆盖
scale = 256 / 512
bx = [int(box[0]*scale), int(box[1]*scale), int(box[2]*scale), int(box[3]*scale)]
inner = m[bx[1]+10:bx[3]-10, bx[0]+10:bx[2]-10]
print(f"box 内部覆盖率: {inner.float().mean().item()*100:.1f}% (病灶应 >50%)")
