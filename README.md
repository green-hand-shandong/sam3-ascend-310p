# SAM 3.1 on Ascend 310P

**English TL;DR:** This repo documents how to run Meta **SAM 3.1** (Segment Anything Model 3, `Sam3VideoModel` / `Sam3Model` in HF `transformers`) on **Huawei Ascend 310P** (Atlas 300I Duo) with `torch_npu`. It contains an operator-level test suite, two required workarounds (**SDPA fusion op unavailable** → manual matmul+softmax attention; **bicubic interpolate (ResizeD) missing kernel** → CPU fallback), a full-model numerical verification, and an end-to-end text-prompted detection demo. Verified on CANN 9.1 / torch_npu 2.10.post4: full-res 1008×1008 fp32 forward ≈ **2.4 s**, **7.5 GB** device memory.

---

## 简介

SAM 3.1 是 Meta 的可提示视觉分割大模型（图像 + 视频，支持**文本提示**）。本仓库记录了把它适配到昇腾 **310P**（Atlas 300I Duo，Ascend 310P3）上的完整过程：算子摸底 → 逐个解决不适配点 → 整模型数值验证 → 端到端推理，所有脚本可直接复现。

模型结构（`facebook/sam3`，HF `transformers>=5.13.1`）：

| 组件 | 结构 |
|---|---|
| 图像编码器 | ViT-H：1008×1008 输入，patch 14 → 5184 tokens，32 层，hidden 1024，窗口注意力 + 4 层全局注意力，RoPE |
| 文本编码器 | CLIP-style：24 层，hidden 1024，词表 49408 |
| 检测头 | DETR 风格 encoder 6 层 + decoder 6 层（256 dim）+ mask decoder |
| tracker（视频） | SAM2 风格 memory attention（RoPE 2D）+ memory encoder |

## 实测结论

环境：Atlas 300I Duo（Ascend 310P3）· 驱动 25.5.1 · CANN 9.1.0 · torch 2.10.0 · torch_npu 2.10.0.post4 · transformers 5.13.1

| 阶段 | 结果 |
|---|---|
| 算子级测试（31 项） | **31/32 通过**，fp32 NPU vs CPU diff ≤1e-4（卷积类 ≤0.08 属累加噪声） |
| 整模型数值验证（随机权重 @336） | 全部 6 路输出与 CPU 一致：`pred_boxes` 4e-5 / `pred_logits` 2.7e-4 / `pred_masks` 3.8e-4 / `semantic_seg` 1e-3 |
| 真实权重前向 @1008 fp32 | 首跑 98.5s（含 CANN 在线编译），热身后 **2.4s/次，显存 7.5GB**，输出无 NaN |
| 端到端（真实图 + 文本提示 "diagram"） | top1 置信度 **0.918** ✅ |

### 两个必须解决的适配点

1. **`F.scaled_dot_product_attention` 原生融合算子不可用**
   报错 `can not cast format when output is input (ERR01007)`。`allow_internal_format=False` 和 `sdpa_kernel(SDPBackend.MATH)` 均无效（torch_npu 在 C++ 层劫持了 SDPA 分发）。
   ✅ 解法：monkeypatch 成手写 `matmul + softmax + matmul`（fp32 diff 2e-4），见 [`ascend310p_compat`](ascend310p_compat/attention.py)（`scripts/03_op_test.py` 内含自包含版本）。

2. **bicubic 插值（ResizeD 算子）在 CANN 9.1 无可用 kernel**
   报错 `No supported Ops kernel and engine are found for [ResizeD]`。**且失败会破坏同进程 allocator，后续所有算子输出错乱**（排查耗时最久的坑）。
   ✅ 解法：bicubic 放 CPU 跑（位置编码插值本就是离线预处理）；测试时用子进程隔离。

原理分析、排查过程和更多细节见 **[docs/tutorial.md](docs/tutorial.md)**。

## 快速开始

```bash
# 0) 宿主机：装有 Ascend Docker Runtime + 驱动；确认空闲卡
npu-smi info

# 1) 下载权重（HF 格式，约 3.4GB）
pip install modelscope
bash scripts/01_download_model.sh facebook/sam3 SAM3     # → /data1/models/SAM3

# 2) 启动测试容器（CANN/torch_npu 环境随镜像提供）
bash scripts/02_run_container.sh                          # 默认用 NPU_DEV=14

# ---- 以下在容器内 ----
# 3) 安装 Python 依赖（torch/torch_npu 用镜像自带的，勿覆盖）
pip install "transformers>=5.13.1" "tokenizers>=0.22,<0.23.1" safetensors pillow

# 4) 算子级摸底（可选，但强烈建议先跑）
python /workspace/scripts/03_op_test.py

# 5) 整模型验证：随机权重数值对比 + 真实权重 @1008 前向计时
python /workspace/scripts/04_model_forward.py

# 6) 端到端：真实图片 + 文本提示
python /workspace/scripts/05_e2e_probe.py /data1/models/SAM3 <图片路径> "person"
```

## 目录

```
├── README.md
├── docs/
│   └── tutorial.md          # 详细教程：环境、算子摸底方法、两个适配点的原理与排查、FAQ
├── scripts/
│   ├── 01_download_model.sh # ModelScope 权重下载
│   ├── 02_run_container.sh  # NPU 测试容器
│   ├── 03_op_test.py        # 算子级测试（NPU vs CPU 逐项对比）
│   ├── 04_model_forward.py  # 整模型验证（数值一致性 + 性能）
│   └── 05_e2e_probe.py      # 端到端文本提示推理
└── results/                 # 实测证据（json / 输出记录）
```

## 已知限制

- **模型尺寸锁定**：SAM3 视觉 RoPE 表按 `config.image_size`（1008）在初始化时固定生成，改输入分辨率必须改 config 重建模型
- **权重格式**：`facebook/sam3.1` 的 `sam3.1_multiplex.pt` 是 Meta 原始包格式（键名 `detector.backbone.vision_backbone.trunk...`），与 HF 布局不同名，不能直接 load；推荐用 `facebook/sam3` 的 HF 格式 `model.safetensors`（`Sam3VideoModel.from_pretrained` 可用，detector 取 `.detector_model`）
- **NPU 按容器命名空间独占**：驱动会将首张被某容器打开的卡绑定到该容器命名空间，新容器对已占用卡报 `aclInit 507899 (Resource Busy)`；一张卡只能属于一个容器
- tracker / 视频会话推理（`Sam3VideoModel` 的 session API）尚未验证；fp16 已在算子级验证可行，整模型 fp16 与 torchair/om 部署为后续方向

## 致谢

- [Meta SAM 3](https://ai.meta.com/sam3/) / HuggingFace `transformers` Sam3 实现
- [ModelScope](https://modelscope.cn/models/facebook/sam3)（权重镜像）· [Ascend CANN / torch_npu](https://gitee.com/ascend)

## License

Apache-2.0
