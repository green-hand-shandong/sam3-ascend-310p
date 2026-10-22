# SAM Family on Ascend 310P

[English](README_EN.md) | 中文

**English TL;DR:** This repo documents how to run the Meta **SAM family** — **SAM 3.1 / SAM 3**, **SAM 2.1**, **SAM (original ViT-H)** and **MedSAM** — on **Huawei Ascend 310P** (Atlas 300I Duo) with `torch_npu`. It contains a reusable compat layer (`ascend310p_compat`, 3 patches), an operator-level test suite, per-model numerical verification and end-to-end demos. Verified on CANN 9.1 / torch_npu 2.10.post4 — all four models run in fp32 eager: SAM3.1 @1008 ≈ **2.4 s / 7.5 GB**, SAM2.1 ≈ **0.88 s / 3.7 GB**, SAM ViT-H ≈ **1.56 s / 7.3 GB**, MedSAM ≈ **0.56 s / 3.9 GB**.

---

## 简介

本仓库记录了 Meta SAM 家族模型适配昇腾 **310P**（Atlas 300I Duo，Ascend 310P3）的完整过程：算子摸底 → 逐个解决不适配点 → 整模型数值验证 → 端到端推理，所有脚本可直接复现。

### 已验证模型

| 模型 | 权重来源 | 提示方式 | fp32 @1024 前向 | 显存 | 端到端 |
|---|---|---|---|---|---|
| **SAM 3.1 / SAM 3**（detector） | `facebook/sam3`（HF safetensors） | 文本 + box | 2.4s @1008（840M） | 7.5GB | ✅ top1 0.918 |
| **SAM 3.1**（Meta 原始权重转换） | `sam3.1_multiplex.pt` → `scripts/08_convert_sam31.py` | 文本 + box | 2.39s @1008 | 7.5GB | ✅ top1 0.554 |
| **SAM 2.1**（hiera-large） | `facebook/sam2.1-hiera-large` | 点 | 0.88s（217M） | 3.7GB | ✅ 无 NaN |
| **SAM**（原版 vit-huge） | `facebook/sam-vit-huge` | box | 1.56s（636M） | 7.3GB | ✅ 无 NaN |
| **MedSAM**（vit-base） | `flaviagiammarino/medsam-vit-base` | box | 0.56s（94M） | 3.9GB | ✅ 病灶覆盖 99.9% |
| MedSAM **fp16** | 同上 | box | **0.35s（1.6x）** | 3.2GB | ✅ 无 NaN |

均使用 HF `transformers>=5.13.1` 的原生实现（`Sam3VideoModel` / `Sam2Model` / `SamModel`）+ 本仓库 `ascend310p_compat` 兼容层。

## 310P 适配点（3 个补丁，见 `ascend310p_compat/`）

| # | 问题 | 报错 | 解法 | 验证 |
|---|---|---|---|---|
| 1 | **SDPA 融合算子不可用**（SAM 全家族都走 SDPA） | `can not cast format when output is input` | 手写 matmul+softmax（`attention.py`） | fp32 diff 2e-4 |
| 2 | **bicubic 插值无 kernel**（SAM2 pos-embed、SAM3 resize 必经） | `No supported Ops kernel ... [ResizeD]`，且失败会污染同进程 allocator | CPU 计算 + 按形状缓存（`interpolate.py`）；bicubic→bilinear 替代不可行（diff 1.4） | 与 CPU bit 级一致 |
| 3 | **激活函数拒绝非连续输入**（`SamLayerNorm` permute 输出触发） | `Only c10::MemoryFormat::Contiguous is supported` | 激活函数自动 `.contiguous()` 守卫（`activations.py`） | 连续零开销 |
| （附） | 1D linear 插值（SAM rel-pos 分支）触发 ResizeD | 同上 | 4D bilinear 退化维等价实现 | diff 0.0 |

另有一个输入侧的坑：`SamProcessor` 输出的 `input_boxes` 是 **float64**，310P 的 `AsStrided` 不支持 fp64，需 `.to(torch.float32)`。

在你的代码里两行接入：

```python
import ascend310p_compat
ascend310p_compat.patch_all()   # SDPA / interpolate / activations 全部补丁
model = model.to("npu")
```

## 快速开始

```bash
# 0) 宿主机：装有 Ascend Docker Runtime + 驱动；确认空闲卡
npu-smi info    # 选 Hugepages 0/0 且无进程的卡（未被任何容器认领过）

# 1) 下载权重（家族全支持）
pip install modelscope
bash scripts/01_download_model.sh sam3       # SAM3（HF safetensors，推荐）
bash scripts/01_download_model.sh sam3.1     # SAM3.1（Meta 原始格式）
bash scripts/01_download_model.sh sam2.1     # SAM2.1 / sam / medsam 同理

# 1.5) SAM3.1 原始权重转 HF 格式（missing=0 即转换完整）
python scripts/08_convert_sam31.py \
    --checkpoint /data1/models/SAM3.1/sam3.1_multiplex.pt \
    --config-dir /data1/models/SAM3.1 --output /data1/models/SAM3.1-hf

# 2) 启动测试容器（CANN/torch_npu 随镜像提供）
bash scripts/02_run_container.sh

# ---- 容器内 ----
pip install "transformers>=5.13.1" "tokenizers==0.22.2" safetensors pillow

# 3) 算子级摸底
python /workspace/scripts/03_op_test.py

# 4) 家族模型测试（NPU 前向 + 计时；--cpu-compare 加数值对比）
python /workspace/scripts/06_family_test.py --model medsam --dir /data1/models/MedSAM --cpu-compare
python /workspace/scripts/06_family_test.py --model sam    --dir /data1/models/SAM-vit-h
python /workspace/scripts/06_family_test.py --model sam2   --dir /data1/models/SAM2.1
python /workspace/scripts/06_family_test.py --model sam3   --dir /data1/models/SAM3

# 5) 端到端
python /workspace/scripts/05_e2e_probe.py /data1/models/SAM3 <图片> "person"   # SAM3 文本提示
python /workspace/scripts/07_medsam_e2e.py <图片> <x1> <y1> <x2> <y2>          # MedSAM box 提示
```

## 目录

```
├── README.md
├── docs/
│   ├── tutorial.md            # 详细教程：方法论、适配点原理、排查过程、FAQ
│   └── ci.yml.example         # GitHub Actions 配置（启用方法见下）
├── ascend310p_compat/         # 可复用兼容层（3 个补丁 + 环境自检）
│   ├── attention.py           #   SDPA -> 手写 matmul+softmax
│   ├── interpolate.py         #   bicubic->CPU 缓存; 1D linear->4D bilinear
│   ├── activations.py         #   激活函数非连续守卫
│   └── env.py                 #   assert_npu_ready / print_env
├── scripts/
│   ├── 01_download_model.sh   # ModelScope 权重下载
│   ├── 02_run_container.sh    # NPU 测试容器
│   ├── 03_op_test.py          # 算子级测试（NPU vs CPU 逐项对比）
│   ├── 04_model_forward.py    # SAM3 整模型验证（数值一致性 + 性能）
│   ├── 05_e2e_probe.py        # SAM3 端到端文本提示推理
│   ├── 06_family_test.py      # SAM 家族通用测试（medsam/sam/sam2/sam3）
│   └── 07_medsam_e2e.py       # MedSAM 医学图像 box 分割 demo
└── results/                   # 实测证据（json / 输出记录）
```

## 已知限制

- **SAM3 尺寸锁定**：视觉 RoPE 表按 `config.image_size`（1008）初始化时固定生成，改分辨率必须改 config 重建模型
- **权重格式**：`facebook/sam3.1` 的 `sam3.1_multiplex.pt` 是 Meta 原始包格式（键名与 HF 不同），不能直接 load；用 `facebook/sam3` 的 HF safetensors 替代（detector 部分等价）
- **MedSAM 权重**：官方 bowang-lab/MedSAM 只发原始 .pt；本仓库用 HF 格式镜像 `flaviagiammarino/medsam-vit-base`（同一 ViT-B 权重）
- **NPU 按容器命名空间独占**：一张卡只能属于一个容器，新容器对已占用卡报 `aclInit 507899`；判别方法见 tutorial FAQ
- tracker / 视频会话推理未验证；fp16 已在算子级验证可行，整模型 fp16 与 torchair/om 部署为后续方向

## 致谢

- [Meta SAM 3](https://ai.meta.com/sam3/) / [SAM 2](https://ai.meta.com/sam2/) / [MedSAM](https://github.com/bowang-lab/MedSAM)、HuggingFace `transformers` Sam 系列实现
- [ModelScope](https://modelscope.cn/models/facebook/sam3)（权重镜像）· [Ascend CANN / torch_npu](https://gitee.com/ascend)

## 启用 CI（可选）

`tests/` 是兼容层的 CPU 数值测试（无需 NPU）。CI 配置在 `docs/ci.yml.example`，
由于推送 workflow 文件需要 token 带 `workflow` scope，请手动启用：

```bash
mkdir -p .github/workflows && cp docs/ci.yml.example .github/workflows/ci.yml
git add .github/workflows/ci.yml && git commit -m "ci: enable CPU parity tests" && git push
```

## License

Apache-2.0
