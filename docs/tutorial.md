# SAM 3.1 适配昇腾 310P 完整教程

> 本文是完整的"怎么做到的"记录：从拿到模型，到算子摸底，到定位并解决两个不适配点，再到整模型数值验证与端到端推理。所有代码在 `scripts/` 下可直接复现，实测数据在 `results/`。

## 0. 总览

适配一个新模型到 NPU，推荐的路线是**先算子、后整模型**：

```
环境准备 → 算子级摸底(03) → 逐个解决不适配点 → 整模型数值验证(04) → 端到端(05) → 性能优化
```

算子摸底的价值：把"整模型跑不起来"这种大而模糊的问题，拆成十几个边界清晰的小问题。SAM 3.1 的两个不适配点（SDPA、bicubic）都是在这一步暴露的。

## 1. 背景与目标

- **模型**：SAM 3.1（Meta，可提示分割，支持文本提示）。HF `transformers>=5.13.1` 提供 `Sam3Model`（图像检测器）与 `Sam3VideoModel`（视频，含 tracker）
- **硬件**：Atlas 300I Duo（Ascend 310P3，单卡 44GB），Ascend 推理卡，无 CUDA
- **目标**：torch_npu eager 模式跑通前向 + 数值对齐 CPU，为后续 fp16/torchair/om 部署摸底

模型内部四大件（决定了要测哪些算子）：

| 组件 | 关键算子 |
|---|---|
| ViT-H 图像编码器（1008²，5184 token） | Conv2d(patch14)、LayerNorm、GELU、Window/Global Attention、RoPE、ConvTranspose2d+GroupNorm(neck) |
| CLIP 文本编码器（24 层） | Embedding(49408)、Causal Attention |
| DETR 头（6+6 层，256 dim） | MultiheadAttention、einsum/bmm、topk、sigmoid、cxcywh→xyxy |
| memory attention（视频 tracker） | RoPE 2D、cross-attention、interpolate |

## 2. 环境准备

### 2.1 镜像选择（踩过坑）

| 镜像 | 结果 |
|---|---|
| `quay.io/ascend/vllm-ascend:v0.23.0-310p`（CANN 9.1.0 + torch_npu 2.10.0.post4, py3.12） | ✅ 使用 |
| `quay.io/ascend/mindie:3.1.0-cann9.0.1-...` | ❌ 本机 aclInit 失败(507899) |

宿主机只装驱动（25.5.1），CANN/torch_npu 全在容器里。容器通过 **Ascend Docker Runtime** 获得设备：`-e ASCEND_VISIBLE_DEVICES=14`（宿主机 `npu-smi info` 里的 Device 列）。

### 2.2 Python 依赖的版本坑

- 镜像自带 transformers 4.51，**没有 Sam3**。需要 `transformers>=5.13.1`
- transformers 5.13 要求 `tokenizers>=0.22,<=0.23.0`，而 pip 默认装 0.23.3 会报版本校验错误 → 显式装 `tokenizers==0.22.2`
- **torch / torch-npu 用镜像自带的，不要 pip install torch 覆盖**（NPU 版 torch 只随 CANN 发行）

### 2.3 权重下载与格式（重要）

ModelScope 上有两个相关仓库：

| 仓库 | 权重 | 格式 |
|---|---|---|
| `facebook/sam3` | `model.safetensors` | **HF transformers 布局**（`detector_model.vision_encoder...`），`from_pretrained` 直接可用 ✅ |
| `facebook/sam3.1` | `sam3.1_multiplex.pt` | **Meta 原始包布局**（`detector.backbone.vision_backbone.trunk...`），与 HF 键名完全不同 ❌ |

判别方法（safetensors 的键名都在文件头 JSON 里，不用下完就能看）：

```python
import json, struct
with open("model.safetensors", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    keys = json.loads(f.read(n)).keys()
```

本仓库教程统一用 `facebook/sam3` 的 safetensors。加载方式：`Sam3VideoModel.from_pretrained(dir)` 加载整个多路模型，图像检测器取 `.detector_model`。

## 3. 第一步：算子级摸底

`scripts/03_op_test.py` 的做法：**每个算子用同一份 CPU 输入分别喂 CPU 和 NPU，对比输出**。

三个容易踩的坑（都踩过）：

1. **输入必须只生成一次再搬运**。如果 CPU/NPU 两侧各自 `torch.randn`（即使种子相同），NPU 的随机数序列和 CPU 不同，对比全是垃圾数据
2. **NPU 侧算完后要 `torch.npu.synchronize()` 再取值**，避免异步未完成
3. **易碎算子要隔离**。bicubic 在 NPU 上失败会破坏 allocator，同进程后续所有算子输出都不可信（见 §4.2），所以 bicubic 单独放子进程跑

结果读取：`max_abs_diff` 的合理量级——逐元素算子（LN/GELU/softmax）≤1e-6；大矩阵乘/卷积 fp32 累加噪声 0.01~0.1 属正常；出现"不可能的 diff"（如 sigmoid 输出 [0,1] 却 diff>1）基本就是进程被污染或输入不一致，不是算子本身的问题。

实测 31/32 通过，唯一失败的就是 bicubic。

## 4. 两个不适配点的原理与解法

### 4.1 SDPA：`can not cast format when output is input`

**现象**：`F.scaled_dot_product_attention` 在 310P 上必现 `ERR01007 OPS feature not supported`。

**排查过程**（按顺序都试过，均无效）：

```python
torch.npu.config.allow_internal_format = False   # 无效
from torch.nn.attention import sdpa_kernel, SDPBackend
with sdpa_kernel([SDPBackend.MATH]): ...          # 无效
```

结论：torch_npu 在 C++ 层整体劫持了 SDPA 分发，Python 侧 backend 选择根本不会生效。

**解法**：整个进程级 monkeypatch 成手写实现（等价于 math 后端）：

```python
import math, torch

def manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, **kw):
    L, E = query.shape[-2], query.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(E)
    attn = (query @ key.transpose(-2, -1)) * scale
    if is_causal:
        causal = torch.ones(L, query.shape[-2], dtype=torch.bool,
                            device=query.device).tril()
        attn = attn.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attn_mask, float("-inf"))
        else:
            attn = attn + attn_mask
    return attn.softmax(-1) @ value

torch.nn.functional.scaled_dot_product_attention = manual_sdpa
```

要点：
- transformers 模型内部都是运行时通过 `torch.nn.functional` 取函数，patch 模块属性即可全局生效
- 必须处理 `bool mask`（masked_fill）与 `float mask`（相加）两种语义，以及 `is_causal`
- 精度：fp32 diff ~2e-4，fp16 diff ~1e-4，整模型验证通过
- 代价：拆开的 attention 比融合算子慢；后续可用 `torch_npu.npu_fusion_attention` 等原生接口做性能替换

### 4.2 bicubic interpolate：`No supported Ops kernel ... [ResizeD]`

**现象**：`F.interpolate(mode="bicubic")` 报 ResizeD 无可用 kernel（bilinear 正常）。

**真正的坑**：这个失败**不是可恢复异常**——CANN 单算子编译失败会把当前 stream 的 allocator 状态破坏，之后同进程里**其它毫无关系的算子也会输出错乱数据**（比如 sigmoid 输出 [0,1] 的东西 diff 出 4.9）。这类"不可能的 diff"全是这个污染的连带现象，和算子本身无关。

**解法**：
- 业务上：bicubic 只出现在离线的位置编码预处理，放 CPU 跑；推理路径用 bilinear（NPU 支持）
- 测试上：易碎算子用子进程隔离探测：

```python
subprocess.run([sys.executable, __file__], env=dict(os.environ, BICUBIC_PROBE="1"))
```

## 5. 第二步：整模型验证

`scripts/04_model_forward.py` 分三步：

**step1 数值一致性**（随机权重，CPU vs NPU）——这一步抓到了两个结构性问题：

1. **config 嵌套传错会静默产出 NaN**。`Sam3VideoConfig`（`model_type=sam3_video`）内含 `detector_config`（真正的 `Sam3Config`）。如果把外层 config 直接传给 `Sam3Model(cfg)`，模型会构出一个结构错配的模型，**前向不报错但输出 NaN**。正确写法：

```python
cfg_full = AutoConfig.from_pretrained(dir)
cfg = cfg_full.detector_config          # 必须取内层
cfg._attn_implementation = "eager"      # 避开 HF sdpa 路径（配合 manual_sdpa 双保险）
model = Sam3Model(cfg)
```

2. **RoPE 表尺寸锁定**。SAM3 视觉 RoPE 的 cos/sin 在 `__init__` 按 `config.image_size=1008` 生成（5184 token），直接喂 336×336 会报 `size 576 vs 5184` 不匹配。**改分辨率必须改 config 重建模型**，不能只改输入。

修正后 @336 随机权重对比：全部 6 路输出与 CPU 一致（`pred_boxes` 4e-5、`pred_logits` 2.7e-4、`pred_masks` 3.8e-4、`semantic_seg` 1e-3）。

**step2 真实权重加载**：`Sam3VideoModel.from_pretrained(dir)` → 取 `.detector_model`。

**step3 原生分辨率前向 @1008**：

- 首跑 ~98.5s：CANN 在线编译算子（属正常现象，kernel 有缓存）
- 热身后平均 **2.405s/次，显存 7.50GB**，输出全部有限

## 6. 第三步：端到端

`scripts/05_e2e_probe.py`：真实图片 + 文本提示 → top-k 检测框。

注意视频版 processor（`AutoProcessor` 会加载 `Sam3VideoProcessor`）不收 text 参数，文本提示要用图像处理器 + tokenizer 手动组合：

```python
image_processor = AutoImageProcessor.from_pretrained(dir)
tokenizer = AutoTokenizer.from_pretrained(dir)
pixel_values = image_processor(images=img, return_tensors="pt")["pixel_values"]
text = tokenizer(["person"], return_tensors="pt", padding="max_length", max_length=32)
out = det(pixel_values=pixel_values.npu(),
          input_ids=text["input_ids"].npu(),
          attention_mask=text["attention_mask"].npu())
scores = out["pred_logits"][0].sigmoid()   # top-k 即检测结果置信度
```

实测（SAM3.1 架构图 + 提示 "diagram"）：top1 score **0.918**，框覆盖整图，语义正确。

## 7. 性能现状与优化路线

| 指标 | fp32 eager（当前） |
|---|---|
| 前向时延 @1008 | 2.4s（首跑 98s 为算子编译） |
| 显存 | 7.5GB（单卡 44GB，余量充足） |

后续优化方向（按性价比排序）：

1. **fp16/bf16**：算子级已验证 fp16 路径可用（linear/attention diff 达标），整模型半精度预计 2~3x
2. **attention 换原生融合**：`torch_npu.npu_fusion_attention` 替换 manual_sdpa
3. **torchair / ONNX→ATC om**：图模式整图下沉，消除算子调度开销，最终部署形态

## 8. SAM 家族扩展（SAM / SAM2 / MedSAM）

`ascend310p_compat` 兼容层打通后，SAM 家族其他模型只是"新的权重 + 少量新坑"：

| 模型 | transformers 类 | 新增坑 | 结果(fp32 @1024) |
|---|---|---|---|
| SAM (vit-huge) | `SamModel` | 无（补丁 1/3/附 已覆盖） | 1.56s / 7.3GB |
| SAM 2.1 | `Sam2Model` | 4D bicubic(pos-embed) 触发 ResizeD | 0.88s / 3.7GB |
| MedSAM | `SamModel` | processor 输出 `input_boxes` 为 float64 | 0.56s / 3.9GB |

通用测试脚本 `scripts/06_family_test.py`（`--model medsam|sam|sam2|sam3`）与端到端 `07_medsam_e2e.py`。

补充的坑与解法：

1. **SAM2 位置编码插值是 4D bicubic**（`modeling_sam2.py` `_get_pos_embed`），每前向必经。bicubic→bilinear 替代差异过大（实测 max diff 1.4，均值 0.68，不可接受），正确做法是 **CPU 计算 + 按形状缓存**（输入是固定权重 pos_embed，缓存命中后零开销）——即补丁 2 的最终形态
2. **激活函数拒绝非连续输入**：`SamLayerNorm(channels_first)` 内部 `permute → layer_norm → permute` 产出非连续张量，310P 的 GELU 直接报 `Only c10::MemoryFormat::Contiguous is supported`。解法是补丁 3（激活函数自动 `.contiguous()` 守卫）。注意最小复现时先确认张量的 is_contiguous——`transpose(2,3).reshape(...)` 后是连续的，单纯 reshape 不是触发点，触发的是 LayerNorm 的 permute 输出
3. **fp64 输入**：`SamProcessor` 输出 `input_boxes` 为 float64，310P 的 AsStrided 不支持 DT_DOUBLE，`.to(torch.float32)` 即可。定位方法：报错信息里 "data type DT_DOUBLE ... not supported" 直接指向 dtype 问题

## 9. FAQ

**Q: 新容器里 `torch.npu.device_count()` 为 0 / aclInit 报 507899 Resource Busy？**
A: 驱动按**容器命名空间**独占设备：一张卡被某个容器打开后即绑定，其他容器（包括新建的）无法初始化。dmesg 可见 `[uda_occupy_dev_by_ns] Conflict open udevid`。
判别哪张卡可认领：`npu-smi info` 里 **Hugepages-Usage 为 0/0 的卡从未建立过计算上下文，可被新容器认领**（实测验证）；有进程或 hugepages 已占满的卡属于现有容器。注意容器 stop 后重启会拿到新命名空间，若期间设备被其他容器认领则原容器也会初始化失败。

**Q: `Warning: kernel [isneginf.out] is not supported ... running on CPU`？**
A: torch_npu 对不支持的小算子自动回退 CPU，仅警告，结果正确，可忽略。

**Q: 为什么不能用 pip 装 torch？**
A: NPU 版 torch/torch_npu 与 CANN 版本强绑定，随镜像发行。pip 装的 CUDA 版会覆盖掉 NPU 支持。

**Q: transformers 报 tokenizers 版本错误？**
A: transformers 5.13 要求 `tokenizers>=0.22,<=0.23.0`，装 0.22.2 即可。

## 10. 复现清单

```bash
git clone https://github.com/green-hand-shandong/sam3-ascend-310p.git
cd sam3-ascend-310p
bash scripts/01_download_model.sh facebook/sam3 SAM3
bash scripts/02_run_container.sh
# 容器内：
pip install "transformers>=5.13.1" "tokenizers==0.22.2" safetensors pillow
python /workspace/scripts/03_op_test.py
python /workspace/scripts/04_model_forward.py
python /workspace/scripts/05_e2e_probe.py /data1/models/SAM3 <图片> "person"
```

期望输出见 `results/`。
