"""Ascend 310P 兼容层。

让依赖 `torch.nn.functional.scaled_dot_product_attention` 的 PyTorch 模型
（如 Meta SAM 3.1 / transformers 系列）能在华为昇腾 310P（Atlas 300I Duo）上
以 eager 模式正确运行。

用法（两行接入，必须在模型上卡之前调用）:

    from ascend310p_compat import enable_manual_sdpa
    enable_manual_sdpa()          # 替换 SDPA 融合算子
    model = model.to("npu")       # 之后正常推理

实测（CANN 9.1.0 / torch_npu 2.10.0.post4 / 310P3）:
    fp32 max_abs_diff ~2e-4，fp16 ~1e-4（对照 CUDA/CPU SDPA）
    SAM 3.1 detector 整模型前向数值与 CPU 一致（见仓库 results/）
"""

from .attention import enable_manual_sdpa, manual_sdpa
from .env import assert_npu_ready, print_env

__version__ = "0.1.0"

__all__ = ["enable_manual_sdpa", "manual_sdpa", "assert_npu_ready", "print_env"]
