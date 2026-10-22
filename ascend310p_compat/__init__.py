"""Ascend 310P 兼容层。

让依赖 `torch.nn.functional.scaled_dot_product_attention` 和 1D `interpolate` 的
PyTorch 模型（如 Meta SAM 3.1 / transformers SAM 家族）能在华为昇腾 310P
（Atlas 300I Duo）上以 eager 模式正确运行。

用法（两行接入，必须在模型上卡之前调用）:

    from ascend310p_compat import enable_manual_sdpa, patch_interpolate
    enable_manual_sdpa()          # 替换 SDPA 融合算子
    patch_interpolate()           # (可选) 3D linear 插值 -> 4D bilinear 等价实现
    model = model.to("npu")       # 之后正常推理

实测（CANN 9.1.0 / torch_npu 2.10.0.post4 / 310P3）:
    SDPA: fp32 max_abs_diff ~2e-4，fp16 ~1e-4（对照 CPU SDPA）
    1D linear interpolate: 4D bilinear 替代 bit 级等价 (diff 0.0)
    SAM 3.1 detector 整模型前向数值与 CPU 一致（见仓库 results/）
"""

from .activations import patch_activations
from .attention import enable_manual_sdpa, manual_sdpa
from .env import assert_npu_ready, print_env
from .interpolate import patch_interpolate

__version__ = "0.3.0"

__all__ = [
    "enable_manual_sdpa",
    "manual_sdpa",
    "patch_interpolate",
    "patch_activations",
    "patch_all",
    "assert_npu_ready",
    "print_env",
]


def patch_all(verbose=True):
    """一键打全部 310P 补丁（SDPA / interpolate / activations）。模型上卡前调用。

    用法:
        import ascend310p_compat
        ascend310p_compat.patch_all()
        model = model.to("npu")
    """
    enable_manual_sdpa(verbose=False)
    patch_interpolate(verbose=False)
    patch_activations(verbose=False)
    if verbose:
        print(f"[ascend310p_compat {__version__}] all patches applied: "
              "sdpa->manual, interpolate(bicubic->CPU cached, linear->bilinear), activations->contiguous guard")
