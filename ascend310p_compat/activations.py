"""非连续张量守卫（310P 适配）。

现象: 310P 上部分算子(如 GELU)拒绝非连续输入,
报 "Only c10::MemoryFormat::Contiguous is supported for creating a npu tensor"。
触发链: SAM/SAM2/MedSAM 的 SamLayerNorm(channels_first) 内部
permute -> layer_norm -> permute 产出非连续张量, 随后的 GELU 即崩。

解法: patch_activations() 给激活函数加自动 .contiguous() 守卫
(非连续才拷贝, 连续零开销)。
"""
import torch


def patch_activations(verbose=True):
    """给 F.gelu / F.relu / F.silu 加非连续输入守卫。"""
    _orig_gelu = torch.nn.functional.gelu
    _orig_relu = torch.nn.functional.relu
    _orig_silu = torch.nn.functional.silu

    def gelu(input, *a, **kw):
        if input.dim() >= 3 and not input.is_contiguous():
            input = input.contiguous()
        return _orig_gelu(input, *a, **kw)

    def relu(input, *a, **kw):
        if input.dim() >= 3 and not input.is_contiguous():
            input = input.contiguous()
        return _orig_relu(input, *a, **kw)

    def silu(input, *a, **kw):
        if input.dim() >= 3 and not input.is_contiguous():
            input = input.contiguous()
        return _orig_silu(input, *a, **kw)

    torch.nn.functional.gelu = gelu
    torch.nn.functional.relu = relu
    torch.nn.functional.silu = silu
    if verbose:
        print("[ascend310p_compat] activations: contiguous guard (310P workaround)")
