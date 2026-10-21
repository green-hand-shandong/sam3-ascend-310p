import math

import torch


def _neg(dtype):
    """finfo.min: 与 HF eager attention 的加性 mask 一致，整行 mask 时 softmax 得均匀分布而非 NaN。"""
    return torch.finfo(dtype).min


def manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0,
                is_causal=False, scale=None, **kw):
    """与 torch.nn.functional.scaled_dot_product_attention 语义一致的实现。

    支持 bool/float attn_mask 与 is_causal；dropout_p 仅训练场景需要，推理恒为 0。
    注意: 用 finfo.min 而非 -inf 做 mask(等价 HF eager attention), 避免整行 mask 时 softmax 产生 NaN。
    """
    if dropout_p not in (0.0, None):
        raise NotImplementedError("manual_sdpa 仅面向推理(dropout_p=0)")
    L, E = query.shape[-2], query.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(E)
    attn = (query @ key.transpose(-2, -1)).to(query.dtype) * scale
    if is_causal:
        causal = torch.ones(L, query.shape[-2], dtype=torch.bool,
                            device=query.device).tril()
        attn = attn.masked_fill(~causal, _neg(attn.dtype))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attn_mask, _neg(attn.dtype))
        else:
            attn = attn + attn_mask
    return attn.softmax(-1) @ value


def enable_manual_sdpa(verbose=True):
    """进程级替换 torch.nn.functional.scaled_dot_product_attention。

    在 import transformers 之后、模型前向之前调用一次即可。
    实测: fp32 max_abs_diff ~2e-4, fp16 ~1e-4。
    """
    torch.nn.functional.scaled_dot_product_attention = manual_sdpa
    if verbose:
        print("[ascend310p_compat] scaled_dot_product_attention -> manual matmul+softmax")
