"""bicubic interpolate 的 310P 适配（CPU 回退 + 缓存）。

现象: bicubic(ResizeD 算子) 在 310P/CANN 9.1 全线无可用 kernel
(SAM2 位置编码插值、SAM3 尺寸变换都会触发), 且失败会破坏同进程 allocator。
bicubic -> bilinear 替代差异过大(实测 1.4 vs 均值 0.68), 不可接受。

解法: patch_interpolate() 拦截所有 bicubic 调用 -> CPU 计算 + 按形状缓存
(输入是固定权重 pos_embed 时命中缓存, 零重复计算)。
"""
import torch


def patch_interpolate(verbose=True):
    """进程级替换 F.interpolate：bicubic -> CPU 计算 + 形状缓存；其余走原生。

    同时拦截 3D+linear(见 interpolate.py, SAM rel-pos 分支)。
    """
    _orig = torch.nn.functional.interpolate
    _cache = {}

    def _bicubic_on_cpu(input, size):
        key = (tuple(input.shape), tuple(size) if isinstance(size, (tuple, list)) else size,
               str(input.dtype), str(input.device))
        if key in _cache:
            return _cache[key].to(input.device, copy=True)
        out = _orig(input.detach().cpu(), size=size, mode="bicubic").to(
            device=input.device, dtype=input.dtype)
        if len(_cache) < 16:  # 防止动态 shape 场景无限增长
            _cache[key] = out.cpu()
        return out

    def _linear_via_bilinear(input, size):
        out = _orig(
            input.unsqueeze(-1), size=(size, 1), mode="bilinear", align_corners=False
        )
        return out.squeeze(-1)

    def interpolate(input, size=None, scale_factor=None, mode="nearest",
                    align_corners=None, recompute_scale_factor=None, **kw):
        if isinstance(input, torch.Tensor):
            if mode == "bicubic":
                if size is not None:
                    return _bicubic_on_cpu(input, size)
                return _orig(input, size=size, scale_factor=scale_factor, mode=mode,
                             align_corners=align_corners,
                             recompute_scale_factor=recompute_scale_factor, **kw)
            if (
                mode == "linear"
                and input.dim() == 3
                and isinstance(size, int)
            ):
                return _linear_via_bilinear(input, size)
        return _orig(input, size=size, scale_factor=scale_factor, mode=mode,
                     align_corners=align_corners,
                     recompute_scale_factor=recompute_scale_factor, **kw)

    torch.nn.functional.interpolate = interpolate
    if verbose:
        print("[ascend310p_compat] interpolate: bicubic->CPU(cached), 3D linear->4D bilinear (310P workaround)")
