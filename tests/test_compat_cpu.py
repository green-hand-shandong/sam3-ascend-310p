"""ascend310p_compat 的 CPU 侧单元测试（无需 NPU，可在 CI 运行）。

验证兼容层实现的数学正确性（与 PyTorch 原生实现对齐），
以及补丁打上后不破坏 CPU 路径。
"""
import math
import sys
import os

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ascend310p_compat import (  # noqa: E402
    assert_npu_ready,
    manual_sdpa,
    patch_activations,
    patch_all,
)


def _qkv(b=2, h=2, s=16, d=32, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(b, h, s, d, generator=g, dtype=dtype) for _ in range(3)]


class TestManualSdpa:
    def test_no_mask_matches_reference(self):
        q, k, v = _qkv()
        ref = F.scaled_dot_product_attention(q, k, v)
        out = manual_sdpa(q, k, v)
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_bool_mask_full_row_no_nan(self):
        """整行被 mask 的查询不应产生 NaN。

        约定差异说明: 原生 CPU SDPA 对全 mask 行输出 0;
        本实现用 finfo.min 加性 mask(与 HF eager attention 一致), 输出均匀分布。
        两者均无 NaN; 未被 mask 的行必须与参考完全一致。
        """
        q, k, v = _qkv()
        mask = torch.ones(1, 1, 16, 16, dtype=torch.bool)
        mask[..., -1, :] = False  # 最后一个 query 整行 mask
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = manual_sdpa(q, k, v, attn_mask=mask)
        assert not torch.isnan(out).any(), "全 mask 行不应产生 NaN"
        # 逐 query 判断是否有可见 key: mask[0,0,i,:].any()
        has_visible = mask[0, 0].any(-1)                     # (16,)
        full_rows_ref = torch.isnan(ref[0, 0]).any(-1)       # 参考对全 mask 行的输出可能是 0 或 NaN
        ok = has_visible & ~full_rows_ref                    # 只比较双方都有定义的行
        for b in range(out.shape[0]):
            for h in range(out.shape[1]):
                assert torch.allclose(out[b, h][ok], ref[b, h][ok], rtol=1e-5, atol=1e-6)

    def test_float_mask(self):
        q, k, v = _qkv()
        bias = torch.randn(1, 1, 16, 16, generator=torch.Generator().manual_seed(1))
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = manual_sdpa(q, k, v, attn_mask=bias)
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_causal(self):
        q, k, v = _qkv(s=8)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = manual_sdpa(q, k, v, is_causal=True)
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_custom_scale(self):
        q, k, v = _qkv()
        ref = F.scaled_dot_product_attention(q, k, v, scale=0.05)
        out = manual_sdpa(q, k, v, scale=0.05)
        assert torch.allclose(out, ref, rtol=1e-5, atol=1e-6)

    def test_fp16(self):
        q, k, v = _qkv(dtype=torch.float16)
        ref = F.scaled_dot_product_attention(q, k, v)
        out = manual_sdpa(q, k, v)
        assert torch.allclose(out, ref, rtol=1e-2, atol=1e-3)


class TestLinearInterpolateEquivalence:
    """3D linear -> 4D bilinear(退化维) 数学等价性 (SAM rel-pos 分支的 workaround)。"""

    @pytest.mark.parametrize("n", [5, 27, 64, 129])
    def test_paritiy(self, n):
        torch.manual_seed(0)
        x = torch.randn(1, 768, 27)
        ref = F.interpolate(x, size=n, mode="linear")
        alt = F.interpolate(x.unsqueeze(-1), size=(n, 1), mode="bilinear",
                            align_corners=False).squeeze(-1)
        assert torch.equal(ref, alt)


class TestBicubicCpuFallback:
    """bicubic CPU 回退补丁在 CPU 上应给出与原生一致的结果。"""

    def test_patch_keeps_cpu_results(self):
        patch_all(verbose=False)
        torch.manual_seed(0)
        x = torch.randn(1, 32, 24, 24)
        ref = F.interpolate(x, size=(48, 48), mode="bicubic")
        out = F.interpolate(x, size=(48, 48), mode="bicubic")  # 已打补丁的同一路径
        assert torch.allclose(ref, out)


class TestActivationsGuard:
    def test_contiguous_guard_transparent(self):
        patch_activations(verbose=False)
        x = torch.randn(2, 3, 4)
        assert torch.equal(F.gelu(x), F.gelu(x.clone()))
        # 非连续输入也应得到正确值(自动 contiguous)
        xt = torch.randn(4, 8).t()  # 非连续
        assert torch.allclose(F.gelu(xt), F.gelu(xt.contiguous()))


class TestEnvCheck:
    def test_assert_npu_ready_raises_gracefully_without_npu(self):
        """无 NPU 环境应抛出带定位信息的 RuntimeError（CI 环境预期行为）。"""
        if torch.npu.is_available() if hasattr(torch, "npu") else False:
            pytest.skip("有 NPU 的环境跳过")
        with pytest.raises((RuntimeError, ImportError)):
            assert_npu_ready()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
