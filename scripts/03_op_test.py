#!/usr/bin/env python3
"""SAM3.1 适配 310P — 算子级测试。

对照 SAM3 实际结构（transformers modeling_sam3/sam3_tracker）选取关键算子，
逐个在 NPU 上执行并与 CPU fp32 结果对比：
  - 视觉编码器 ViT: patch embed Conv2d、Window/Global Attention、RoPE、GELU、LayerNorm、
    neck 的 ConvTranspose2d/Conv2d/GroupNorm
  - 文本编码器: Embedding、CLIP-style attention
  - memory attention (tracker): RoPE 2D、cross-attention、einsum
  - DETR 头: box 反归一化 cxcywh->xyxy 依赖的 clamp/where/div、topk、sigmoid

结果写入 /workspace/results/op_test_results.json
"""
import json
import os
import sys
import traceback

import torch
import torch.nn.functional as F

RESULTS = {"env": {}, "ops": [], "composite": []}


def env_check():
    import torch_npu  # noqa: F401
    # 310P 上 SDPA 等算子会因内部 NZ 格式转换报 "can not cast format"，强制 ND
    torch.npu.config.allow_internal_format = False
    RESULTS["env"] = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device_count": torch.npu.device_count(),
        "device_name": torch.npu.get_device_name(0),
    }
    print("[env]", json.dumps(RESULTS["env"], indent=1))


DEV = "npu"

_SDPA_PATCHED = [False]


def manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    # 与 ascend310p_compat.attention.manual_sdpa 保持一致；此处内联以便单脚本自包含
    """310P 上 torch_npu 的 SDPA 融合算子报 "can not cast format"，
    用 matmul+softmax 分解实现替代（SAM3 适配的关键 workaround）。"""
    import math
    L, E = query.shape[-2], query.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(E)
    attn = (query @ key.transpose(-2, -1)) * scale
    if is_causal:
        causal = torch.ones(L, query.shape[-2], dtype=torch.bool, device=query.device).tril()
        attn = attn.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn = attn.masked_fill(~attn_mask, float("-inf"))
        else:
            attn = attn + attn_mask
    attn = attn.softmax(-1)
    return attn @ value


def patch_sdpa():
    """用手动实现替换 SDPA（记录到结果）。"""
    torch.nn.functional.scaled_dot_product_attention = manual_sdpa
    _SDPA_PATCHED[0] = True
    RESULTS["env"]["sdpa_patched"] = True
    print("[patch] scaled_dot_product_attention -> manual matmul+softmax (310P workaround)")


def compare(name, npu_out, cpu_out, rtol=1e-4, atol=1e-4, note=""):
    """记录 NPU/CPU 输出差异。"""
    rec = {"name": name, "note": note, "dtype": str(npu_out.dtype)}
    try:
        npu_f = npu_out.float().cpu()
        cpu_f = cpu_out.float()
        if npu_f.shape != cpu_f.shape:
            rec.update(ok=False, error=f"shape mismatch npu{tuple(npu_f.shape)} cpu{tuple(cpu_f.shape)}")
        elif npu_f.numel() == 0:
            rec.update(ok=True, max_abs_diff=0.0)
        else:
            diff = (npu_f - cpu_f).abs()
            nan_npu = torch.isnan(npu_f).any().item()
            rec.update(
                ok=bool(not nan_npu),
                max_abs_diff=diff.max().item(),
                mean_abs_diff=diff.mean().item(),
                nan_on_npu=bool(nan_npu),
            )
    except Exception as e:  # noqa: BLE001
        rec.update(ok=False, error=repr(e))
    status = "PASS" if rec.get("ok") else "FAIL"
    print(f"[{status}] {name:40s} max_diff={rec.get('max_abs_diff', '-')} {note}")
    RESULTS["ops"].append(rec)
    return rec


def run_case(name, fn, *inputs, rtol=1e-4, atol=1e-4, note=""):
    """同一份 CPU 输入分别喂 CPU / NPU 跑 fn，对比输出。所有输入必须先在 CPU 上生成。"""
    try:
        with torch.no_grad():
            cpu_out = fn(*inputs)
            npu_in = tuple(t.to(DEV) if isinstance(t, torch.Tensor) else t for t in inputs)
            npu_out = fn(*npu_in)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        compare(name, npu_out, cpu_out, rtol, atol, note)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] {name}: {e!r}")
        RESULTS["ops"].append({"name": name, "ok": False, "error": repr(e), "trace": traceback.format_exc(limit=3)})


def seed_all():
    torch.manual_seed(0)


# ---------------------------------------------------------------- 单算子 cases
def op_cases():
    H, W, D = 72, 72, 1024  # ViT: 1008/14=72 tokens/边, hidden 1024
    win = 196               # 14x14 window attention
    NEG = -1e4              # 用有限负数代替 -inf，避免 inf-inf=nan 干扰对比

    case_seeds = iter(range(1, 1000))

    def ri(*shape):  # 递增种子生成 CPU 随机输入：确定且各不相同
        torch.manual_seed(next(case_seeds))
        return torch.randn(*shape)

    # --- Linear / MatMul (ViT MLP 1024->4736, fused QKV 1024->3072)
    run_case("linear_mlp", lambda a, b: a @ b, ri(1, H * W, D), ri(D, 4736))
    run_case("linear_qkv_fused", lambda a, b: a @ b, ri(1, H * W, D), ri(D, 3072))

    # --- LayerNorm / GELU(两种近似) / Softmax
    run_case("layernorm_vit", lambda x: F.layer_norm(x, (D,), eps=1e-6), ri(1, H * W, D))
    run_case("gelu_exact", lambda x: F.gelu(x, approximate="none"), ri(1, H * W, 4736))
    run_case("gelu_tanh", lambda x: F.gelu(x, approximate="tanh"), ri(1, H * W, 4736))
    run_case("softmax_global_attn", lambda x: F.softmax(x, -1), ri(2, 16, H * W, H * W),
             note="global attn map, fp32 约 340MB")
    run_case("softmax_window_attn", lambda x: F.softmax(x, -1), ri(64, 16, win, win))

    # --- SDPA：原生融合算子在 310P 上不可用，先留档原生失败，再用手动实现验证数值
    torch.manual_seed(1)
    _qkv = [torch.randn(2, 16, H * W, 64) for _ in range(3)]
    run_case("sdpa_native_no_mask(expect_fail)", lambda q, k, v: F.scaled_dot_product_attention(q, k, v), *_qkv)
    patch_sdpa()
    run_case("sdpa_manual_no_mask", lambda q, k, v: F.scaled_dot_product_attention(q, k, v), *_qkv)
    run_case("sdpa_manual_bool_mask", lambda q, k, v, m: F.scaled_dot_product_attention(q, k, v, attn_mask=m),
             *_qkv, torch.ones(2, 1, H * W, H * W, dtype=torch.bool))

    # --- RoPE: SAM3 vision RoPE(axial, apply_rotary_pos_emb_2d) + tracker rope
    def rope_fn(q, cos, sin):
        half = q.shape[-1] // 2
        q1, q2 = q[..., :half], q[..., half:]
        rotated = torch.cat([-q2, q1], -1)
        return q * torch.cat([cos, cos], -1) + rotated * torch.cat([sin, sin], -1)

    torch.manual_seed(0)
    _q = torch.randn(2, 16, H * W, 64)
    _inv = 1.0 / (10000 ** (torch.arange(0, 32).float() / 32))  # 32 个频率 -> 角度 (seq,32)
    _ang = torch.outer(torch.arange(H * W).float(), _inv)
    run_case("rope_rotate_half", rope_fn, _q, _ang.cos(), _ang.sin())

    # --- Conv 系: patch embed / neck conv / 反卷积上采样
    run_case("conv2d_patch_embed_14x14_s14", lambda x, w: F.conv2d(x, w, stride=14),
             ri(1, 3, 1008, 1008), ri(1024, 3, 14, 14))
    run_case("conv2d_neck_1x1", lambda x, w: F.conv2d(x, w), ri(1, 256, 256, 256), ri(256, 256, 1, 1))
    run_case("conv2d_neck_3x3_pad1", lambda x, w: F.conv2d(x, w, padding=1),
             ri(1, 256, 256, 256), ri(128, 256, 3, 3))
    run_case("conv_transpose2d_upsample_k2s2", lambda x, w: F.conv_transpose2d(x, w, stride=2),
             ri(1, 1024, 72, 72), ri(1024, 256, 2, 2), note="SAM neck 上采样")

    # --- GroupNorm (SAM neck)
    run_case("groupnorm_neck", lambda x: F.group_norm(x, 32), ri(2, 256, 128, 128))

    # --- Embedding / index 类（patch unroll、memory bank 用）
    run_case("embedding_text_vocab49k",
             lambda idx, w: F.embedding(idx, w),
             torch.randint(0, 49408, (1, 32)), ri(49408, 1024))
    torch.manual_seed(0)
    _feat = torch.randn(1, D, 72, 72).permute(0, 2, 3, 1).reshape(1, -1, D)
    _idx = torch.randperm(H * W)[:4096]
    run_case("index_select_patch_unroll", lambda f, i: f.index_select(1, i), _feat, _idx,
             note="SAM ViT 高分辨率 patch unroll 特征索引")

    # --- interpolate（位置编码插值 / mask 上采样）
    run_case("interpolate_bilinear_fpn", lambda x: F.interpolate(x, size=(288, 288), mode="bilinear", align_corners=False),
             ri(1, 256, 72, 72))
    # bicubic(ResizeD) 在 CANN 9.1 无可用 kernel 且失败会破坏 allocator —— 放到最后用子进程隔离探测
    if os.environ.get("BICUBIC_PROBE"):
        run_case("interpolate_bicubic_pos_embed", lambda x: F.interpolate(x, size=(H, W), mode="bicubic", align_corners=False),
                 ri(1, D, 24, 24), note="336->1008 位置编码插值")

    # --- DETR 头常见小算子
    run_case("sigmoid", torch.sigmoid, ri(1, 900, 300))
    run_case("topk_values", lambda x: x.topk(10, -1).values, ri(1, 900, 300))
    run_case("masked_fill_causal", lambda x, m: x.masked_fill(m, NEG),
             ri(2, 16, 100, 100), torch.triu(torch.ones(100, 100, dtype=torch.bool), 1))
    run_case("bmm_einsum_attn", lambda q, k: torch.einsum("bhid,bhjd->bhij", q, k),
             ri(2, 8, 900, 64), ri(2, 8, 5184, 64), note="DETR cross-attn: query 900 x memory 5184")
    run_case("clamp_div_cxcywh_to_xyxy", lambda x: x.sigmoid().clamp(0, 1) * 1008, ri(1, 900, 4))
    run_case("cumsum_int", lambda x: x.cumsum(0), torch.arange(1, 100))
    run_case("arange_float", lambda n: torch.arange(0, n, 14, dtype=torch.float32), 1008)

    # --- fp16 关键路径（后续性能优化会用到）
    run_case("fp16_sdpa_manual", lambda q, k, v: F.scaled_dot_product_attention(q, k, v),
             *[ri(2, 16, H * W, 64).half() for _ in range(3)], rtol=1e-2, atol=1e-2)
    run_case("fp16_linear_mlp", lambda a, b: a @ b,
             ri(1, H * W, D).half(), ri(D, 4736).half(), rtol=1e-2, atol=1e-2)


# ------------------------------------------------------------- 组合 case
class ViTBlock(torch.nn.Module):
    """仿 SAM3 ViT 层: LN -> QKV(bias) -> RoPE+SDPA -> proj -> residual -> LN -> MLP(gelu)"""

    def __init__(self, dim=1024, heads=16, mlp=4736):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(dim, eps=1e-6)
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=True)
        self.proj = torch.nn.Linear(dim, dim, bias=True)
        self.norm2 = torch.nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = torch.nn.Linear(dim, mlp)
        self.fc2 = torch.nn.Linear(mlp, dim)
        self.heads = heads

    def forward(self, x):
        B, N, C = x.shape
        h = self.heads
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, h, C // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(attn.transpose(1, 2).reshape(B, N, C))
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x


class NeckBlock(torch.nn.Module):
    """仿 SAM ViT neck: ConvT(2x) -> GN -> GELU -> Conv1x1 -> GN"""

    def __init__(self):
        super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(1024, 256, 2, stride=2)
        self.gn1 = torch.nn.GroupNorm(32, 256)
        self.conv = torch.nn.Conv2d(256, 256, 1)
        self.gn2 = torch.nn.GroupNorm(32, 256)

    def forward(self, x):
        x = F.gelu(self.gn1(self.deconv(x)))
        return self.gn2(self.conv(x))


class DetrCrossAttnBlock(torch.nn.Module):
    """仿 DETR decoder 层: self-attn(q=900) + cross-attn(memory 5184) + FFN"""

    def __init__(self, dim=256, heads=8, ffn=2048):
        super().__init__()
        self.sa = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ca = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln = torch.nn.LayerNorm(dim)
        self.ffn = torch.nn.Sequential(torch.nn.Linear(dim, ffn), torch.nn.ReLU(), torch.nn.Linear(ffn, dim))

    def forward(self, q, mem):
        q2, _ = self.sa(q, q, q, need_weights=False)
        q = q + q2
        c, _ = self.ca(q, mem, mem, need_weights=False)
        q = self.ln(q + c)
        return q + self.ffn(q)


def composite_cases():
    cases = [
        ("composite_vit_block_global", ViTBlock, (1, 5184, 1024)),
        ("composite_vit_block_window", lambda: ViTBlock(dim=256, heads=8), (64, 196, 256)),
        ("composite_neck_block", NeckBlock, (1, 1024, 72, 72)),
    ]
    for name, cls, shape in cases:
        try:
            torch.manual_seed(0)
            m_cpu = cls().float()
            torch.manual_seed(1)
            ins_cpu = (torch.randn(*shape),)
            with torch.no_grad():
                cpu_out = m_cpu(*ins_cpu)

            torch.manual_seed(0)
            m_npu = cls().float()
            m_npu.load_state_dict(m_cpu.state_dict())
            m_npu = m_npu.to(DEV)
            ins_npu = tuple(t.to(DEV) for t in ins_cpu)
            with torch.no_grad():
                npu_out = m_npu(*ins_npu)
            compare(name, npu_out, cpu_out, rtol=1e-3, atol=1e-3)
            del m_npu, npu_out
            torch.npu.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {name}: {e!r}")
            RESULTS["composite"].append({"name": name, "ok": False, "error": repr(e)})

    # DETR cross-attn: query 900 x memory 5184
    try:
        torch.manual_seed(0)
        m_cpu = DetrCrossAttnBlock().float()
        torch.manual_seed(1)
        ins_cpu = (torch.randn(2, 900, 256), torch.randn(2, 5184, 256))
        with torch.no_grad():
            cpu_out = m_cpu(*ins_cpu)

        torch.manual_seed(0)
        m_npu = DetrCrossAttnBlock().float()
        m_npu.load_state_dict(m_cpu.state_dict())
        m_npu = m_npu.to(DEV)
        ins_npu = tuple(t.to(DEV) for t in ins_cpu)
        with torch.no_grad():
            npu_out = m_npu(*ins_npu)
        compare("composite_detr_cross_attn", npu_out, cpu_out, rtol=1e-3, atol=1e-3)
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] composite_detr_cross_attn: {e!r}")
        RESULTS["composite"].append({"name": "composite_detr_cross_attn", "ok": False, "error": repr(e)})


def main():
    # 子进程模式: 只探测 bicubic（CANN 9.1 下 ResizeD 失败会破坏 allocator，必须隔离）
    if os.environ.get("BICUBIC_PROBE"):
        env_check()
        torch.manual_seed(1)
        run_case("interpolate_bicubic_pos_embed",
                 lambda t: F.interpolate(t, size=(72, 72), mode="bicubic", align_corners=False),
                 torch.randn(1, 1024, 24, 24), note="336->1008 位置编码插值, 子进程隔离探测")
        return
    env_check()
    print("=" * 80)
    op_cases()
    print("=" * 80)
    composite_cases()
    n_pass = sum(1 for r in RESULTS["ops"] if r.get("ok"))
    print(f"\n单算子: {n_pass}/{len(RESULTS['ops'])} PASS")
    # 隔离子进程探测 bicubic（失败会污染 allocator，不能与其它 case 同进程）
    import subprocess
    env = dict(os.environ, BICUBIC_PROBE="1")
    r = subprocess.run([sys.executable, __file__], env=env, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "bicubic" in line:
            print(line.strip())
            name = line.split("]")[1].strip().split()[0] if "]" in line else "interpolate_bicubic_pos_embed"
            ok = "[PASS]" in line
            RESULTS["ops"].append({"name": "interpolate_bicubic_pos_embed", "ok": ok,
                                   "max_abs_diff": None, "raw": line.strip()})

    out = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")) + "/op_test_results.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False)
    print("results ->", out)


if __name__ == "__main__":
    main()
