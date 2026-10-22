# SAM Family on Ascend 310P (华为昇腾)

[中文文档](README.md) | English

Run the Meta **SAM family** — **SAM 3.1 / SAM 3**, **SAM 2.1**, **SAM (original ViT-H)** and **MedSAM** — on **Huawei Ascend 310P** (Atlas 300I Duo) with `torch_npu`.

## What works

| Model | Weights | Prompt | fp32 forward | Memory | E2E |
|---|---|---|---|---|---|
| SAM 3.1 / 3 (detector, 840M) | `facebook/sam3` (HF safetensors) | text + box | 2.4 s @1008 | 7.5 GB | top-1 score 0.918 |
| SAM 3.1 converted from Meta `sam3.1_multiplex.pt` | this repo's converter | text + box | 2.39 s @1008 | 7.5 GB | no NaN |
| SAM 2.1 (hiera-large, 217M) | `facebook/sam2.1-hiera-large` | point | 0.88 s @1024 | 3.7 GB | no NaN |
| SAM (vit-huge, 636M) | `facebook/sam-vit-huge` | box | 1.56 s @1024 | 7.3 GB | no NaN |
| MedSAM (vit-base, 94M) | `flaviagiammarino/medsam-vit-base` | box | 0.56 s @1024 | 3.9 GB | 99.9% lesion coverage |

Environment: CANN 9.1.0 · torch 2.10.0 · torch_npu 2.10.0.post4 · transformers 5.13.1 · fp32 eager.

## The three 310P workarounds (`ascend310p_compat/`)

1. **Fused SDPA is unavailable** — `F.scaled_dot_product_attention` raises
   `can not cast format when output is input`. Neither `allow_internal_format=False`
   nor `sdpa_kernel(SDPBackend.MATH)` helps (torch_npu intercepts dispatch in C++).
   → `enable_manual_sdpa()`: process-wide replacement with matmul+softmax (+finfo.min
   masking so fully-masked rows stay NaN-free). fp32 parity ~2e-4.
2. **Bicubic interpolate has no kernel** (`ResizeD`), and its failure *corrupts the
   process allocator* — every later op returns garbage. SAM2's pos-embed path hits this
   every forward; bilinear substitution is NOT acceptable (max diff 1.4).
   → `patch_interpolate()`: compute bicubic on CPU, cached by shape. 1D `linear`
   interpolate is rerouted to an exactly equivalent 4D bilinear call (diff 0.0).
3. **Activations reject non-contiguous input** — `SamLayerNorm(channels_first)`
   permutes → LayerNorm → permutes, and the following GELU raises
   `Only c10::MemoryFormat::Contiguous is supported`.
   → `patch_activations()`: transparent `.contiguous()` guard.

Also: `SamProcessor` emits `input_boxes` in **float64**; cast to float32
(310P `AsStrided` has no fp64 kernel).

## Quick start

```bash
# host: Ascend Docker Runtime + driver; pick an unclaimed NPU (Hugepages 0/0)
bash scripts/01_download_model.sh sam3        # or sam3.1 / sam2.1 / sam / medsam
bash scripts/02_run_container.sh              # CANN image, ASCEND_VISIBLE_DEVICES

# in container:
pip install "transformers>=5.13.1" "tokenizers==0.22.2" safetensors pillow
python scripts/03_op_test.py                              # operator sweep
python scripts/06_family_test.py --model sam3 --dir /data1/models/SAM3
python scripts/05_e2e_probe.py /data1/models/SAM3 image.jpg "person"
python scripts/07_medsam_e2e.py image.jpg x1 y1 x2 y2
```

Convert the Meta-format SAM3.1 checkpoint:

```bash
python scripts/08_convert_sam31.py \
    --checkpoint /data1/models/SAM3.1/sam3.1_multiplex.pt \
    --config-dir /data1/models/SAM3.1 \
    --output /data1/models/SAM3.1-hf
# expected: missing=0; unexpected=interactive/propagation neck copies + rope buffers (by design)
# note: sam3.1's TriNeck is 3-level — the script overrides scale_factors to [4.0, 2.0, 1.0]
```

Use the compat layer in your own code:

```python
import ascend310p_compat
ascend310p_compat.patch_all()   # before .to("npu")
model = model.to("npu")
```

## Repo layout

- `ascend310p_compat/` — reusable patch layer (`attention.py`, `interpolate.py`, `activations.py`, `env.py`)
- `scripts/` — 01 download · 02 container · 03 operator tests · 04 SAM3 full-model check · 05 SAM3 e2e · 06 family tests · 07 MedSAM e2e · 08 SAM3.1 converter
- `docs/tutorial.md` — full walkthrough (Chinese): methodology, root causes, debugging stories, FAQ
- `results/` — measured evidence
- `tests/` — CPU parity tests of the compat layer (no NPU required); CI config at `docs/ci.yml.example`

## Known limitations

- SAM3's vision RoPE tables are built for `config.image_size` at init — the model is size-locked
- The video tracker / session APIs are not covered yet; fp16 verified at operator level and on MedSAM
- NPU devices are exclusive per container namespace (see tutorial FAQ)

## License

Apache-2.0. The SAM3.1 key mapping in `scripts/08_convert_sam31.py` is adapted from
HuggingFace `transformers` `convert_sam3_to_hf.py` (Apache-2.0).
