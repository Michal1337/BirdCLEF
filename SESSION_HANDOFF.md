# Session Handoff — SSM+CNN Blend Attempt (2026-04-28)

## TL;DR

Tried to lift LB from 0.93 (SSM-only) by blending in a focal-trained CNN.
First submission **regressed to 0.921** because the per-taxon weights overfit
the 708-row OOF — specifically Aves at 80% CNN where the OOF lift was only
+0.005 (within noise). Conservative weights are queued but not yet submitted.
AST is the real prize but Kaggle CPU is too slow without quantization, and
INT8 quantization is mostly working but blocked on a HuggingFace ONNX shape
inference bug.

---

## What was built

### New scripts
- `birdclef/scripts/_freeze_ast_to_onnx.py` — exports a fine-tuned AST
  checkpoint to ONNX (`(input_values) → logits` signature). Saves a
  `wrapper_config.json` sidecar with sample rates / max_length. Works.
- `birdclef/scripts/_quantize_ast_onnx.py` — INT8 dynamic quantization of
  the AST ONNX. **Workaround applied**: `quant_pre_process` step with
  `skip_symbolic_shape=True` (HF transformer exports break ORT's symbolic
  shape inference on Concat ops). Quantization itself works after that.
- `birdclef/scripts/_08c_dump_ast_val.py` — runs AST on labeled
  soundscapes, row-aligned to SSM OOF. Output: `ast_probs.npz`.
- `birdclef/scripts/_08d_dump_cnn_val.py` — same shape for the focal CNN.
  Output: `cnn_probs.npz`.
- `birdclef/scripts/_09_blend_search.py` — per-taxon convex-combination
  search across N members. Coarse grid (default step=0.1) to avoid
  overfitting the small OOF. Prints a paste-ready `_TAXON_WEIGHTS = {...}`
  block.

### Notebook patches (one-shot scripts, can be deleted)
- `_patch_lb_cnn_pytorch.py` — replaces LB notebook cell 26 with PyTorch
  eager CNN inference (~1-2s/file, fast enough for Kaggle).
- `_patch_lb_cnn_weights.py` — replaces cell 27's `_TAXON_WEIGHTS` dict.
  **Currently contains the BAD weights that produced LB 0.921.**
- `_patch_lb_ast_onnx_perch_style.py` — alternative cell 26 patch with
  ONNX AST inference. Not currently active in the notebook.

---

## What works

| component | status | LB / metric |
|---|---|---|
| SSM-only baseline (LB_0931_seed.ipynb stack) | ✅ | LB 0.93 |
| Stitched 5-fold OOF infrastructure (`_07b_dump_oof_probs`) | ✅ | 708 rows × 234 classes |
| Per-class audit (`_08b_per_class_audit`) | ✅ (after Unicode workaround `PYTHONIOENCODING=utf-8`) | overall SSM macro 0.8063 |
| Blend search (`_09_blend_search`) | ✅ | works for N≥2 members |
| AST ONNX export | ✅ | ~700 MB fp32 |
| AST INT8 quantization | ✅ (with `skip_symbolic_shape=True`) | ~330 MB int8 |
| CNN PyTorch inference on Kaggle | ✅ | ~1-2s/file |

## What didn't work

| attempt | result | root cause |
|---|---|---|
| **SSM+CNN per-taxon blend** | **LB 0.921 (regression -0.009)** | Per-taxon weights overfit the 708-row OOF. Aves at 80% CNN had only +0.005 OOF lift (within ±0.02 noise) but tanked LB. |
| AST eager PyTorch on Kaggle CPU | Too slow | 290s for 20 files = ~14.5s/file. Need quantization. |
| AST ONNX export → ORT symbolic shape inference | Crashed | `_merge_symbols` assertion on Concat op for the freshly-init 234-class head. Worked around by `skip_symbolic_shape=True`. |
| AST ONNX → quantize_dynamic without preprocessing | Crashed | Shape mismatch `(768) vs (234)` from stale shape annotations. Fixed by `quant_pre_process` step. |

---

## Authoritative numbers (from current 708-row OOF)

**Run `python -m birdclef.scripts._08b_per_class_audit` to refresh.** The CSV
in the repo before this session was stale (per-class AUCs differed from
sklearn-on-current-OOF by up to 0.39 — mean abs diff 0.02). Now refreshed.

### SSM per-taxon macro AUC (correct numbers — use these, not the blend search printout from prior session)

| taxon | n_seen / n_total | SSM macro |
|---|---|---|
| Amphibia | 17 / 35 | 0.8677 |
| Aves | 25 / 162 | 0.8990 |
| Insecta | 25 / 28 | 0.6565 |
| Mammalia | 3 / 8 | 0.8786 |
| Reptilia | 1 / 1 | 0.9699 |
| **Overall** | 71 / 234 | **0.8063** |

### CNN per-taxon (from prior blend search printout — verify by re-running blend search)

| taxon | SSM | CNN | per-taxon blend gain |
|---|---|---|---|
| Aves | 0.899 | ~0.856 | **+0.005** ← within noise |
| Amphibia | 0.868 | ~0.746 | n/a (SSM only is best) |
| Insecta | 0.657 | ~0.496 | **+0.05** ← real signal |
| Mammalia | 0.879 | ~0.885 | +0.02 ← mild |
| Reptilia | 0.970 | ~0.520 | n/a (SSM dominates) |

`cnn_probs.npz` is **not currently on this machine** — was generated on Hopper
during the blend search. Need to regenerate to re-verify CNN numbers.

---

## Current LB notebook state (`LB_0931_seed.ipynb`)

- **Cell 26**: PyTorch CNN inference (loads from
  `/kaggle/input/birdclef-cnn-best/best_model.pt`, fallback to
  `/kaggle/working/cnn_best_model.pt`). Variable kept as `ast_test_probs`
  for cell 27 compatibility.
- **Cell 27**: Has the BAD per-taxon weights from the regressed submission:
  ```python
  _TAXON_WEIGHTS = {
      "Aves":     (0.20, 0.80),  # ← BAD, should be (1.00, 0.00)
      "Amphibia": (1.00, 0.00),
      "Insecta":  (0.30, 0.70),
      "Mammalia": (0.10, 0.90),  # ← too aggressive, should be (0.50, 0.50)
      "Reptilia": (1.00, 0.00),
  }
  ```

### Recommended weights (not yet applied)
```python
_TAXON_WEIGHTS = {
    "Aves":     (1.00, 0.00),  # +0.005 OOF lift = noise → SSM only
    "Amphibia": (1.00, 0.00),  # SSM dominates already
    "Insecta":  (0.30, 0.70),  # +0.05 lift, real signal → keep CNN heavy
    "Mammalia": (0.50, 0.50),  # +0.02 mild lift → split the difference
    "Reptilia": (1.00, 0.00),  # SSM dominates
}
```

Decision rule used: **only put weight on CNN if the per-taxon OOF lift > 0.02
(twice the per-taxon AUC noise floor on 708 rows)**.

---

## Recommended next moves (in order)

1. **Patch cell 27 with conservative weights → submit.** Expected LB ~0.93
   (Aves restored) plus small Insecta/Mammalia gains. If this doesn't recover
   0.93, something else is regressed in the notebook (sanity-check by
   submitting SSM-only).

2. **Get AST working.** Bigger expected lift than CNN (SSM+AST projected
   blend OOF ~0.864 vs SSM+CNN actual 0.833). Two paths:
   - Re-quantize AST with current scripts and time it on Kaggle (current
     scripts work, just need to verify INT8 speedup is real on AVX2 CPUs).
   - Or try `torch.onnx.dynamo_export` for cleaner shape annotations
     (avoids the `quant_pre_process` workaround entirely).

3. **3-member blend** (SSM + CNN + AST) once AST works. `_09_blend_search`
   handles 3 members natively. Likely +0.005-0.01 over SSM+AST alone since
   CNN's Mammalia signal is decorrelated from AST's.

4. **Stitched OOF re-build with more labeled data** if available — biggest
   single thing that would shrink the OOF→LB gap. Currently 708 rows from 59
   files; doubling that would halve the per-taxon noise floor.

---

## Useful commands

```bash
# Refresh per-class audit (stale by default)
PYTHONIOENCODING=utf-8 python -m birdclef.scripts._08b_per_class_audit --top-k 30

# Re-run blend search (need cnn_probs.npz regenerated first)
python -m birdclef.scripts._08d_dump_cnn_val \
    --cnn-ckpt birdclef_example/outputs/focal/sota14_nfnet_lr7e-04_1e-04/best_model.pt
python -m birdclef.scripts._09_blend_search \
    --members ssm:birdclef/outputs/blend_search/oof/ssm_probs.npz \
              cnn:birdclef/outputs/blend_search/oof/cnn_probs.npz \
    --step 0.1

# Export + quantize AST
python -m birdclef.scripts._freeze_ast_to_onnx \
    --ast-ckpt birdclef_example/outputs/ast/ast_lr3e-05_e15/best_model.pt \
    --out-dir  birdclef_example/outputs/ast/ast_lr3e-05_e15_onnx --validate
python -m birdclef.scripts._quantize_ast_onnx \
    --in-onnx  birdclef_example/outputs/ast/ast_lr3e-05_e15_onnx/model.onnx \
    --out-onnx birdclef_example/outputs/ast/ast_lr3e-05_e15_onnx/model_int8.onnx \
    --validate
```

---

## Key insight (don't repeat this mistake)

**Per-taxon blend search on a small OOF latches onto noise.** Always check
that the per-taxon blend gain exceeds the per-taxon AUC noise floor before
trusting the weights. With 708 OOF rows and per-taxon class counts of 1-25,
the noise floor is ~±0.02 per taxon. **Require ≥0.02 OOF lift before putting
any weight on a member**, and even then prefer SSM-dominant splits unless the
lift is large (≥0.05).

The Aves trap: `_09_blend_search` reported a 0.904 best blend vs 0.899 SSM
alone — a +0.005 "lift" — and the script's grid happened to pick (0.20,
0.80) as optimal. Following blindly cost ~0.01 LB.
