# BirdCLEF 2026 — Strategy & Action Plan

> Current state: public LB **0.93**, local OOF macro‑AUC ≈ **0.91** (best sweep row: `low_prior_hrm_blend_f`, [outputs/sweep4/oof_fullstack_sweep_results.csv](outputs/sweep4/oof_fullstack_sweep_results.csv)). Pipeline = frozen Perch‑v2 ONNX embeddings + logits → LightProtoSSM (bidirectional Mamba + cross‑attn + per‑class prototypes) blended with per‑class MLP probes + site/hour prior → ResidualSSM correction → temperature scaling → file‑confidence, rank and adaptive smoothing → per‑class isotonic threshold. Training data = labeled `train_soundscapes` only; `train_audio` (Xeno‑Canto) is untouched except for the unmapped‑head fine‑tune in [perch_onnx_head_ft_no_signal.py](birdclef_example/perch_onnx_head_ft_no_signal.py).

Reference points for the season:
- Best public‑LB single notebook = **Perch‑v2 only, 0.908** ([yashanathaniel](https://www.kaggle.com/code/yashanathaniel/birdclef-2026-perch-v2-0-908)). You are already **+0.02 LB above the strongest public baseline** — the post‑Perch stack is working.
- Public SED baseline = 0.862, starter CNN = 0.79.
- BirdCLEF 2024 winner = 0.69 private, **2025 winner ≈ 0.922 private** (Babych), **2025 2nd = 0.928 private** (Sydorskyy). Both used trained SED + multi‑round pseudo‑labeling.

---

## 1. Gap analysis vs. winning recipes (2023–2025)

| Ingredient | Top‑5 2023–25 | You | Status |
|---|---|---|---|
| SED + EfficientNet/NFNet/ConvNeXt backbone trained from ImageNet | Universal (all winners 2021–25) | ❌ Only frozen Perch‑v2 embeddings | **Biggest gap** |
| Xeno‑Canto focal pretraining → fine‑tune on soundscapes | Universal | ❌ `train_audio` unused | **Big gap** |
| Multi‑round pseudo‑labeling on unlabeled soundscapes (noisy‑student) | 2024 #1/#2/#3, 2025 #1/#2/#3 | ❌ Not done | **Biggest gap** |
| FocalBCE (or BCE+Focal mean) | 2024 #2, all 2025 top‑5 | ⚠ `focal_gamma=2.5` is in the CFG but the active loss is BCE + pos_weight + distill MSE ([sota_oof_two_pass_ssm_advanced_pp.py:770](birdclef_example/sota_oof_two_pass_ssm_advanced_pp.py#L770)) | Drop‑in win |
| Temporal smoothing ([0.1,0.2,0.4,0.2,0.1] Gaussian or similar) | 2024 #1/#3 universal | ✅ You have `adaptive_delta_smooth` + `smooth_predictions` | Tune kernel |
| Soundscape‑wide probability boost (if species fires anywhere in file, lift everywhere) | 2024 #1/#3 | ⚠ `file_confidence_scale` is a soft version | Try hard boost |
| Secondary‑label loss *masked* | 2024 +0.01 | N/A (no training of own backbone) | — |
| TTA ±2.5 s shift | 2025 #1 (+0.012) | ⚠ You roll within 12‑window grid (shift=1 is 5 s, not 2.5 s) | Sub‑window TTA |
| Model‑diverse small ensemble (3–14 ONNX models) | Universal | ❌ Single Perch + single SSM stack | Large gap |
| Validation grouped by *site/session/date/recordist*, not just filename | 2025 winners' explicit advice | ❌ GroupKFold on `filename` only | **Fix first** |

TL;DR — your Perch+SSM stack is a strong *head*, but the field's +0.02–0.04 private‑LB wins all came from a **trained SED backbone + iterative pseudo‑labeling**, neither of which you have.

---

## 2. Most promising approaches (ranked by expected LB gain)

### A. Train your own SED model on `train_audio` + labeled `train_soundscapes` — highest ceiling
Every single BirdCLEF winner 2021–2025 trained a SED (PANN attention head on mel‑spectrogram CNN) from scratch. Frozen foundation models **never won**. Recommended recipe (convergent across years):

- Backbone: **`tf_efficientnetv2_s_in21k`** or **`eca_nfnet_l0`** (2023/25 winners' pick). Keep a second, more diverse backbone (ConvNeXt‑Tiny) for ensembling.
- Mel: 32 kHz, 128–256 mel bins, hop 320–512, window 2048, log‑mel with PCEN variant for diversity.
- **Sampling**: 5 s random crops; for training clips <5 s pad; for Xeno‑Canto, bias sampling to the first 5–30 s where the primary label is most likely present.
- **Targets**: multi‑hot primary ∪ secondary, but **mask the secondary‑label loss** (copy 2024 3rd: +0.01).
- **Loss**: **FocalBCE** (α≈0.25, γ≈2), not plain BCE with pos_weight. Your current `focal_gamma` in CFG is never consumed.
- **Augs** (stable across years): mixup (α≈0.5, label=max), SpecAugment (2 time + 2 freq masks), background mix from `ff1010_nocall` / `train_soundscapes` no‑call windows, random gain, pink/white noise. These shrink the focal‑vs‑soundscape domain gap — the single biggest aug lever.
- Export to **ONNX/OpenVINO FP16** from day one (CPU 90‑min budget is the hard constraint).

Expected payoff: a single SED model should clear the 0.862 public SED baseline and, blended with your Perch head, push LB toward **0.94–0.95**.

### B. Multi‑iterative pseudo‑labeling (noisy student) — second highest ceiling
2025 1st explicit ablation: Xeno pretrain 0.84 → + 2 rounds pseudo 0.91 → + TTA 0.922. This is the technique, not a detail.

1. Train teacher on `train_audio` (+ labeled `train_soundscapes`).
2. Run teacher on **all unlabeled `train_soundscapes`**. Save probabilities.
3. Confidence‑filter (e.g. keep top‑k per species per file, or prob > 0.5, discard low‑entropy empties).
4. Train student on labeled ∪ pseudo (50/50 batch mix, target = max(GT, pseudo)).
5. Repeat 1–2 more times. Stop when OOF plateaus.

Works even without a new backbone: you can pseudo‑label with your current Perch+SSM pipeline and use those labels to *supervise a trained SED backbone*. This is the cheapest path to combining A + B.

### C. Diverse ensemble — reliable linear gain
Winners used 3–14 models, but OOF‑correlation between members must be *low*, otherwise blending does nothing. Plan:
- Perch‑v2 head (current pipeline) — **keep as an ensemble member**.
- Trained SED EffNetV2‑S.
- Trained ECA‑NFNet‑L0 or ConvNeXt‑Tiny (different backbone family).
- (optional) Perch head fine‑tuned end‑to‑end (you already have `export_perch_finetuned_onnx.py`).

Sigmoid‑mean or rank‑mean blend with weights tuned on the **validation‑anchor** split (§3), not on OOF of the whole set.

### D. Cheap local wins (do these this week regardless of A–C)
- Switch the ProtoSSM loss from BCE+pos_weight to **FocalBCE** (or BCE + Focal mean, 2024 2nd's choice). `focal_gamma` is declared but not wired.
- Replace `tta_shifts=[0,1,-1,2,-2]` (5 s hops) with **sub‑window waveform shifts of ±1.25 s and ±2.5 s** before Perch inference — matches 2025 #1 (+0.012).
- Add a hard **soundscape‑wide boost**: if `max_over_windows(p_c) > θ`, raise all 12 windows' `p_c` toward that max with weight 0.2–0.3. (2024 #1/#3 universal trick.)
- Try a Gaussian temporal smoothing kernel `[0.1, 0.2, 0.4, 0.2, 0.1]` instead of the adaptive delta smoother (2024 #1/#3). Your current adaptive smoother is an over‑engineering of the same idea.
- **Ensemble by RANK, not probability**, as a cross‑check — often +0.001–0.004 for free when members are mis‑calibrated.

### E. Competition‑specific 2026 hooks
- **`train_soundscapes` has expert labels this year** (new for 2026). Use them (a) as **the primary validation anchor** (§3), and (b) as additional supervised training data mixed with Xeno‑Canto at ~1:1 batch ratio, since they match the test distribution. You already do (b); (a) is missing.
- **Pantanal** = wetland, many Amphibia/Insecta textures. Your `TEXTURE_TAXA = {Amphibia, Insecta}` temperature split ([L1742](birdclef_example/sota_oof_two_pass_ssm_advanced_pp.py#L1742)) is already right‑thinking. Train a small dedicated head on these taxa (2025 2nd did this: separate B0 head for amphibian/insect).
- **Runtime risk**: the two‑pass SSM + ResidualSSM + MLP probes + TTA×5 pipeline is CPU‑heavy. Measure end‑to‑end wall time on a full 60‑file test set **now**, not on submission day. Community discussion [#684693](https://www.kaggle.com/competitions/birdclef-2026/discussion/684693) flags Kaggle runtime library incompatibilities — dry‑run the final inference notebook repeatedly with headroom, keep a conservative fallback submission alongside the bold one.

---

## 3. Truthful evaluation — the part you flagged

Your local OOF (0.91) ≠ public LB (0.93), and sweep rankings disagree with LB rankings. Here's what the 2025 winners and the strategy playbook explicitly recommend:

### Why your current OOF is misleading
- `GroupKFold` groups only by **filename** ([L1055](birdclef_example/sota_oof_two_pass_ssm_advanced_pp.py#L1055)). Two files from the same site/hour/session end up in different folds → *optimistic* because the model learns site‑specific acoustic biases, and the validation fold is not actually out‑of‑distribution.
- You calibrate per‑class thresholds on the full training set *before* the OOF pass in `run_pipeline_oof_fullstack` — check: thresholds are refit per fold, which is correct, but the **prior tables** `build_prior_tables` are fit per fold yet *used to score the val fold*, which is fine. The risk is the `ResidualSSM` train/val split inside the pipeline (15 %, seed=42, [L933](birdclef_example/sota_oof_two_pass_ssm_advanced_pp.py#L933)) — that early‑stops on in‑fold data and then is applied to the held‑out fold. That leak is small but inflates OOF AUC by a few thousandths.
- Macro AUC over the full set treats common and rare species equally but the public LB sub‑sample is a **noisy** slice — any single rare class with 1–2 positives can move public LB by 0.01 and local OOF by 0.0001. This is the biggest source of ranking disagreement between your sweep and the LB.

### The decision hierarchy top scorers use (ranked)
1. **Mean OOF** (primary, directional).
2. **Fold variance** — a config with OOF 0.912 ± 0.006 beats 0.915 ± 0.020.
3. **Subgroup slices** — macro‑AUC recomputed on:
   - rare species only (train‑support quintile 1)
   - frequent species only (quintile 5)
   - each site separately
   - night vs day hours
4. **Runtime** (90 min CPU is a first‑class constraint).
5. **Public LB** — only as a weak directional cross‑check, *never* as the selector.

### Concrete validation changes I recommend

1. **Rebuild CV groups = site × date_block** (not filename). Parse `S{site}_{yyyymmdd}` from the filename regex you already have — FNAME_RE captures site and date. Grouping this way means any two recordings from the same site on the same day share a fold. This is what the 2025 winners' playbook calls "grouped CV by site/session/date‑block."
2. **Hold out a permanent validation anchor**: pick ~15 % of the labeled `train_soundscapes` files, stratified by site + hour, *never* train on them, compute AUC + every subgroup metric on them. Call this the **V‑anchor**. It's your best proxy for the private LB because it's drawn from the same distribution as the test set.
3. **Log per‑fold, per‑stage, per‑subgroup AUC** for every sweep config, not just the global final number. Your `oof_fullstack_stage_metrics_*.json` already captures per‑stage — extend it with `per_site`, `per_hour_bucket`, `rare_classes`, `frequent_classes`.
4. **Rank sweeps on (mean V‑anchor AUC − 1·std_across_sites)**, not global OOF. The std penalty kills configs that are overfit to a lucky site and would crater on private LB.
5. **Fast replay protocol** for iteration: 1 fold × 50 most frequent classes × fixed seed. Runs in minutes. Only promote configs that improve this to full‑stack sweeps.
6. Fix the small `ResidualSSM` internal split leak: use a disjoint val fold (a second held‑out slice of the *outer training* fold), or predict residuals with OOF from a mini 3‑fold inside the training fold.
7. **Compare LB↔V‑anchor gap over time**. When you have 5+ submissions, regress LB on V‑anchor. If correlation is high, trust V‑anchor entirely; if noisy, weight more toward *fold‑stability* and subgroup metrics.

One extra dataset trick recommended across 2021–22 writeups: download and hand‑label (or use public labels) from **past BirdCLEF soundscape test sets** that overlap in taxa. For 2026 Pantanal that's limited — but any Xeno‑Canto soundscape recording with site + date metadata is useful V‑anchor‑style validation.

---

## 4. Action plan (time‑boxed)

Final‑submission deadline is **2026‑06‑03**; entry deadline **2026‑05‑27**. ~5–6 weeks from today (2026‑04‑23).

### Week 1 (now → 2026‑04‑30): validation & cheap wins
> **2026-04-25 update**: V-anchor was abandoned after empirical LB calibration failed (see [STRATEGY_V2.md](STRATEGY_V2.md) §11). All A/B testing below now runs against **stitched 5-fold OOF macro AUC** built by `_02_build_splits.py` — file-level `StratifiedKFold` over all 59 labeled soundscapes, no permanent hold-out.
- [x] Change GroupKFold groups from `filename` to `site|date_block` (later replaced again by file-level StratifiedKFold — see V2).
- [x] V-anchor was built as a permanent stratified hold-out, then removed. Now folds 0..n−1 are the only validation infrastructure.
- [x] Extend metrics JSON with per‑site, per‑hour, rare/frequent subgroup AUC.
- [x] Wire the already‑declared `focal_gamma` into the ProtoSSM loss — replace BCE+pos_weight with FocalBCE. A/B on stitched OOF.
- [x] Add sub‑window TTA (±1.25 s, ±2.5 s waveform shifts *before* Perch), compare vs current window‑rolling TTA.
- [x] Add Gaussian kernel `[0.1,0.2,0.4,0.2,0.1]` smoothing and hard soundscape‑wide probability boost as post‑processing variants. A/B on stitched OOF.
- [ ] Measure end‑to‑end CPU runtime on a full 60‑file synthetic test. Record headroom vs 90 min.
- [x] Make one baseline submission to establish LB↔local calibration. (Result: legacy SSM = 0.93 LB; SED 5-fold = 0.747; gap is real.)

### Week 2–3 (2026‑05‑01 → 2026‑05‑14): trained SED backbone
- [ ] Stand up a clean SED trainer on `train_audio` + labeled `train_soundscapes`. Start from [train_ddp_sota_perch.py](birdclef_example/train_ddp_sota_perch.py) as scaffold, target `tf_efficientnetv2_s_in21k` SED head.
- [ ] FocalBCE, mixup (α=0.5, label=max), SpecAugment, background‑mix from soundscape no‑call windows, random gain, pink noise. Mask secondary‑label loss.
- [ ] 5‑fold CV on the new grouping. Export each fold to ONNX FP16.
- [ ] Ensemble SED + Perch head. Weight search on V‑anchor. Expect +0.01–0.03 LB.
- [ ] Second backbone (ECA‑NFNet‑L0) if V‑anchor gain is ≥0.005 from SED alone (diversity is wasted on identical behavior).

### Week 4 (2026‑05‑15 → 2026‑05‑21): pseudo‑labeling round 1
- [ ] Run best ensemble teacher on **all unlabeled `train_soundscapes`**. Save logits.
- [ ] Confidence‑filter: keep top‑k per file per species, or prob > 0.5; drop files where all species stay under 0.2 (probable empties).
- [ ] Retrain SED backbone(s) with 50/50 labeled/pseudo batches; target = max(GT, pseudo).
- [ ] Submit once, check LB↔V‑anchor gap before a second round.

### Week 5 (2026‑05‑22 → 2026‑05‑27): pseudo round 2, ensemble lock, runtime hardening
- [ ] Second pseudo round if V‑anchor still improves (2025 #1: +0.04 total from 2 rounds).
- [ ] Final ensemble freeze: choose 3–6 members by V‑anchor + low member correlation. Lock weights.
- [ ] Dry‑run the inference notebook **end‑to‑end** on Kaggle at least 3× with different seeds and network conditions. Pad runtime budget to ≤75 min to leave headroom.
- [ ] Prepare two submissions: (a) the *bold* best‑V‑anchor ensemble; (b) a *conservative* safe ensemble (current Perch+SSM stack + SED fold‑0 only). Use both of Kaggle's 2 daily submission slots to observe LB behavior, then choose final‑selection pair.

### Week 6 (2026‑05‑28 → 2026‑06‑03): cushion
- Buffer for library‑compat failures (see [discussion #684693](https://www.kaggle.com/competitions/birdclef-2026/discussion/684693)). Avoid any non‑trivial code change after 2026‑06‑01.
- Select two final submissions: the one with best V‑anchor, and the one with best fold stability even if slightly lower mean.

---

## 5. Non‑goals / things NOT to chase

- **Bigger SSM / transformer replacements for ProtoSSM**: your current SSM head is already pulling 0.93 on LB — extra capacity in the head saturates once the *inputs* (Perch embeddings) are fixed. Effort goes to new backbones, not deeper heads.
- **More hyper‑parameter sweeps on the existing stack**: your sweep4 already shows configs clustered within ±0.015 OOF. Diminishing returns. Budget here is capped at 1 day.
- **Public‑LB probing**: each Kaggle submission is information‑costly on public‑private correlation. Don't re‑tune on LB bumps.
- **Metadata/geo filters** (used by 2021 top‑5, abandoned since 2023): no evidence they help on Pantanal yet, and they routinely zero out rare‑species recall.
- **Exotic losses** (asymmetric, DBLoss, etc.): every 2024/2025 top‑5 used FocalBCE or BCE. Don't spend time here.

---

## 6. Key references

- Competition: https://www.kaggle.com/competitions/birdclef-2026
- ImageCLEF 2026 mirror (specs): https://www.imageclef.org/BirdCLEF2026
- 2025 1st (Babych, "Multi‑iterative Noisy Student"): https://www.kaggle.com/competitions/birdclef-2025/writeups/nikita-babych-1st-place-solution-multi-iterative-n
- 2025 2nd (Sydorskyy): https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place and CEUR paper https://ceur-ws.org/Vol-4038/paper_256.pdf
- 2024 1st (jfpuget+Henkel+Theo Viel): https://github.com/jfpuget/birdclef-2024 — walkthrough https://www.youtube.com/watch?v=prLlEZ38eaw
- 2024 3rd (Theo Viel): https://github.com/TheoViel/kaggle_birdclef2024
- 2023 1st (Sydorskyy, "Correct Data is All You Need"): https://github.com/VSydorskyy/BirdCLEF_2023_1st_place
- 2022 1st (KDL, "It's not all BirdNet"): https://github.com/Selimonder/birdclef-2022
- Perch v2 paper (why it's the best frozen embedding): https://arxiv.org/html/2508.04665v1
- Best public 2026 notebook so far: https://www.kaggle.com/code/yashanathaniel/birdclef-2026-perch-v2-0-908
