"""One-shot patch: add AST inference cell + per-taxon blend to LB_0931_seed.ipynb,
disable the flat-head override (it hurt LB by 0.011 in A/B)."""
from __future__ import annotations
import json
from pathlib import Path

NB = Path("LB_0931_seed.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

# ----- Patch 1: disable the flat-head override in cell 26 -----------------
old_override = (
    "# ── Step J0: Override no-signal classes with the FLAT-HEAD predictions ─\n"
    "# The 28 no-signal classes (sonotypes + a few Amphibia Perch can't see)\n"
    "# get 0.5 from the rest of the stack because there is no Perch signal for\n"
    "# them. The flat head gives real, calibrated probabilities. Inject AFTER\n"
    "# per-class thresholds because PER_CLASS_THRESHOLDS for these classes was\n"
    "# computed on near-random first-pass scores and is not informative.\n"
    "probs[:, NO_SIGNAL_POS] = head_test_probs\n"
    "print(f\"Injected flat-head predictions into {len(NO_SIGNAL_POS)} no-signal classes\")"
)
new_override_disabled = (
    "# ── Step J0: (DISABLED) flat-head override hurt LB by -0.011 in A/B test\n"
    "# (LB_0931 with override = 0.913, without = 0.924). Likely OOD distribution\n"
    "# shift on unseen sites. Keep flat-head training cell above (useful for\n"
    "# pseudo-labeling later) but skip the override here.\n"
    "# probs[:, NO_SIGNAL_POS] = head_test_probs"
)
src26 = "".join(nb["cells"][26]["source"])
assert old_override in src26, "could not locate flat-head override block"
src26 = src26.replace(old_override, new_override_disabled)

# ----- Patch 2: insert per-taxon AST blend right before per-class thresholds -----
old_postproc = (
    "# ── Step I: Post-processing pipeline ──────────────────────────────────\n"
    "probs = file_confidence_scale(probs, n_windows=N_WINDOWS,\n"
    "                               top_k=2,       power=0.4)\n"
    "probs = rank_aware_scaling(   probs, n_windows=N_WINDOWS,\n"
    "                               power=0.4)\n"
    "probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS,\n"
    "                               base_alpha=0.20)\n"
    "probs = np.clip(probs, 0.0, 1.0)\n"
    "\n"
    "probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)"
)
new_postproc = (
    "# ── Step I: Post-processing pipeline ──────────────────────────────────\n"
    "probs = file_confidence_scale(probs, n_windows=N_WINDOWS,\n"
    "                               top_k=2,       power=0.4)\n"
    "probs = rank_aware_scaling(   probs, n_windows=N_WINDOWS,\n"
    "                               power=0.4)\n"
    "probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS,\n"
    "                               base_alpha=0.20)\n"
    "probs = np.clip(probs, 0.0, 1.0)\n"
    "\n"
    "# ── Step I-blend: Per-taxon blend with AST predictions ────────────────\n"
    "# AST was AudioSet-pretrained, beats SSM stack on Insecta/Mammalia/\n"
    "# Reptilia, loses on Aves/Amphibia (where Perch genus-proxy dominates).\n"
    "# Per-taxon weights derived from val_auc_<taxon> on each model:\n"
    "#   taxon       SSM   AST   reasoning\n"
    "#   Aves        0.60  0.40  Perch dominant on birds (val 0.90 vs 0.89)\n"
    "#   Amphibia    0.55  0.45  Perch genus-proxy strong (val 0.85 vs 0.77)\n"
    "#   Insecta     0.30  0.70  AST much stronger (val 0.70 vs 0.50)\n"
    "#   Mammalia    0.30  0.70  AST stronger (val 0.90 vs 0.83)\n"
    "#   Reptilia    0.30  0.70  AST learned Caiman; Perch random\n"
    "#   (other)     0.50  0.50  safe default\n"
    "W_SSM_PER_CLASS = np.full(N_CLASSES, 0.5, dtype=np.float32)\n"
    "W_AST_PER_CLASS = np.full(N_CLASSES, 0.5, dtype=np.float32)\n"
    "_TAXON_WEIGHTS = {\n"
    "    \"Aves\":     (0.60, 0.40),\n"
    "    \"Amphibia\": (0.55, 0.45),\n"
    "    \"Insecta\":  (0.30, 0.70),\n"
    "    \"Mammalia\": (0.30, 0.70),\n"
    "    \"Reptilia\": (0.30, 0.70),\n"
    "}\n"
    "for ci, lbl in enumerate(PRIMARY_LABELS):\n"
    "    cls_name = CLASS_NAME_MAP.get(lbl, \"Unknown\")\n"
    "    if cls_name in _TAXON_WEIGHTS:\n"
    "        W_SSM_PER_CLASS[ci], W_AST_PER_CLASS[ci] = _TAXON_WEIGHTS[cls_name]\n"
    "probs = (W_SSM_PER_CLASS[None, :] * probs\n"
    "         + W_AST_PER_CLASS[None, :] * ast_test_probs)\n"
    "probs = np.clip(probs, 0.0, 1.0)\n"
    "print(f\"AST blend applied — per-taxon mean SSM={W_SSM_PER_CLASS.mean():.3f} \"\n"
    "      f\"AST={W_AST_PER_CLASS.mean():.3f}\")\n"
    "\n"
    "probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)"
)
assert old_postproc in src26, "could not locate post-processing block"
src26 = src26.replace(old_postproc, new_postproc)
nb["cells"][26]["source"] = src26.splitlines(keepends=True)
print("✓ patched cell 26 — flat-head override disabled, per-taxon AST blend added")

# ----- Patch 3: insert new AST inference cell at index 26 -----------------
AST_CELL = '''# ── Cell 9c: AST (audio transformer) inference ────────────────────────
# Runs the AudioSet-pretrained AST model fine-tuned per ast_lr3e-05_e15
# (best of the 9-config sweep, val_auc_focal_seen=0.8562) on every test
# window. Produces ast_test_probs (n_test_rows × N_CLASSES) which is
# per-taxon blended with the Perch+SSM probs in cell 10.
#
# AST checkpoint is ~328 MB; needs to be uploaded as a Kaggle dataset.

import torch.nn as _nn
import torchaudio as _ta
try:
    from transformers import ASTConfig as _ASTConfig, ASTForAudioClassification as _ASTModel
except ImportError:
    raise SystemExit("transformers not installed — add `pip install transformers` to cell 0.")

# === EDIT THIS PATH for your Kaggle dataset mount ===
AST_CKPT_PATH = Path("/kaggle/input/birdclef-ast-best/best_model.pt")
if not AST_CKPT_PATH.exists():
    AST_CKPT_PATH = Path("/kaggle/working/ast_best_model.pt")  # fallback

_FBANK_MEAN = -4.2677
_FBANK_STD  = 4.5689


def _ast_interpolate_pe(old_pe, n_freq, n_time_old, n_time_new, n_special=2):
    embed_dim = old_pe.shape[-1]
    special = old_pe[:, :n_special, :]
    patches = old_pe[:, n_special:, :]
    grid = patches.transpose(1, 2).reshape(1, embed_dim, n_freq, n_time_old).float()
    new_grid = torch.nn.functional.interpolate(
        grid, size=(n_freq, n_time_new), mode="bicubic", align_corners=False,
    )
    new_patches = new_grid.reshape(1, embed_dim, n_freq * n_time_new).transpose(1, 2)
    return torch.cat([special, new_patches.to(old_pe.dtype)], dim=1)


class _ASTSpec(_nn.Module):
    def __init__(self, n_classes, hf_name, max_length=512, num_mel_bins=128,
                 input_sr=32000, target_sr=16000):
        super().__init__()
        orig_cfg = _ASTConfig.from_pretrained(hf_name)
        tmp = _ASTModel.from_pretrained(hf_name, num_labels=int(n_classes),
                                        ignore_mismatched_sizes=True)
        state = tmp.state_dict(); del tmp
        n_freq      = (num_mel_bins - orig_cfg.patch_size) // orig_cfg.frequency_stride + 1
        n_time_old  = (orig_cfg.max_length - orig_cfg.patch_size) // orig_cfg.time_stride + 1
        n_time_new  = (max_length - orig_cfg.patch_size) // orig_cfg.time_stride + 1
        pe_key = "audio_spectrogram_transformer.embeddings.position_embeddings"
        if n_time_new != n_time_old and pe_key in state:
            state[pe_key] = _ast_interpolate_pe(state[pe_key], n_freq, n_time_old, n_time_new, 2)
        cfg = _ASTConfig.from_pretrained(hf_name)
        cfg.num_labels = int(n_classes); cfg.max_length = int(max_length); cfg.num_mel_bins = int(num_mel_bins)
        self.model = _ASTModel(cfg)
        self.model.load_state_dict(state, strict=False)
        self.target_sr = int(target_sr); self.max_length = int(max_length)
        self.num_mel_bins = int(num_mel_bins)
        self.resample = (_ta.transforms.Resample(int(input_sr), int(target_sr))
                         if input_sr != target_sr else _nn.Identity())

    def _fbank_one(self, wav):
        fb = _ta.compliance.kaldi.fbank(
            wav, htk_compat=True, sample_frequency=self.target_sr,
            use_energy=False, window_type="hanning",
            num_mel_bins=self.num_mel_bins, dither=0.0,
            frame_shift=10, frame_length=25,
        )
        fb = (fb - _FBANK_MEAN) / (_FBANK_STD * 2.0)
        n = fb.size(0)
        if n < self.max_length:
            fb = torch.nn.functional.pad(fb, (0, 0, 0, self.max_length - n))
        elif n > self.max_length:
            fb = fb[: self.max_length, :]
        return fb

    def forward(self, wav_BxT):
        # wav_BxT: (B, T) at 32 kHz already (BirdCLEF window). Resample to 16 kHz,
        # compute kaldi fbank in fp32 (kaldi doesn't support bf16), feed to AST.
        wav = self.resample(wav_BxT)
        with torch.autocast(device_type=wav.device.type, enabled=False):
            wav_f32 = wav.float()
            fb = torch.stack([self._fbank_one(wav_f32[i:i+1]) for i in range(wav_f32.size(0))], dim=0)
        return self.model(input_values=fb).logits


_ast_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ast] loading {AST_CKPT_PATH} on {_ast_dev}...")
_ast_ckpt = torch.load(AST_CKPT_PATH, map_location="cpu", weights_only=False)
_ast_mc = _ast_ckpt["model_config"]
ast_model = _ASTSpec(
    n_classes=int(_ast_mc["n_classes"]),
    hf_name=str(_ast_mc["hf_model_name"]),
    max_length=int(_ast_mc.get("max_length", 512)),
    num_mel_bins=int(_ast_mc.get("num_mel_bins", 128)),
    input_sr=int(_ast_mc.get("input_sample_rate", 32000)),
    target_sr=int(_ast_mc.get("target_sample_rate", 16000)),
).to(_ast_dev)
ast_model.load_state_dict(_ast_ckpt["model_state"], strict=False)
ast_model.eval()
print(f"[ast] checkpoint epoch={_ast_ckpt.get('epoch')}  "
      f"best_val_auc_focal_seen={_ast_ckpt.get('best_val_auc_focal_seen')}")

# Inference: each test soundscape (60s @ 32kHz) → 12 windows of 5s → AST forward
import soundfile as _sf
_t0_ast = time.time()
n_test_files = len(test_paths)
ast_test_probs = np.zeros((n_test_files * N_WINDOWS, N_CLASSES), dtype=np.float32)
for fi, fpath in enumerate(test_paths):
    y, _sr = _sf.read(str(fpath), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.shape[0] < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - y.shape[0]))
    else:
        y = y[:FILE_SAMPLES]
    wins = torch.from_numpy(y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)).to(_ast_dev)
    with torch.no_grad():
        with torch.autocast(device_type=_ast_dev.type, dtype=torch.bfloat16,
                            enabled=_ast_dev.type == "cuda"):
            logits = ast_model(wins)
    ast_test_probs[fi * N_WINDOWS:(fi + 1) * N_WINDOWS] = (
        torch.sigmoid(logits.float()).cpu().numpy().astype(np.float32)
    )
    if fi % 50 == 0:
        elapsed = time.time() - _t0_ast
        print(f"[ast]   {fi+1}/{n_test_files}  elapsed={elapsed:.1f}s  "
              f"eta={elapsed/(fi+1)*(n_test_files-fi-1):.1f}s")

print(f"[ast] inference done  shape={ast_test_probs.shape}  "
      f"mean={ast_test_probs.mean():.4f}  total={time.time()-_t0_ast:.1f}s")
'''

new_cell = {
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": AST_CELL.splitlines(keepends=True),
}
nb["cells"].insert(26, new_cell)
print(f"✓ inserted AST inference cell at index 26 — total cells now {len(nb['cells'])}")

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"✓ saved {NB}")
