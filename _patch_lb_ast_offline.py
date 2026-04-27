"""Patch LB_0931_seed.ipynb cell 26 to load AST offline (no Hub fetch).

Replaces the network-dependent _ASTSpec/_from_pretrained path with a
pure-local from_pretrained(local_dir, local_files_only=True) call. This
requires the user to first run _freeze_ast_for_kaggle.py to produce the
offline directory + upload it as a Kaggle dataset.
"""
import json
from pathlib import Path

NB = Path("LB_0931_seed.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

NEW_AST_CELL = '''# ── Cell 9c: AST (audio transformer) inference (OFFLINE) ─────────────
# Runs the AudioSet-pretrained AST model fine-tuned per ast_lr3e-05_e15
# (best of the 9-config sweep, val_auc_focal_seen=0.8562) on every test
# window. Produces ast_test_probs (n_test_rows × N_CLASSES) which is
# per-taxon blended with the Perch+SSM probs in cell 10.
#
# The model is loaded ENTIRELY OFFLINE from a Kaggle dataset directory
# that contains the pre-frozen HuggingFace-format model:
#     config.json + model.safetensors + wrapper_config.json
# Build it once on Hopper:
#     python -m birdclef.scripts._freeze_ast_for_kaggle \\
#         --ast-ckpt birdclef_example/outputs/ast/ast_lr3e-05_e15/best_model.pt \\
#         --out-dir  birdclef_example/outputs/ast/ast_lr3e-05_e15_offline
# Upload the resulting directory as a Kaggle dataset named birdclef-ast-best.

import torch.nn as _nn
import torchaudio as _ta
try:
    from transformers import ASTForAudioClassification as _ASTModel
except ImportError:
    raise SystemExit("transformers not installed — add `pip install transformers` to cell 0.")

# === EDIT THIS PATH for your Kaggle dataset mount ===
AST_OFFLINE_DIR = Path("/kaggle/input/birdclef-ast-best/ast_lr3e-05_e15_offline")
if not AST_OFFLINE_DIR.exists():
    # fallback: maybe the dataset was uploaded with the parent dir flattened
    AST_OFFLINE_DIR = Path("/kaggle/input/birdclef-ast-best")
if not (AST_OFFLINE_DIR / "config.json").exists():
    raise SystemExit(
        f"AST offline dir missing or incomplete: {AST_OFFLINE_DIR}\\n"
        f"Expected config.json + model.safetensors inside.\\n"
        f"Run _freeze_ast_for_kaggle on Hopper, then upload the dir as "
        f"a Kaggle dataset and edit AST_OFFLINE_DIR above."
    )

_FBANK_MEAN = -4.2677
_FBANK_STD  = 4.5689

# Wrapper-level params (max_length, sample rates) saved by _freeze_ast_for_kaggle.
_wrapper_cfg_path = AST_OFFLINE_DIR / "wrapper_config.json"
if _wrapper_cfg_path.exists():
    with open(_wrapper_cfg_path, encoding="utf-8") as _f:
        _wrapper_cfg = json.load(_f)
else:
    _wrapper_cfg = {"input_sample_rate": 32000, "target_sample_rate": 16000,
                    "max_length": 512, "num_mel_bins": 128, "n_classes": N_CLASSES}
print(f"[ast] wrapper_cfg = {_wrapper_cfg}")


class _ASTSpec(_nn.Module):
    """Wraps a pre-loaded ASTForAudioClassification with the kaldi-fbank
    preprocessing the trainer used. No HuggingFace Hub calls.
    """
    def __init__(self, model, target_sr, max_length, num_mel_bins, input_sr):
        super().__init__()
        self.model = model
        self.target_sr = int(target_sr)
        self.max_length = int(max_length)
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
        wav = self.resample(wav_BxT)
        with torch.autocast(device_type=wav.device.type, enabled=False):
            wav_f32 = wav.float()
            fb = torch.stack([self._fbank_one(wav_f32[i:i+1]) for i in range(wav_f32.size(0))], dim=0)
        return self.model(input_values=fb).logits


_ast_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ast] loading {AST_OFFLINE_DIR} on {_ast_dev}  (local_files_only=True — no Hub fetch)")
_ast_inner = _ASTModel.from_pretrained(str(AST_OFFLINE_DIR), local_files_only=True)
ast_model = _ASTSpec(
    model=_ast_inner.to(_ast_dev),
    target_sr=int(_wrapper_cfg["target_sample_rate"]),
    max_length=int(_wrapper_cfg["max_length"]),
    num_mel_bins=int(_wrapper_cfg["num_mel_bins"]),
    input_sr=int(_wrapper_cfg["input_sample_rate"]),
).to(_ast_dev)
ast_model.eval()
print(f"[ast] loaded — n_classes={_wrapper_cfg.get('n_classes', '?')}, "
      f"trained_epoch={_wrapper_cfg.get('trained_epoch', '?')}, "
      f"best_val_auc_focal_seen={_wrapper_cfg.get('best_val_auc_focal_seen', '?')}")

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

nb["cells"][26]["source"] = NEW_AST_CELL.splitlines(keepends=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("✓ patched cell 26 with offline AST loader")
print(f"  cell 26 size: {len(NEW_AST_CELL)}b")
