"""Patch LB_0931_seed.ipynb cell 26 — replace AST with focal CNN inference
(PyTorch eager, no ONNX). The CNN is small (~50 MB nfnet_l0) so PyTorch
eager on CPU is fine; ~1-2s per file vs AST's 14s.
"""
import json
from pathlib import Path

NB = Path("LB_0931_seed.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

NEW_CELL = '''# ── Cell 9c: Focal CNN inference (PyTorch eager) ──────────────────────
# Loads the fine-tuned timm CNN (nfnet_l0, ~50 MB) and runs PyTorch eager
# inference on every test soundscape. CPU is fine — small backbone, no
# transformer overhead. ~1-2s per file. ThreadPoolExecutor prefetches the
# next files' audio while the current file is processed (same shape as
# the Perch/AST cells).

import torch.nn as _nn
import torchaudio as _ta
import concurrent.futures as _cf
from collections import deque as _deque
try:
    import timm as _timm
except ImportError:
    raise SystemExit("timm not installed — `pip install timm` (should be in Kaggle base image).")

# === EDIT THIS PATH for your Kaggle dataset mount ===
CNN_CKPT_PATH = Path("/kaggle/input/birdclef-cnn-best/best_model.pt")
if not CNN_CKPT_PATH.exists():
    CNN_CKPT_PATH = Path("/kaggle/working/cnn_best_model.pt")  # fallback
if not CNN_CKPT_PATH.exists():
    raise SystemExit(
        f"CNN checkpoint missing: {CNN_CKPT_PATH}. Upload "
        f"birdclef_example/outputs/focal/<best>/best_model.pt as a Kaggle "
        f"dataset (e.g. birdclef-cnn-best) and edit CNN_CKPT_PATH above."
    )


# Inline TimmSpectrogramClassifier (inference-only — no augmentations).
# Same forward path as birdclef_example/train_ddp_sota.py:
#   waveform (B, 1, T) → mel-spec → clamp+norm → resize → repeat to 3ch → timm
class _SpecTransform(_nn.Module):
    def __init__(self, sample_rate, n_mels, n_fft, hop_length, f_min=20, f_max=None):
        super().__init__()
        self.melspec = _ta.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max or sample_rate // 2, power=2.0,
        )
        self.to_db = _ta.transforms.AmplitudeToDB(stype="power")
    def forward(self, x):
        spec = self.to_db(self.melspec(x))
        mean = spec.mean(dim=[-2, -1], keepdim=True)
        std = spec.std(dim=[-2, -1], keepdim=True).clamp(min=1e-6)
        return (spec - mean) / std


class _TimmCNN(_nn.Module):
    """Inference-only TimmSpectrogramClassifier (drops wave/spec aug)."""
    def __init__(self, n_classes, backbone_name, image_size, sample_rate,
                 n_mels, n_fft, hop_length, dropout=0.35):
        super().__init__()
        self.image_size = int(image_size)
        self.spectrogram = _SpecTransform(sample_rate, n_mels, n_fft, hop_length)
        self.backbone = _timm.create_model(
            backbone_name, pretrained=False, in_chans=3,
            num_classes=int(n_classes), drop_rate=float(dropout),
        )
    def forward(self, waveform):
        x = self.spectrogram(waveform)                  # (B, 1, n_mels, T)
        x = x.clamp(min=-4.0, max=4.0)
        x = (x + 4.0) / 8.0
        x = torch.nn.functional.interpolate(
            x, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )
        x = x.repeat(1, 3, 1, 1)                        # 3-ch input for timm
        return self.backbone(x)


_cnn_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[cnn] loading {CNN_CKPT_PATH} on {_cnn_dev}...")
_cnn_ckpt = torch.load(CNN_CKPT_PATH, map_location="cpu", weights_only=False)
_cnn_mc = _cnn_ckpt["model_config"]
print(f"[cnn]   backbone={_cnn_mc['backbone_name']}  "
      f"image_size={_cnn_mc['image_size']}  n_mels={_cnn_mc['n_mels']}")
print(f"[cnn]   epoch={_cnn_ckpt.get('epoch')}  "
      f"best_val_auc_focal_seen={_cnn_ckpt.get('best_val_auc_focal_seen')}")

cnn_model = _TimmCNN(
    n_classes=int(_cnn_mc["n_classes"]),
    backbone_name=str(_cnn_mc["backbone_name"]),
    image_size=int(_cnn_mc["image_size"]),
    sample_rate=int(_cnn_mc.get("sample_rate", 32000)),
    n_mels=int(_cnn_mc["n_mels"]),
    n_fft=int(_cnn_mc["n_fft"]),
    hop_length=int(_cnn_mc["hop_length"]),
    dropout=float(_cnn_mc.get("dropout", 0.35)),
).to(_cnn_dev)
cnn_model.load_state_dict(_cnn_ckpt["model_state"], strict=False)
cnn_model.eval()


def _cnn_read_60s(fpath):
    """Read 60s soundscape, mono-mix, pad/truncate. Releases GIL during read."""
    import soundfile as _sf
    y, _sr = _sf.read(str(fpath), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.shape[0] < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - y.shape[0]))
    else:
        y = y[:FILE_SAMPLES]
    return y.astype(np.float32, copy=False)


# Inference: 1 file per forward (12 windows). 4 I/O threads keep the
# pipeline full so disk reads overlap with the model forward.
n_test_files = len(test_paths)
ast_test_probs = np.zeros((n_test_files * N_WINDOWS, N_CLASSES), dtype=np.float32)
_t0_cnn = time.time()
_PREFETCH_DEPTH = 4
print(f"[cnn] 1 file / forward, prefetch depth={_PREFETCH_DEPTH}, {n_test_files} files")

with _cf.ThreadPoolExecutor(max_workers=4) as _io:
    _q = _deque()
    _next_idx = min(_PREFETCH_DEPTH, n_test_files)
    for _i in range(_next_idx):
        _q.append(_io.submit(_cnn_read_60s, test_paths[_i]))

    for _fi in range(n_test_files):
        y = _q.popleft().result()
        if _next_idx < n_test_files:
            _q.append(_io.submit(_cnn_read_60s, test_paths[_next_idx]))
            _next_idx += 1

        # CNN expects (B, 1, T) — 12 windows of 5s @ 32kHz.
        wins = (
            torch.from_numpy(y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32))
            .unsqueeze(1).to(_cnn_dev)
        )
        with torch.no_grad():
            with torch.autocast(device_type=_cnn_dev.type, dtype=torch.bfloat16,
                                enabled=_cnn_dev.type == "cuda"):
                logits = cnn_model(wins)                # (12, N_CLASSES)
        ast_test_probs[_fi * N_WINDOWS:(_fi + 1) * N_WINDOWS] = (
            torch.sigmoid(logits.float()).cpu().numpy().astype(np.float32)
        )

        if _fi % 50 == 0 or _fi == n_test_files - 1:
            elapsed = time.time() - _t0_cnn
            done = _fi + 1
            eta = elapsed / done * (n_test_files - done)
            print(f"[cnn]   {done}/{n_test_files}  elapsed={elapsed:.1f}s  eta={eta:.1f}s")

# Note: the variable is named `ast_test_probs` to keep cell 27's blend
# code unchanged (it references ast_test_probs by name). Substituting the
# CNN here is purely a member swap — the blend math is identical.
print(f"[cnn] inference done  shape={ast_test_probs.shape}  "
      f"mean={ast_test_probs.mean():.4f}  total={time.time()-_t0_cnn:.1f}s")
'''

nb["cells"][26]["source"] = NEW_CELL.splitlines(keepends=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("✓ patched cell 26 with CNN PyTorch inference")
print(f"  cell 26 size: {len(NEW_CELL)}b")
