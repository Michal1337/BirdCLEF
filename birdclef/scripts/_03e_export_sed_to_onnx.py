"""Export trained SED fold checkpoints to ONNX for Kaggle inference.

Loads `best.pt` from each `MODEL_ROOT/sed/<config>/fold{k}/` and exports the
SED via `SEDExportWrapper`, which swaps the torchaudio mel front-end for
`ConvMelSpectrogram` (Conv1d + matmul + log) so the graph contains only
ONNX-exportable ops. EMA weights are applied if present in the checkpoint.

Output is a 2-tuple `(clip_logits, framewise_logits)` matching Tucker's
distilled-SED bundle signature, so the same Kaggle inference cell can feed
both bundles with no code changes (just iterate sessions across both dirs).

Usage:
    PYTHONIOENCODING=utf-8 python -m birdclef.scripts._03e_export_sed_to_onnx \\
        --config sed_b0_dual --out-dir models/sed_inhouse
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from birdclef.config.paths import MODEL_ROOT
from birdclef.models.sed import SED, SEDConfig, SEDExportWrapper


def _load_sed_from_ckpt(ckpt_path: Path, device: torch.device) -> tuple[SED, dict]:
    """Build SED from saved cfg, load state_dict, apply EMA shadow if present."""
    state = torch.load(ckpt_path, map_location="cpu")
    cfg = state["cfg"]
    sed_cfg = SEDConfig(
        backbone=cfg["backbone"], n_classes=cfg["n_classes"], dropout=cfg["dropout"],
        sample_rate=cfg["sample_rate"], n_mels=cfg["n_mels"], n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"], f_min=cfg["f_min"], f_max=cfg["f_max"],
    )
    m = SED(sed_cfg).to(device)
    m.load_state_dict(state["state_dict"], strict=False)
    ema_shadow = state.get("ema") or {}
    if ema_shadow:
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n in ema_shadow:
                    p.data.copy_(ema_shadow[n].to(p.device))
    m.eval()
    return m, cfg


def _export_one(sed: SED, out_path: Path, opset: int = 17,
                use_dynamo: bool = False) -> None:
    """Wrap SED in SEDExportWrapper (ONNX-safe mel) and torch.onnx.export.

    Defaults to the LEGACY exporter (`dynamo=False`) — the dynamo exporter
    in torch 2.10+ has known issues hardcoding internal Reshape shapes from
    the batch=1 trace, so when ORT runs with batch=12 it fails at a
    `gemm_input_reshape` node. The legacy exporter respects `dynamic_axes`
    on intermediate ops more reliably.

    Default opset 17 — Tucker's bundle uses 17 and ORT supports it
    everywhere. Bump to 20 if your local torch warns about deprecated ops
    on 17.
    """
    from birdclef.config.paths import WINDOW_SAMPLES

    wrapper = SEDExportWrapper(sed).eval().cpu()
    # Dummy input: a representative-sized batch (B=12 = N_WINDOWS) so the
    # trace doesn't pin batch=1 in any internal op. Combined with
    # dynamic_axes batch=0, the resulting graph accepts any batch size.
    dummy = torch.zeros(12, WINDOW_SAMPLES, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export_kwargs = dict(
        opset_version=int(opset),
        input_names=["waveform"],
        output_names=["clip_logits", "framewise_logits"],
        dynamic_axes={
            "waveform": {0: "batch"},
            "clip_logits": {0: "batch"},
            "framewise_logits": {0: "batch"},
        },
    )
    # `dynamo` kwarg only exists on torch ≥ 2.5; older versions silently use
    # the legacy path. Pass it conditionally.
    try:
        torch.onnx.export(wrapper, dummy, str(out_path),
                          dynamo=bool(use_dynamo), **export_kwargs)
    except TypeError:
        # Older torch — no dynamo kwarg, default is legacy path.
        torch.onnx.export(wrapper, dummy, str(out_path), **export_kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="SED config name; checkpoints expected at "
                         f"{MODEL_ROOT}/sed/<config>/fold{{0..K-1}}/best.pt")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out-dir", required=True,
                    help="Where to write sed_fold{0..K-1}.onnx files. "
                         "e.g. `models/sed_inhouse` (a sibling to models/sed_kaggle).")
    ap.add_argument("--ckpt-name", default="best.pt",
                    help="Checkpoint filename inside each fold dir. "
                         "Default `best.pt`.")
    ap.add_argument("--opset", type=int, default=17,
                    help="ONNX opset version. Default 17 — matches Tucker's "
                         "bundle and is widely supported by ORT.")
    ap.add_argument("--use-dynamo", action="store_true",
                    help="Use the new dynamo exporter (torch ≥ 2.5). "
                         "Default OFF — legacy exporter handles dynamic batch "
                         "more reliably for SED graphs.")
    args = ap.parse_args()

    base_dir = Path(MODEL_ROOT) / "sed" / args.config
    if not base_dir.exists():
        raise SystemExit(f"No SED config dir at {base_dir}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[export] config={args.config}  base_dir={base_dir}")
    print(f"[export] out_dir={out_dir}  opset={args.opset}")

    device = torch.device("cpu")
    n_exported = 0
    for fold in range(int(args.n_folds)):
        ckpt = base_dir / f"fold{fold}" / args.ckpt_name
        if not ckpt.exists():
            print(f"[export] WARN: missing {ckpt}; skipping fold {fold}")
            continue
        out_path = out_dir / f"sed_fold{fold}.onnx"
        print(f"[export] fold {fold}: {ckpt} → {out_path}")
        sed, cfg = _load_sed_from_ckpt(ckpt, device)
        _export_one(sed, out_path, opset=int(args.opset),
                    use_dynamo=bool(args.use_dynamo))
        # Quick sanity check on the exported model — try BOTH a small batch
        # (1 window) AND the full N_WINDOWS=12 batch the inference loop will
        # use. Catches dynamic-axis bugs at export time, not later in eval.
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        x_one = np.zeros((1, cfg["sample_rate"] * 5), dtype=np.float32)
        outs_one = sess.run(None, {in_name: x_one})
        x_full = np.zeros((12, cfg["sample_rate"] * 5), dtype=np.float32)
        outs_full = sess.run(None, {in_name: x_full})
        size_mb = out_path.stat().st_size / 1024**2
        print(f"[export]   exported OK — size={size_mb:.1f} MB  "
              f"B=1 clip={outs_one[0].shape} fw={outs_one[1].shape}  "
              f"B=12 clip={outs_full[0].shape} fw={outs_full[1].shape}")
        n_exported += 1

    if n_exported == 0:
        raise SystemExit(f"No checkpoints exported. Looked under {base_dir}")
    print()
    print(f"[export] DONE — wrote {n_exported} ONNX files to {out_dir}")
    print(f"[export] Next: upload {out_dir} as a Kaggle dataset, attach to "
          f"the LB notebook, and update the inference cell to also run this "
          f"bundle (or replace Tucker's).")


if __name__ == "__main__":
    main()
