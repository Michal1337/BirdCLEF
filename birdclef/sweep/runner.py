"""Generic sweep runner used by SSM, SED, post-processing, and ensemble sweeps.

Usage:
    from birdclef.sweep.runner import run_sweep
    run_sweep(
        name="cheap_wins",
        configs=[...],
        stage_fn=lambda cfg: {"metrics": {...}, "stage": {...}, "hparams_out": {...}},
        output_root=OUTPUT_ROOT / "sweep",
    )
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from birdclef.eval.metrics import primary_score
from birdclef.sweep.writer import write_config_json, write_hparams_diff_csv, write_summary_csv
from birdclef.utils.seed import seed_everything


def _config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _extract_summary_row(name: str, cfg: dict, result: dict, path: Path) -> dict:
    """Flatten a stage-fn result into the lean CSV schema."""
    metrics = result.get("metrics", {}) or {}
    m_global = metrics.get("global", metrics) or {}
    m_vanchor = metrics.get("v_anchor", {}) or {}
    m_global_fp = metrics.get("global_first_pass", {}) or {}
    m_vanchor_fp = metrics.get("v_anchor_first_pass", {}) or {}
    per_fold = metrics.get("per_fold", {}) or {}
    if per_fold:
        def _fold_auc(v):
            # per_fold may be {"final": {...}, "first_pass": {...}} or flat dict
            if "final" in v and isinstance(v["final"], dict):
                return v["final"].get("macro_auc", np.nan)
            return v.get("macro_auc", np.nan)
        mean_oof = float(np.mean([_fold_auc(v) for v in per_fold.values()]))
    else:
        mean_oof = float(m_global.get("macro_auc", float("nan")))
    v_auc = float(m_vanchor.get("macro_auc", float("nan"))) if m_vanchor else float("nan")
    site_std = float(m_global.get("site_auc_std", 0.0))
    if not np.isnan(v_auc):
        primary = v_auc - 1.0 * (site_std or 0.0)
    else:
        primary = primary_score(m_global, std_penalty=1.0)
    fp_global = float(m_global_fp.get("macro_auc", m_global.get("first_pass_auc", float("nan"))))
    fp_vanchor = float(m_vanchor_fp.get("macro_auc", m_vanchor.get("first_pass_auc", float("nan"))))
    return {
        "config_name": name,
        "primary": primary,
        "macro_auc": m_global.get("macro_auc", float("nan")),
        "first_pass_auc": fp_global,
        "v_anchor_auc": v_auc,
        "v_anchor_first_pass_auc": fp_vanchor,
        "site_auc_std": site_std,
        "mean_oof_auc": mean_oof,
        "rare_auc": m_global.get("rare_auc", float("nan")),
        "frequent_auc": m_global.get("frequent_auc", float("nan")),
        "runtime_min": result.get("runtime_min", float("nan")),
        "stage_metrics_path": str(path),
    }


def run_sweep(
    name: str,
    configs: List[dict],
    stage_fn: Callable[[dict], dict],
    output_root: Path,
    resume: bool = True,
) -> List[dict]:
    output_root = Path(output_root)
    sweep_dir = output_root / name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root / f"{name}_summary.csv"
    hparams_csv = output_root / f"{name}_hparams.csv"

    rows: List[dict] = []
    for i, cfg in enumerate(configs, 1):
        cname = cfg.get("name") or f"cfg_{i:03d}_{_config_hash(cfg)}"
        cfg = {**cfg, "name": cname}
        jpath = sweep_dir / f"{cname}.json"
        if resume and jpath.exists():
            try:
                payload = json.loads(jpath.read_text(encoding="utf-8"))
                row = _extract_summary_row(cname, cfg, payload.get("result", {}), jpath)
                rows.append(row)
                print(f"[sweep:{name}] ({i}/{len(configs)}) skip (cached) {cname}  "
                      f"primary={row['primary']:.4f}")
                write_summary_csv(summary_csv, rows)
                continue
            except Exception:
                pass
        # Deterministic starting state per config — every stage_fn call
        # sees the same RNG sequence regardless of iteration order or
        # whether this is a resumed sweep. Pass `seed` via the config
        # itself to customize (default 42).
        seed_everything(int(cfg.get("seed", 42)))
        t0 = time.time()
        try:
            result = stage_fn(cfg)
        except Exception as exc:
            print(f"[sweep:{name}] ERROR in {cname}: {exc}")
            traceback.print_exc()
            continue
        runtime_min = (time.time() - t0) / 60.0
        result["runtime_min"] = runtime_min
        write_config_json(jpath, {
            "config": cfg,
            "config_hash": _config_hash(cfg),
            "result": result,
        })
        row = _extract_summary_row(cname, cfg, result, jpath)
        rows.append(row)
        write_summary_csv(summary_csv, rows)
        print(f"[sweep:{name}] ({i}/{len(configs)}) done {cname}  "
              f"primary={row['primary']:.4f}  runtime={runtime_min:.2f}m")

    write_hparams_diff_csv(hparams_csv, configs)
    return rows
