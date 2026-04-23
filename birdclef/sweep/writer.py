"""Lean summary CSV + full per-config JSON + hparams diff CSV.

The summary CSV is atomically rewritten after every config so a crash never
loses prior rows. Sorting is by `primary` desc; tiebreak by mean_oof_auc desc.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

from birdclef.sweep.schema import SUMMARY_COLUMNS, round_summary_row


def _atomic_write(path: Path, fn) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8",
                                     dir=str(path.parent), delete=False) as f:
        fn(f)
        tmp = f.name
    os.replace(tmp, path)


def write_summary_csv(path: Path, rows: List[dict]) -> None:
    rows = [round_summary_row(r) for r in rows]
    rows.sort(
        key=lambda r: (
            -(r.get("primary") or float("-inf")),
            -(r.get("mean_oof_auc") or float("-inf")),
        )
    )
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    def _w(f):
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in SUMMARY_COLUMNS})

    _atomic_write(path, _w)


def write_config_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o):
    import numpy as np

    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _varied_keys(configs: Iterable[dict]) -> list:
    cfgs = [dict(c) for c in configs]
    if not cfgs:
        return []
    keys = set().union(*[set(c.keys()) for c in cfgs])
    varied = []
    for k in sorted(keys):
        vals = [json.dumps(c.get(k), sort_keys=True, default=str) for c in cfgs]
        if len(set(vals)) > 1:
            varied.append(k)
    return varied


def write_hparams_diff_csv(path: Path, configs: Iterable[dict]) -> None:
    cfgs = list(configs)
    if not cfgs:
        return
    varied = _varied_keys(cfgs)
    cols = ["config_name", *varied]

    def _w(f):
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for c in cfgs:
            row = {"config_name": c.get("name", "")}
            for k in varied:
                v = c.get(k, "")
                if isinstance(v, (list, tuple, dict)):
                    v = json.dumps(v, sort_keys=True, default=str)
                row[k] = v
            writer.writerow(row)

    _atomic_write(path, _w)
