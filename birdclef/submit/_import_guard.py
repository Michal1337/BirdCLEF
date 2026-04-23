"""Fails loudly if submit-side code tries to pull in training deps.

Run as `python -m birdclef.submit._import_guard` after editing submit/.
"""
import importlib
import sys

FORBIDDEN = ["torchaudio", "timm", "birdclef.train"]


def check() -> None:
    bad = [name for name in FORBIDDEN if name in sys.modules]
    if bad:
        raise RuntimeError(
            f"Submit-side import guard tripped: forbidden modules already imported: {bad}. "
            f"Submit code must stay lean to fit Kaggle's CPU runtime."
        )


if __name__ == "__main__":
    importlib.import_module("birdclef.submit.inference_template")
    check()
    print("submit/ import guard: OK")
