"""Build a Kaggle .ipynb that runs SSM (LB-0.93 stack) and SED 5-fold blended.

Forks the in-repo LB_093.ipynb and appends two cells (SED inference, blend).
The output is a single self-contained notebook — Kaggle only accepts .ipynb
uploads to the leaderboard, so everything must live inside one file.

Examples:

    # Default 90/10 blend with default Kaggle dataset path:
    python -m birdclef.scripts._10_build_blend_notebook \\
        --out outputs/submit/blend_w90.ipynb

    # Custom weight + explicit SED ONNX paths (most reliable on Kaggle):
    python -m birdclef.scripts._10_build_blend_notebook \\
        --out outputs/submit/blend_w85.ipynb \\
        --w-ssm 0.85 --w-sed 0.15 \\
        --sed-onnx \\
            /kaggle/input/birdclef-sed-onnx/fold0/best.onnx \\
            /kaggle/input/birdclef-sed-onnx/fold1/best.onnx \\
            /kaggle/input/birdclef-sed-onnx/fold2/best.onnx \\
            /kaggle/input/birdclef-sed-onnx/fold3/best.onnx \\
            /kaggle/input/birdclef-sed-onnx/fold4/best.onnx
"""
from __future__ import annotations

from birdclef.submit.build_blend_notebook import main

if __name__ == "__main__":
    main()
