#!/usr/bin/env python3

import tensorflow as tf
import torch
import timm
import numpy as np
from pathlib import Path


# =============================
# 1. LOAD PERCH VIA TFSMLayer
# =============================
def load_perch_weights(model_dir):
    class PerchWrapper(tf.keras.Model):
        def __init__(self, path):
            super().__init__()
            self.perch = tf.keras.layers.TFSMLayer(
                str(path), call_endpoint="serving_default"
            )

        def call(self, x):
            out = self.perch(x)
            if isinstance(out, dict) and "embedding" in out:
                return out["embedding"]
            return out

    model = PerchWrapper(model_dir)
    model.build((None, 160000))

    tf_vars = {
        v.name.replace(":0", ""): v.numpy()
        for v in model.weights
    }

    print(f"[INFO] Extracted {len(tf_vars)} TF variables")
    return tf_vars


# =============================
# 2. LOAD PYTORCH MODEL
# =============================
def build_pytorch_model():
    model = timm.create_model(
        "efficientnet_b3",
        pretrained=False,
        num_classes=0,
        global_pool=""
    )
    return model


# =============================
# 3. CONVERSION HELPERS
# =============================
def convert_conv(w):
    return np.transpose(w, (3, 2, 0, 1))


def convert_dense(w):
    return w.T


def convert_dwconv(w):
    return np.transpose(w, (3, 2, 0, 1))


# =============================
# 4. MATCHING FUNCTION
# =============================
def match_and_convert(tf_vars, pt_model):
    pt_state = pt_model.state_dict()
    new_state = {}
    used = set()

    def try_match(pt_name, pt_shape):
        for tf_name, w in tf_vars.items():
            if tf_name in used:
                continue

            # direct match
            if list(w.shape) == list(pt_shape):
                used.add(tf_name)
                return tf_name, w

            # conv
            if len(w.shape) == 4:
                w_conv = convert_conv(w)
                if list(w_conv.shape) == list(pt_shape):
                    used.add(tf_name)
                    return tf_name, w_conv

            # dense
            if len(w.shape) == 2:
                w_fc = convert_dense(w)
                if list(w_fc.shape) == list(pt_shape):
                    used.add(tf_name)
                    return tf_name, w_fc

        return None, None

    matched = 0

    for pt_name, pt_weight in pt_state.items():
        tf_name, tf_weight = try_match(pt_name, pt_weight.shape)

        if tf_weight is not None:
            new_state[pt_name] = torch.tensor(tf_weight)
            matched += 1
            print(f"[OK] {pt_name} ← {tf_name}")
        else:
            print(f"[MISS] {pt_name}")

    print(f"\n[SUMMARY] Matched {matched}/{len(pt_state)} layers")
    return new_state


# =============================
# 5. MAIN
# =============================
def main():
    model_dir = Path("models/perch_v2/1")

    print("[STEP] Loading TF weights...")
    tf_vars = load_perch_weights(model_dir)

    print("[STEP] Building PyTorch model...")
    pt_model = build_pytorch_model()

    print("[STEP] Matching weights...")
    new_state = match_and_convert(tf_vars, pt_model)

    print("[STEP] Loading into PyTorch...")
    pt_model.load_state_dict(new_state, strict=False)

    print("[STEP] Saving...")
    torch.save(pt_model.state_dict(), "perch_backbone.pt")

    print("✅ Done!")


if __name__ == "__main__":
    main()