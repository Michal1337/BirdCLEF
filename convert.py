#!/usr/bin/env python3

import tensorflow as tf
import torch
import timm
import numpy as np
from pathlib import Path
import re


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


def convert_se_dense_to_conv(w):
    # TF SE kernels are [in, out], while timm expects [out, in, 1, 1].
    return np.transpose(w, (1, 0))[:, :, None, None]


def adapt_stem_conv_channels(w_conv, out_channels):
    # Perch stem is mono; timm EfficientNet-B3 stem is RGB.
    # Repeat across 3 channels and normalize to preserve activation scale.
    if w_conv.shape[1] == 1 and out_channels == 3:
        return np.repeat(w_conv, 3, axis=1) / 3.0
    return w_conv


def mbconv_index(stage_idx, block_idx):
    # EfficientNet-B3 stage depths in timm: [2, 3, 3, 5, 5, 6, 2]
    offsets = [0, 2, 5, 8, 13, 18, 24]
    return offsets[stage_idx] + block_idx


def map_pt_to_tf_name(pt_name):
    if pt_name == "conv_stem.weight":
        return "params.embedding_model.backbone.Stem_0.Conv_0.kernel"

    if pt_name.startswith("bn1."):
        suffix_map = {
            "weight": "params.embedding_model.backbone.Stem_0.BatchNorm_0.scale",
            "bias": "params.embedding_model.backbone.Stem_0.BatchNorm_0.bias",
            "running_mean": "batch_stats.embedding_model.backbone.Stem_0.BatchNorm_0.mean",
            "running_var": "batch_stats.embedding_model.backbone.Stem_0.BatchNorm_0.var",
        }
        k = pt_name.split(".", 1)[1]
        return suffix_map.get(k)

    if pt_name == "conv_head.weight":
        return "params.embedding_model.backbone.Head_0.Conv_0.kernel"

    if pt_name.startswith("bn2."):
        suffix_map = {
            "weight": "params.embedding_model.backbone.Head_0.BatchNorm_0.scale",
            "bias": "params.embedding_model.backbone.Head_0.BatchNorm_0.bias",
            "running_mean": "batch_stats.embedding_model.backbone.Head_0.BatchNorm_0.mean",
            "running_var": "batch_stats.embedding_model.backbone.Head_0.BatchNorm_0.var",
        }
        k = pt_name.split(".", 1)[1]
        return suffix_map.get(k)

    m = re.match(r"^blocks\.(\d+)\.(\d+)\.(.+)$", pt_name)
    if not m:
        return None

    stage_idx = int(m.group(1))
    block_idx = int(m.group(2))
    local_name = m.group(3)
    mb_idx = mbconv_index(stage_idx, block_idx)
    base = f"embedding_model.backbone.MBConv_{mb_idx}"

    if local_name == "conv_dw.weight":
        return f"params.{base}.DepthwiseConv.kernel"

    if local_name == "conv_pw.weight":
        # First stage blocks have no expansion and use conv_pw as projection.
        if mb_idx < 2:
            return f"params.{base}.ProjectConv.kernel"
        return f"params.{base}.ExpandConv.kernel"

    if local_name == "conv_pwl.weight":
        return f"params.{base}.ProjectConv.kernel"

    if local_name.startswith("se.conv_reduce"):
        leaf = local_name.split(".")[-1]
        if leaf == "weight":
            return f"params.{base}.SqueezeAndExcitation_0.Reduce.kernel"
        if leaf == "bias":
            return f"params.{base}.SqueezeAndExcitation_0.Reduce.bias"

    if local_name.startswith("se.conv_expand"):
        leaf = local_name.split(".")[-1]
        if leaf == "weight":
            return f"params.{base}.SqueezeAndExcitation_0.Expand.kernel"
        if leaf == "bias":
            return f"params.{base}.SqueezeAndExcitation_0.Expand.bias"

    bn_map = {
        "weight": "scale",
        "bias": "bias",
        "running_mean": "mean",
        "running_var": "var",
    }

    for bn_idx in ("bn1", "bn2", "bn3"):
        prefix = f"{bn_idx}."
        if local_name.startswith(prefix):
            stat = bn_map.get(local_name.split(".")[-1])
            if stat is None:
                return None

            if mb_idx < 2:
                # No expansion in MBConv_0 and MBConv_1.
                bn_tf = {
                    "bn1": "DepthwiseBatchNorm",
                    "bn2": "ProjectBatchNorm",
                }.get(bn_idx)
            else:
                bn_tf = {
                    "bn1": "ExpandBatchNorm",
                    "bn2": "DepthwiseBatchNorm",
                    "bn3": "ProjectBatchNorm",
                }.get(bn_idx)

            if bn_tf is None:
                return None

            root = "params" if stat in ("scale", "bias") else "batch_stats"
            return f"{root}.{base}.{bn_tf}.{stat}"

    return None


def convert_tf_weight_for_pt(tf_weight, tf_name, pt_name, pt_shape):
    if tf_weight.ndim == 4:
        out = convert_conv(tf_weight)
        if pt_name == "conv_stem.weight":
            out = adapt_stem_conv_channels(out, pt_shape[1])
        if list(out.shape) == list(pt_shape):
            return out

    if tf_weight.ndim == 2:
        if "se.conv_" in pt_name and pt_name.endswith(".weight"):
            out = convert_se_dense_to_conv(tf_weight)
            if list(out.shape) == list(pt_shape):
                return out

        out = convert_dense(tf_weight)
        if list(out.shape) == list(pt_shape):
            return out

    if list(tf_weight.shape) == list(pt_shape):
        return tf_weight

    return None


# =============================
# 4. MATCHING FUNCTION
# =============================
def match_and_convert(tf_vars, pt_model):
    pt_state = pt_model.state_dict()
    new_state = {}
    used = set()

    def try_match_by_name(pt_name, pt_shape):
        tf_name = map_pt_to_tf_name(pt_name)
        if tf_name is None or tf_name in used or tf_name not in tf_vars:
            return None, None

        converted = convert_tf_weight_for_pt(tf_vars[tf_name], tf_name, pt_name, pt_shape)
        if converted is None:
            return None, None

        used.add(tf_name)
        return tf_name, converted

    def try_match_by_shape(pt_name, pt_shape):
        # Fallback for any still-unmapped tensors.
        for tf_name, w in tf_vars.items():
            if tf_name in used:
                continue

            converted = convert_tf_weight_for_pt(w, tf_name, pt_name, pt_shape)
            if converted is not None:
                used.add(tf_name)
                return tf_name, converted

        return None, None

    matched = 0
    matched_by_name = 0
    matched_by_shape = 0

    for pt_name, pt_weight in pt_state.items():
        tf_name, tf_weight = try_match_by_name(pt_name, pt_weight.shape)
        match_kind = "name"

        if tf_weight is None:
            tf_name, tf_weight = try_match_by_shape(pt_name, pt_weight.shape)
            match_kind = "shape"

        if tf_weight is not None:
            new_state[pt_name] = torch.tensor(tf_weight, dtype=pt_weight.dtype)
            matched += 1
            if match_kind == "name":
                matched_by_name += 1
            else:
                matched_by_shape += 1
            print(f"[OK] {pt_name} <- {tf_name}")
        else:
            print(f"[MISS] {pt_name}")

    print(f"\n[SUMMARY] Matched {matched}/{len(pt_state)} layers")
    print(f"[SUMMARY] Matched by name: {matched_by_name}")
    print(f"[SUMMARY] Matched by shape fallback: {matched_by_shape}")
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

    print("Done!")


if __name__ == "__main__":
    main()