"""Preprocessing utilities for MTNCL experiments."""

from __future__ import annotations

from typing import List
import numpy as np


def downsample_pool(img_flat: np.ndarray, out_hw: int = 7) -> np.ndarray:
    """Average-pool 28x28 image to out_hw x out_hw."""
    img = img_flat.reshape(28, 28)
    k = 28 // out_hw
    pooled = img.reshape(out_hw, k, out_hw, k).mean(axis=(1, 3))
    return pooled


def encode_binary(pooled: np.ndarray, threshold: float = 64.0) -> List[float]:
    return (pooled > threshold).astype(float).reshape(-1).tolist()


def encode_multibit(pooled: np.ndarray, bits: int = 2) -> List[float]:
    """Quantize pooled pixels and emit bit planes (LSB..MSB per pixel)."""
    levels = (1 << bits)
    q = np.clip((pooled / 256.0) * levels, 0, levels - 1e-9).astype(int)
    feats = []
    flat = q.reshape(-1)
    for v in flat:
        for b in range(bits):
            feats.append(float((v >> b) & 1))
    return feats


def handcrafted_features(pooled: np.ndarray) -> List[float]:
    """Simple binary geometry features to help MTNCL classification."""
    x = pooled / 255.0
    h, w = x.shape
    s = (x > 0.25).astype(float)

    # zoning features (3x3)
    zones = []
    zh, zw = h // 3, w // 3
    for r in range(3):
        for c in range(3):
            block = x[r * zh:(r + 1) * zh, c * zw:(c + 1) * zw]
            zones.append(float(block.mean() > 0.2))

    # stroke density by row/col halves
    top = float(s[: h // 2].mean() > 0.2)
    bottom = float(s[h // 2 :].mean() > 0.2)
    left = float(s[:, : w // 2].mean() > 0.2)
    right = float(s[:, w // 2 :].mean() > 0.2)

    # symmetry proxies
    vflip = np.fliplr(s)
    hflip = np.flipud(s)
    vert_sym = float((1.0 - np.abs(s - vflip).mean()) > 0.7)
    horz_sym = float((1.0 - np.abs(s - hflip).mean()) > 0.7)

    return zones + [top, bottom, left, right, vert_sym, horz_sym]


def build_features(img_flat: np.ndarray, mode: str = "binary") -> List[float]:
    pooled = downsample_pool(img_flat, out_hw=7)
    if mode == "binary":
        return encode_binary(pooled)
    if mode == "multibit2":
        return encode_multibit(pooled, bits=2)
    if mode == "multibit4":
        return encode_multibit(pooled, bits=4)
    if mode == "hybrid2":
        return encode_multibit(pooled, bits=2) + handcrafted_features(pooled)
    raise ValueError(f"Unknown feature mode: {mode}")
