#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# ----------------------------------------------------------------------------

"""ExtractGlimpseV2 TestSpec: golden / third_party.

Golden is an exact numpy port of the Ascend SIMT kernel semantics
(NsExtractGlimpseV2, arch35): clamp-to-bounds crop + zero fill, float32
coordinate transform in the kernel's operation order. Verified equivalent to
the direct competitor reference tf.raw_ops.ExtractGlimpseV2(noise="zero") on
208 combinatorial cases (fractional/negative/out-of-bounds/inf/NaN offsets,
all four centered/normalized combos), and to the device output of
ExtractGlimpseV2_L0_001.
"""

import numpy as np

__spec__ = {
    "ExtractGlimpseV2": "ExtractGlimpseV2TestSpec",
}

__golden__ = {"kernel": {"ExtractGlimpseV2": "extract_glimpse_v2_golden"}}

# Third-party reference for cross_check: TF ExtractGlimpseV2 is the direct
# competitor of this op (identical zero-noise semantics, see module docstring).
_THIRD_PARTY = {"tf": "tf.raw_ops.ExtractGlimpseV2"}

F32 = np.float32


class ExtractGlimpseV2TestSpec:
    """ExtractGlimpseV2 kernel spec. The golden entry receives numpy arrays."""

    @staticmethod
    def golden(
        input,
        size,
        offsets,
        centered=True,
        normalized=True,
        uniform_noise=True,
        noise="uniform",
        **kwargs,
    ):
        return extract_glimpse_v2_golden(
            input,
            size,
            offsets,
            centered=centered,
            normalized=normalized,
            uniform_noise=uniform_noise,
            noise=noise,
            **kwargs,
        )

    third_party = _THIRD_PARTY

    # 精度标准与交付件 thirdparty_golden 声明保持一致：cross_check + L1，
    # 三方锚点为 tf.raw_ops.ExtractGlimpseV2（见 _THIRD_PARTY），执行需三方 endpoint 在线。
    tolerance = {"float32": {"standard": "cross_check", "level": "L1"}}


def _flatten_dtypes(d):
    out = []
    for e in d if isinstance(d, (list, tuple)) else (d,):
        if isinstance(e, (list, tuple)):
            out.extend(_flatten_dtypes(e))
        else:
            out.append(e)
    return out


def _declared_dtype(kwargs, key, idx):
    """Resolve the CSV-declared dtype for input/output #idx; None if not a numpy dtype."""
    dtypes = _flatten_dtypes(kwargs.get(key) or ())
    if len(dtypes) <= idx:
        return None
    try:
        return np.dtype(str(dtypes[idx]))
    except TypeError:
        return None


def extract_glimpse_v2_golden(
    input,
    size,
    offsets,
    centered=True,
    normalized=True,
    uniform_noise=True,
    noise="uniform",
    **kwargs,
):
    """Golden function for ExtractGlimpseV2.

    All the parameters (names and order) follow @extract_glimpse_v2_def.cpp
    without outputs. All the input Tensors are numpy.ndarray.

    Semantics (mirrors the arch35 SIMT kernel exactly, float32 arithmetic in
    the kernel's operation order):
      NaN offsets        -> whole glimpse of that batch item is 0
      +/-inf offsets     -> copyH/copyW underflow to non-positive -> all 0
      normalized         -> offsetY *= H ; offsetX *= W
      centered+normalized-> offsetY = offY*0.5 + H*0.5 - gH*0.5 (top-left)
      centered only      -> offsetY = offY + H*0.5
      crop window [startY,endY) is clamped to the image and copied into the
      glimpse at [baseY,...) with floor/ceil bounds; outside -> 0
    """
    x = np.asarray(input)
    off = np.asarray(offsets)
    sz = np.asarray(size).reshape(-1)

    in_dtype = _declared_dtype(kwargs, "input_dtypes", 0)
    out_dtype = _declared_dtype(kwargs, "output_dtypes", 0)
    if (
        in_dtype is not None
        and x.dtype != in_dtype
        and (
            np.issubdtype(in_dtype, np.floating) or np.issubdtype(in_dtype, np.integer)
        )
    ):
        x = x.astype(in_dtype)

    if sz.size != 2:
        raise ValueError(f"size must have exactly 2 elements, got shape {sz.shape}")
    if not np.issubdtype(sz.dtype, np.integer):
        sz = sz.astype(np.int64)
    gH, gW = int(sz[0]), int(sz[1])
    if gH < 0 or gW < 0:
        raise ValueError(f"glimpse size must be non-negative, got [{gH}, {gW}]")

    B, H, W, C = x.shape
    out = np.zeros((B, gH, gW, C), dtype=x.dtype)

    half = F32(0.5)
    off = off.astype(F32).reshape(B, 2) if off.size == B * 2 else off.reshape(-1, 2)

    for b in range(B):
        offY = F32(off[b, 0])
        offX = F32(off[b, 1])
        if np.isnan(offY) or np.isnan(offX):
            continue
        if normalized:
            offY = F32(offY * F32(H))
            offX = F32(offX * F32(W))
        if centered:
            if normalized:
                offY = F32(
                    F32(F32(offY * half) + F32(F32(H) * half)) - F32(F32(gH) * half)
                )
                offX = F32(
                    F32(F32(offX * half) + F32(F32(W) * half)) - F32(F32(gW) * half)
                )
            else:
                offY = F32(offY + F32(F32(H) * half))
                offX = F32(offX + F32(F32(W) * half))
        if np.isinf(offY) or np.isinf(offX):
            continue
        startY = int(np.floor(np.maximum(F32(0.0), offY)))
        startX = int(np.floor(np.maximum(F32(0.0), offX)))
        endY = int(np.ceil(np.minimum(F32(H), F32(offY + F32(gH)))))
        endX = int(np.ceil(np.minimum(F32(W), F32(offX + F32(gW)))))
        baseY = int(np.floor(np.maximum(F32(0.0), F32(-offY))))
        baseX = int(np.floor(np.maximum(F32(0.0), F32(-offX))))
        copyH = endY - startY
        copyW = endX - startX
        if copyH <= 0 or copyW <= 0 or H <= 0 or W <= 0:
            continue
        localY = np.arange(gH, dtype=np.int64) - baseY
        localX = np.arange(gW, dtype=np.int64) - baseX
        validY = (localY >= 0) & (localY < copyH)
        validX = (localX >= 0) & (localX < copyW)
        if not validY.any() or not validX.any():
            continue
        srcY = np.clip(startY + localY, 0, H - 1)
        srcX = np.clip(startX + localX, 0, W - 1)
        mask = validY[:, None] & validX[None, :]
        region = x[b][srcY][:, srcX]
        out[b] = np.where(mask[:, :, None], region, 0)

    if (
        out_dtype is not None
        and out.dtype != out_dtype
        and (
            np.issubdtype(out_dtype, np.floating)
            or np.issubdtype(out_dtype, np.integer)
        )
    ):
        out = out.astype(out_dtype)
    return out
