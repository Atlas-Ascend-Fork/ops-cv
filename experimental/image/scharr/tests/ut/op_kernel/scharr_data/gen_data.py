#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Generate input + golden for Scharr kernel UT (pure numpy, no cv2 dependency).

The golden reproduces OpenCV's float32 Scharr (ddepth=CV_32F) bit-exactly:
  * two separable passes (row kernel then column kernel, as in sepFilter2D);
  * main columns use fma(10, center, 3*(a+c)) / fma(3, (a+c), 10*b);
  * the rightmost column of the row-smooth pass uses the coefficient-combined
    form fma(3, (a+c), 10*b) when W is odd (verified empirically vs cv2 5.x).
Validated bitwise against cv2 in development on 162 configs (shape x dtype x
direction x border).
"""

import argparse
import json
from pathlib import Path

import numpy as np

F3 = np.float32(3)
F10 = np.float32(10)


def fma32(a, b, c):
    """Single-rounded fma(a, b, c) in float32 (product exact in float64)."""
    return (np.float64(a) * np.float64(b) + np.float64(c)).astype(np.float32)


def idxmap(n, border):
    idx = np.arange(-1, n + 1)
    if border == "REFLECT_101":
        idx = np.where(idx < 0, -idx, idx)
        idx = np.where(idx >= n, 2 * n - 2 - idx, idx)
    elif border == "REPLICATE":
        idx = np.clip(idx, 0, n - 1)
    return idx


def pad_reflect(x, border):
    h, w = x.shape
    iy = idxmap(h, border)
    ix = idxmap(w, border)
    yi = np.clip(iy, 0, h - 1)
    xi = np.clip(ix, 0, w - 1)
    sub = x[np.ix_(yi, xi)].astype(np.float32)
    if border == "CONSTANT":
        mask = ((iy >= 0) & (iy < h))[:, None] & ((ix >= 0) & (ix < w))[None, :]
        sub = np.where(mask, sub, np.float32(0)).astype(np.float32)
    return sub


def smooth_row(a, b, c, last_col=False):
    t = (a + c).astype(np.float32)
    if last_col:
        return fma32(F3, t, F10 * b)
    return fma32(F10, b, F3 * t)


def smooth_col(a, b, c):
    t = (a + c).astype(np.float32)
    return fma32(F3, t, F10 * b)


def diff_anti(a, c):
    return (np.float32(-1) * a + c).astype(np.float32)


def scharr_plane(x, direction, border):
    x = x.astype(np.float32)
    w = x.shape[1]
    p = pad_reflect(x, border)
    left = p[1:-1, 0:-2]
    center = p[1:-1, 1:-1]
    right = p[1:-1, 2:]
    if direction == "X":
        r = diff_anti(left, right)
        rp = pad_reflect(r, border)
        return smooth_col(rp[0:-2, 1:-1], rp[1:-1, 1:-1], rp[2:, 1:-1])
    r = smooth_row(left, center, right)
    if w % 2 == 1:
        r[:, w - 1] = smooth_row(
            left[:, w - 1], center[:, w - 1], right[:, w - 1], last_col=True
        )
    rp = pad_reflect(r, border)
    return diff_anti(rp[0:-2, 1:-1], rp[2:, 1:-1])


def compute_golden(x, direction, border):
    dims = x.shape
    h, w = dims[-2], dims[-1]
    batch = int(np.prod(dims[:-2])) if len(dims) > 2 else 1
    src = x.reshape(batch, h, w)
    golden = np.empty((batch, h, w), dtype=np.float32)
    for b in range(batch):
        golden[b] = scharr_plane(src[b], direction, border)
    return golden.reshape(dims)


def generate_input(rng, shape, dtype, min_value, max_value, pattern):
    np_dtype = np.dtype(dtype)
    if pattern == "dtype_edges" and np_dtype == np.dtype(np.uint8):
        return np.resize(
            np.array([0, 1, 254, 255], dtype=np.uint8), int(np.prod(shape))
        ).reshape(shape)
    if np_dtype == np.dtype(np.uint8):
        return rng.integers(
            int(min_value), int(max_value) + 1, size=shape, dtype=np.uint8
        )
    return rng.uniform(min_value, max_value, size=shape).astype(np_dtype)


def main():
    parser = argparse.ArgumentParser(description="生成 Scharr 单 case 数据")
    parser.add_argument("--shape", required=True)
    parser.add_argument("--direction", required=True, choices=("X", "Y"))
    parser.add_argument(
        "--border-type", required=True, choices=("REFLECT_101", "REPLICATE", "CONSTANT")
    )
    parser.add_argument(
        "--dtype", required=True, choices=("uint8", "float16", "float32")
    )
    parser.add_argument("--min-value", type=float, default=0.0)
    parser.add_argument("--max-value", type=float, default=255.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pattern", default="random", choices=("random", "dtype_edges")
    )
    args = parser.parse_args()

    shape = tuple(int(v) for v in args.shape.split(","))
    rng = np.random.default_rng(args.seed)
    input_x = generate_input(
        rng, shape, args.dtype, args.min_value, args.max_value, args.pattern
    )
    golden = compute_golden(input_x, args.direction, args.border_type)

    Path("input").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    input_x.tofile(Path("input") / "input_scharr.bin")
    golden.tofile(Path("output") / "golden.bin")

    (Path("output") / "case_meta.json").write_text(
        json.dumps(
            {
                "shape": list(shape),
                "input_dtype": args.dtype,
                "output_dtype": "float32",
                "direction": args.direction,
                "border_type": args.border_type,
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"generated shape={shape}, dtype={args.dtype}, direction={args.direction}, "
        f"border={args.border_type}, seed={args.seed}"
    )


if __name__ == "__main__":
    main()
