#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IntegralImage op_kernel UT data generator.

用法: python3 gen_data.py <out_dir> <tag> <H> <W> <C> <in_dtype> <sat_dtype>

生成文件（<tag> 为用例标识）:
  <tag>_input.bin        输入图像，rank3 为 (H, W, C) HWC 布局，rank2(C==1) 为 (H, W)
  <tag>_golden_sat.bin   sat 参考结果 (H+1, W+1[, C])，dtype 由 sat_dtype 指定
  <tag>_golden_sqsum.bin sqsum 参考结果 (H+1, W+1[, C])，固定 float32
  <tag>_golden_tilted.bin tilted 参考结果 (H+1, W+1[, C])，dtype 与 sat 相同

参考语义（与 examples/test_aclnn_integral_image.cpp 的 CPU 参考一致）:
  sat[y][x]    = sum_{i<y, j<x} src[i][j]                                  (OpenCV cv::integral sum)
  sqsum[y][x]  = sum_{i<y, j<x} src[i][j]^2
  tilted[y][x] = sum_{k<=y} sum_{j in [x-(y-k), x+(y-k)] clipped} src[k][j]
边界: sat/sqsum 的第 0 行与第 0 列为 0；tilted 的第 H 行与第 W 列为 0，
      tilted 的第 0 行等于输入第 0 行。
"""

import os
import sys

import numpy as np

# 固定随机种子，保证每次生成的输入完全可复现
SEED = 20260919

IN_DTYPES = {"uint8": np.uint8, "float16": np.float16, "float32": np.float32}
SAT_DTYPES = {"int32": np.int32, "float32": np.float32}


def build_src(h, w, c, in_dtype):
    rng = np.random.default_rng(SEED)
    shape = (h, w) if c == 1 else (h, w, c)
    if in_dtype == "uint8":
        # 0..255 全范围，覆盖累加可能出现的进位
        return rng.integers(0, 256, size=shape).astype(np.uint8)
    # 浮点输入使用整数取值，便于 golden 与 kernel 结果做严格比较
    return rng.integers(0, 256, size=shape).astype(IN_DTYPES[in_dtype])


def compute_golden(plane):
    h, w = plane.shape
    src = plane.astype(np.float64)
    sat = np.zeros((h + 1, w + 1), dtype=np.float64)
    sat[1:, 1:] = src.cumsum(axis=0).cumsum(axis=1)
    sq = src * src
    sqsum = np.zeros((h + 1, w + 1), dtype=np.float64)
    sqsum[1:, 1:] = sq.cumsum(axis=0).cumsum(axis=1)
    tilted = np.zeros((h + 1, w + 1), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            acc = 0.0
            for k in range(y + 1):
                lo = max(0, x - (y - k))
                hi = min(w - 1, x + (y - k))
                acc += float(src[k, lo : hi + 1].sum())
            tilted[y, x] = acc
    return sat, sqsum, tilted


def main():
    if len(sys.argv) != 8:
        print(__doc__)
        return 1
    out_dir, tag, h, w, c, in_dtype, sat_dtype = sys.argv[1:8]
    h, w, c = int(h), int(w), int(c)
    if in_dtype not in IN_DTYPES or sat_dtype not in SAT_DTYPES:
        print("unsupported dtype: %s / %s" % (in_dtype, sat_dtype))
        return 1

    src = build_src(h, w, c, in_dtype)
    sat_dtype_np = SAT_DTYPES[sat_dtype]

    if c == 1:
        sat, sqsum, tilted = compute_golden(src)
    else:
        planes = [compute_golden(src[:, :, ch]) for ch in range(c)]
        sat = np.stack([p[0] for p in planes], axis=-1)
        sqsum = np.stack([p[1] for p in planes], axis=-1)
        tilted = np.stack([p[2] for p in planes], axis=-1)

    os.makedirs(out_dir, exist_ok=True)
    src.tofile(os.path.join(out_dir, tag + "_input.bin"))
    sat.astype(sat_dtype_np).tofile(os.path.join(out_dir, tag + "_golden_sat.bin"))
    sqsum.astype(np.float32).tofile(os.path.join(out_dir, tag + "_golden_sqsum.bin"))
    tilted.astype(sat_dtype_np).tofile(
        os.path.join(out_dir, tag + "_golden_tilted.bin")
    )
    print(
        "gen_data OK: tag=%s H=%d W=%d C=%d in=%s sat=%s"
        % (tag, h, w, c, in_dtype, sat_dtype)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
