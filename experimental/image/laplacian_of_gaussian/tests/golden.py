# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# NOTE: Portions of this code were AI-generated and have been
# technically reviewed for functional accuracy and security
# ----------------------------------------------------------------------------------------------------------
import math
import numpy as np


def weights(s, r):
    g = np.array(
        [math.exp(-0.5 * (i / s) ** 2) for i in range(-r, r + 1)], dtype=np.float64
    )
    g /= sum(g)
    d = np.array(
        [((i / s) ** 2 - 1) / (s * s) * v for i, v in zip(range(-r, r + 1), g)]
    )
    return g, d


def direct(x, s, r, border):
    g, d = weights(s, r)
    kernel = np.outer(d, g) + np.outer(g, d)
    h, w = x.shape[-2:]
    xx = x.reshape(-1, h, w).astype(np.float64)

    # Independent coordinate map, applying every 2D tap to the complete image.
    def idx(q, L):
        if border == "CONSTANT":
            return np.clip(q, 0, L - 1), (q >= 0) & (q < L)
        if border == "REPLICATE":
            return np.clip(q, 0, L - 1), np.ones(q.shape, bool)
        if L == 1:
            return np.zeros(q.shape, dtype=int), np.ones(q.shape, bool)
        if border == "REFLECT":
            z = np.mod(q, 2 * L)
            return np.where(z < L, z, 2 * L - 1 - z), np.ones(q.shape, bool)
        z = np.mod(q, 2 * (L - 1))
        return np.where(z < L, z, 2 * (L - 1) - z), np.ones(q.shape, bool)

    yy = np.zeros_like(xx)
    for i in range(-r, r + 1):
        yi, ym = idx(np.arange(h) + i, h)
        for j in range(-r, r + 1):
            xi, xm = idx(np.arange(w) + j, w)
            yy += (
                kernel[i + r, j + r]
                * xx[:, yi[:, None], xi[None, :]]
                * (ym[:, None] & xm[None, :])
            )
    return yy.reshape(x.shape), g, d
