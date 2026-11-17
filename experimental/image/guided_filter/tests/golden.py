#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""GuidedFilter golden 实现（float64 积分图，边界 BORDER_REFLECT，语义对齐 cv::ximgproc::guidedFilter）。"""

import numpy as np

__golden__ = {"kernel": {"guided_filter": "guided_filter_golden"}}

# OpenCV 位深常量：CV_8U=0, CV_32F=5, CV_16F=7, -1 表示跟随 src
_CV_DEPTH_TO_DTYPE = {
    0: np.uint8,
    5: np.float32,
    7: np.float16,
}


def _reflect_index(idx, n):
    """BORDER_REFLECT：period = 2n，折叠到 [0, 2n) 后反射（与 kernel 的 MirrorIdx 一致）。"""
    period = 2 * n
    m = np.mod(idx, period)
    return np.where(m < n, m, period - 1 - m)


def _box_mean(a, radius):
    """半径 radius 的均值滤波，float64 积分图实现，O(1)/像素。"""
    if radius == 0:
        return a.astype(np.float64).copy()
    h, w = a.shape
    ys = _reflect_index(np.arange(-radius, h + radius), h)
    xs = _reflect_index(np.arange(-radius, w + radius), w)
    p = a.astype(np.float64)[np.ix_(ys, xs)]
    c = np.zeros((p.shape[0] + 1, p.shape[1] + 1), dtype=np.float64)
    c[1:, 1:] = p.cumsum(0).cumsum(1)
    k = 2 * radius + 1
    s = c[k : k + h, k : k + w] - c[0:h, k : k + w] - c[k : k + h, 0:w] + c[0:h, 0:w]
    return s / float(k * k)


def _to_hwc(x):
    x = np.asarray(x)
    if x.ndim == 2:
        return x[:, :, None], "hw"
    if x.ndim == 3:
        return x, "hwc"
    if x.ndim == 4:
        return x, "nhwc"
    raise ValueError("unsupported rank %d" % x.ndim)


def guided_filter_golden(
    guide, src, radius=4, eps=0.01, dDepth=-1, scale=1.0, **kwargs
):
    """guide: [H,W] / [H,W,C] / [N,H,W,C]；src 空间尺寸一致、通道数可不同。"""
    del kwargs  # 兼容框架透传的多余参数
    radius = int(radius)
    assert radius >= 1, "radius must be >= 1"
    assert abs(float(scale) - 1.0) < 1e-6, "only scale == 1 is supported"

    g, g_layout = _to_hwc(guide)
    p, p_layout = _to_hwc(src)
    assert g_layout == p_layout, "guide and src must have the same rank"
    spatial_g = g.shape[:-1] if g_layout == "nhwc" else g.shape[:2]
    spatial_p = p.shape[:-1] if p_layout == "nhwc" else p.shape[:2]
    assert spatial_g == spatial_p, "guide/src spatial size mismatch"

    c_num = g.shape[-1]
    s_num = p.shape[-1]
    assert 1 <= c_num <= 3, "guide channels must be in [1,3]"

    # 展开成 [N, H, W, C] 便于逐图二维滤波
    if g_layout == "hw":
        gh, gw = g.shape[0], g.shape[1]
    else:
        gh, gw = g.shape[-3], g.shape[-2]
    gg = g.astype(np.float64).reshape(-1, gh, gw, c_num)
    pp = p.astype(np.float64).reshape(-1, gh, gw, s_num)
    out = np.zeros_like(pp)

    for b in range(gg.shape[0]):
        gb = gg[b]
        pb = pp[b]
        mu = np.stack([_box_mean(gb[:, :, c], radius) for c in range(c_num)], axis=-1)
        nu = np.stack([_box_mean(pb[:, :, s], radius) for s in range(s_num)], axis=-1)

        sigma = np.zeros(gb.shape[:2] + (c_num, c_num), dtype=np.float64)
        for c in range(c_num):
            for d in range(c, c_num):
                v = (
                    _box_mean(gb[:, :, c] * gb[:, :, d], radius)
                    - mu[:, :, c] * mu[:, :, d]
                )
                sigma[:, :, c, d] = v
                sigma[:, :, d, c] = v
        for c in range(c_num):
            sigma[:, :, c, c] += float(eps)

        cov = np.zeros(gb.shape[:2] + (c_num, s_num), dtype=np.float64)
        for c in range(c_num):
            for s in range(s_num):
                cov[:, :, c, s] = (
                    _box_mean(gb[:, :, c] * pb[:, :, s], radius)
                    - mu[:, :, c] * nu[:, :, s]
                )

        if c_num == 1:
            # 1x1 直接取倒数；统一保持 [H, W, C, Cs] 形状，便于后续索引
            alpha = np.zeros(gb.shape[:2] + (1, s_num), dtype=np.float64)
            alpha[:, :, 0, :] = cov[:, :, 0, :] * (1.0 / sigma[:, :, 0, 0])[:, :, None]
        else:
            alpha = np.einsum("hwcd,hwds->hwcs", np.linalg.inv(sigma), cov)

        beta = nu - np.einsum("hwc,hwcs->hws", mu, alpha)

        for s in range(s_num):
            acc = _box_mean(beta[:, :, s], radius)
            for c in range(c_num):
                acc = acc + _box_mean(alpha[:, :, c, s], radius) * gb[:, :, c]
            out[b, :, :, s] = acc

    if g_layout == "hw":
        res = out[:, :, :, 0]
    elif g_layout == "hwc":
        res = out[0]
    else:
        res = out
    res = np.array(res, copy=True)

    out_dtype = _CV_DEPTH_TO_DTYPE.get(int(dDepth), np.asarray(src).dtype)
    if out_dtype == np.uint8:
        res = np.clip(np.rint(res), 0, 255)
    else:
        finfo = np.finfo(out_dtype)
        res = np.clip(res, finfo.min, finfo.max)
    res = res.astype(out_dtype)

    return res
