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

"""GuidedFilter kernel UT 数据生成。

管线与同仓 kernel UT（pixel_shuffle / rgb2yuv422）保持一致：
  gen_data.py 产出 *input*.bin 与 *golden*.bin；
  test_guided_filter.cpp 把内核结果写成 *output*.bin；
  compare_data.py 逐元素比对。

参考实现用 **numpy float64**，与内核的 C++ float32 实现互相独立，是真正的第二实现：

    mu_c  = box(g_c)                          nu_s  = box(p_s)
    Sigma = box(g_c*g_d) - mu_c*mu_d + eps*I  Omega = box(g_c*p_s) - mu_c*nu_s
    alpha = Sigma^-1 Omega                    beta  = nu - mu·alpha
    q_s   = sum_c box(alpha_cs) * g_c + box(beta_s)

盒式均值用积分图实现，边界 **cv2.BORDER_REFLECT**：`fedcba|abcdefgh|hgfedcb`，
即**端点重复**，对应 numpy 的 `mode="symmetric"`。
（注意别写成 `mode="reflect"`：那是 BORDER_REFLECT_101 `fedcb|abcdefgh|gfedc`，
端点不重复，两者结果差很多。内核的 MirrorIdx 与 examples 里的 C++ 参考实现用的都是
重复端点这一种。）

输入布局与内核一致，为 NHWC 连续：
    guide: (N,H,W,C)   src: (N,H,W,Cs)   输出: (N,H,W,Cs)

用法: gen_data.py '(N,H,W,C,Cs)' <dtype> <radius> <eps> [seed]
"""

import os
import sys

import numpy as np

DTYPE_MAP = {"float32": np.float32, "float16": np.float16}


def parse_shape(shape_str):
    shape_str = shape_str.strip().strip("(").strip(")")
    return tuple(int(x) for x in shape_str.split(","))


def box_mean(x, r):
    """(2r+1)^2 归一化盒式均值，边界 BORDER_REFLECT。"""
    k = 2 * r + 1
    h, w = x.shape[0], x.shape[1]
    # cv2.BORDER_REFLECT = 端点重复 => numpy 的 symmetric（不是 reflect）
    xp = np.pad(np.asarray(x, dtype=np.float64), ((r, r), (r, r)), mode="symmetric")
    # 积分图：S[i,j] = sum(xp[0:i, 0:j])
    S = np.zeros((xp.shape[0] + 1, xp.shape[1] + 1), dtype=np.float64)
    S[1:, 1:] = xp.cumsum(0).cumsum(1)
    # 输出 (y,x) 对应 padded 窗口 rows y..y+k-1 / cols x..x+k-1
    win = S[k : k + h, k : k + w] - S[0:h, k : k + w] - S[k : k + h, 0:w] + S[0:h, 0:w]
    return win / float(k * k)


def guided_filter_ref(g, p, r, eps):
    """g: (H,W,C) / p: (H,W,Cs) -> q: (H,W,Cs)，全程 float64。"""
    h, w, C = g.shape
    Cs = p.shape[2]
    gg = np.asarray(g, dtype=np.float64)
    pp = np.asarray(p, dtype=np.float64)

    mu = np.stack([box_mean(gg[..., c], r) for c in range(C)], axis=-1)
    nu = np.stack([box_mean(pp[..., s], r) for s in range(Cs)], axis=-1)

    sig = np.empty((h, w, C, C), dtype=np.float64)
    for c in range(C):
        for d in range(C):
            sig[..., c, d] = (
                box_mean(gg[..., c] * gg[..., d], r) - mu[..., c] * mu[..., d]
            )
    for c in range(C):
        sig[..., c, c] += float(eps)  # 正则项加在对角线上

    omg = np.empty((h, w, C, Cs), dtype=np.float64)
    for c in range(C):
        for s in range(Cs):
            omg[..., c, s] = (
                box_mean(gg[..., c] * pp[..., s], r) - mu[..., c] * nu[..., s]
            )

    alpha = np.linalg.solve(sig, omg)  # (H,W,C,Cs)
    beta = nu - np.einsum("hwc,hwcs->hws", mu, alpha)  # (H,W,Cs)

    q = np.zeros((h, w, Cs), dtype=np.float64)
    for c in range(C):
        for s in range(Cs):
            q[..., s] += box_mean(alpha[..., c, s], r) * gg[..., c]
    for s in range(Cs):
        q[..., s] += box_mean(beta[..., s], r)
    return q


def gen_data_and_golden(shape_str, d_type="float32", radius=4, eps=1e-2, seed=2026):
    np_type = DTYPE_MAP[d_type]
    if len(parse_shape(shape_str)) != 5:
        raise ValueError("shape 必须是 (N,H,W,C,Cs)，实际 %s" % shape_str)
    n, h, w, C, Cs = parse_shape(shape_str)

    # 固定种子的伪随机：可复现，且值域 [-1,1] 覆盖正负、量级不同，索引/通道错位会立刻暴露
    rng = np.random.default_rng(seed)
    guide = rng.uniform(-1.0, 1.0, (n, h, w, C)).astype(np_type)
    src = rng.uniform(-1.0, 1.0, (n, h, w, Cs)).astype(np_type)

    golden = np.empty((n, h, w, Cs), dtype=np_type)
    for b in range(n):
        golden[b] = guided_filter_ref(guide[b], src[b], int(radius), float(eps)).astype(
            np_type
        )

    guide.tofile("%s_input_guided_filter_guide.bin" % d_type)
    src.tofile("%s_input_guided_filter_src.bin" % d_type)
    golden.tofile("%s_golden_guided_filter.bin" % d_type)

    # 顺带打印 golden 的有限性与量级，NaN/Inf 一眼可见（ICPU 模型下曾出现全 NaN）
    finite = np.isfinite(golden).all()
    print(
        "shape_in=(%d,%d,%d,%d,%d) r=%d eps=%g dtype=%s elems=%d "
        "golden[min=%.6f max=%.6f finite=%s]"
        % (
            n,
            h,
            w,
            C,
            Cs,
            radius,
            eps,
            d_type,
            golden.size,
            float(np.nanmin(golden)),
            float(np.nanmax(golden)),
            finite,
        )
    )
    if not finite:
        print("ERROR: golden 含 NaN/Inf，参考实现本身有问题")
        return 1
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Param num must be at least 5: <shape> <dtype> <radius> <eps> [seed]")
        sys.exit(1)
    os.system("rm -f *.bin")
    sd = int(sys.argv[5]) if len(sys.argv) > 5 else 2026
    sys.exit(
        gen_data_and_golden(
            sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sd
        )
    )
