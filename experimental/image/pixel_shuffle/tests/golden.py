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


__golden__ = {"kernel": {"pixel_shuffle": "pixel_shuffle_golden"}}


def pixel_shuffle_golden(x, **kwargs):
    """PixelShuffle 参考实现（numpy），支持 4D/5D 亚像素重排。

    - 4D: (N, C, H, W) -> (N, C/r^2, H*r, W*r)
    - 5D: (N, C, D, H, W) -> (N, C/r^3, D*r, H*r, W*r)
    """
    r = int(kwargs.get("scale_factor", 2))
    input_dtype = x.dtype
    dim = x.ndim
    if dim == 4:
        n, c, h, w = x.shape
        c_out = c // (r * r)
        out = x.reshape(n, c_out, r, r, h, w)
        out = out.transpose(0, 1, 4, 2, 5, 3)
        out = out.reshape(n, c_out, h * r, w * r)
    elif dim == 5:
        n, c, d, h, w = x.shape
        c_out = c // (r * r * r)
        out = x.reshape(n, c_out, r, r, r, d, h, w)
        out = out.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        out = out.reshape(n, c_out, d * r, h * r, w * r)
    else:
        raise ValueError(
            "pixel_shuffle only supports 4D or 5D input, got rank %d" % dim
        )
    return out.astype(input_dtype)
