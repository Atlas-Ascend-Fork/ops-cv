#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""BoundingBoxEncode 在 Kernel/GEIR 测试路径下的 golden 编写。

Kernel/GEIR 的 golden 收到 numpy.ndarray，先转 torch.Tensor，再用 torch 小算子
拼接计算编码偏移，最后转回 numpy；third_party 的 torch vendor 收到 torch.Tensor，
复用同一套 torch 计算。禁止裸 numpy 计算。

公式（与 CANN kernel 的 Caffe/MMDetection-v1 约定一致）：
    pw = x2 - x1 + 1, ph = y2 - y1 + 1
    pcx = (x1 + x2) / 2, pcy = (y1 + y2) / 2
    raw = [(gcx - pcx) / pw, (gcy - pcy) / ph, ln(gw / pw), ln(gh / ph)]
    delta = (raw - means) / stds
"""

__spec__ = {
    # Kernel / GEIR 共用；key 必须与 CSV op_name 完全一致。
    "BoundingBoxEncode": "BoundingBoxEncodeKernelSpec",
    "bounding_box_encode": "BoundingBoxEncodeKernelSpec",
}

import numpy as np
import torch

_MEANS = (0.0, 0.0, 0.0, 0.0)
_STDS = (1.0, 1.0, 1.0, 1.0)


def _is_bf16_dtype(dtype):
    if getattr(dtype, "name", str(dtype)) == "bfloat16":
        return True
    return getattr(dtype, "kind", None) == "V" and getattr(dtype, "itemsize", None) == 2


def _numpy_to_torch(array):
    if _is_bf16_dtype(array.dtype):
        return torch.from_numpy(np.ascontiguousarray(array).view(np.int16)).view(
            torch.bfloat16
        )
    return torch.from_numpy(np.ascontiguousarray(array))


def _torch_to_numpy(tensor, dtype):
    tensor = tensor.detach().cpu()
    if _is_bf16_dtype(dtype):
        return (
            tensor.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(dtype)
        )
    return tensor.numpy().astype(dtype, copy=False)


def _torch_encode(anchor, gt, m, s):
    """纯 torch 计算，golden 与 third_party 共用。"""
    a = anchor
    g = gt

    pw = a[..., 2] - a[..., 0] + 1.0
    ph = a[..., 3] - a[..., 1] + 1.0
    pcx = (a[..., 0] + a[..., 2]) * 0.5
    pcy = (a[..., 1] + a[..., 3]) * 0.5

    gw = g[..., 2] - g[..., 0] + 1.0
    gh = g[..., 3] - g[..., 1] + 1.0
    gcx = (g[..., 0] + g[..., 2]) * 0.5
    gcy = (g[..., 1] + g[..., 3]) * 0.5

    raw = torch.stack(
        [
            (gcx - pcx) / pw,
            (gcy - pcy) / ph,
            torch.log(gw / pw),
            torch.log(gh / ph),
        ],
        dim=-1,
    )
    return (raw - m) / s


class BoundingBoxEncodeKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray，third_party 收到 torch.Tensor。"""

    def golden(anchor_box, ground_truth_box, *, means=_MEANS, stds=_STDS, **kwargs):
        anchor = _numpy_to_torch(anchor_box)
        gt = _numpy_to_torch(ground_truth_box)
        m = torch.as_tensor(means, dtype=anchor.dtype, device=anchor.device)
        s = torch.as_tensor(stds, dtype=anchor.dtype, device=anchor.device)
        out = _torch_encode(anchor, gt, m, s)
        return [_torch_to_numpy(out, anchor_box.dtype)]

    class ThirdPartyImpl:
        """torch vendor 小算子拼接；__init__ 预处理属性，__call__ 只做计算。"""

        def __init__(self, anchor_box, *, means=_MEANS, stds=_STDS, **kwargs):
            self.dtype = anchor_box.dtype
            self.m = torch.as_tensor(means, dtype=self.dtype, device=anchor_box.device)
            self.s = torch.as_tensor(stds, dtype=self.dtype, device=anchor_box.device)

        def __call__(self, anchor_box, ground_truth_box, **kwargs):
            return [_torch_encode(anchor_box, ground_truth_box, self.m, self.s)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
        "bfloat16": {"standard": "cross_check", "level": "L1"},
    }
