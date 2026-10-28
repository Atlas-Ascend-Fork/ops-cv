#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""rgb2_yuv422 在 kernel/geir 测试路径下的 golden 编写。

Kernel/GEIR 的 golden 收到 numpy.ndarray，转 torch tensor 后用小算子拼接计算，结果转回 numpy；
third_party 的 torch vendor 收到 torch.Tensor，直接调 torch 小算子，无需转换。
无现成框架接口，CSC 系数 + YUYV packing 用 torch 小算子拼接实现。
"""

__spec__ = {"rgb2_yuv422": "Rgb2Yuv422KernelSpec"}

import torch


class Rgb2Yuv422KernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray"""

    def golden(rgb, **kwargs):
        x_t = torch.from_numpy(rgb).to(torch.float32)
        r, g, b = x_t[..., 0], x_t[..., 1], x_t[..., 2]
        y_full = (77.0 / 256.0) * r + (150.0 / 256.0) * g + (29.0 / 256.0) * b
        u_full = (-43.0 / 256.0) * r + (-85.0 / 256.0) * g + (128.0 / 256.0) * b + 128.0
        v_full = (
            (128.0 / 256.0) * r + (-107.0 / 256.0) * g + (-21.0 / 256.0) * b + 128.0
        )
        yuv422 = torch.zeros(*x_t.shape[:-1], 2, dtype=torch.float32)
        yuv422[..., 0::2, 0] = y_full[..., 0::2]
        yuv422[..., 0::2, 1] = u_full[..., 0::2]
        yuv422[..., 1::2, 0] = y_full[..., 1::2]
        yuv422[..., 1::2, 1] = v_full[..., 1::2]
        result = torch.clamp(torch.round(yuv422), 0, 255).to(torch.uint8)
        return [result.numpy()]

    class ThirdPartyImpl:
        def __init__(self, rgb, **kwargs):
            dev = rgb.device
            self.x_t = rgb.to(torch.float32)
            csc = torch.tensor(
                [
                    [77.0 / 256.0, 150.0 / 256.0, 29.0 / 256.0],
                    [-43.0 / 256.0, -85.0 / 256.0, 128.0 / 256.0],
                    [128.0 / 256.0, -107.0 / 256.0, -21.0 / 256.0],
                ],
                dtype=torch.float32,
                device=dev,
            )
            self.csc_t = csc.t()
            self.offset = torch.tensor(
                [0.0, 128.0, 128.0], dtype=torch.float32, device=dev
            )
            w = self.x_t.shape[-2]
            self.is_even = torch.arange(w, device=dev) % 2 == 0

        def __call__(self, rgb, **kwargs):
            yuv = self.x_t @ self.csc_t + self.offset
            chroma = torch.where(self.is_even, yuv[..., 1], yuv[..., 2])
            yuv422 = torch.stack([yuv[..., 0], chroma], dim=-1)
            return [torch.clamp(torch.round(yuv422), 0, 255).to(torch.uint8)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {"uint8": {"standard": "binary_equal"}}
