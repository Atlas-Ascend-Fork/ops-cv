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

import numpy as np
import torch

__spec__ = {
    "bounding_box_decode": "BoundingBoxDecodeTestSpec",
    "BoundingBoxDecode": "BoundingBoxDecodeTestSpec",
}


class BoundingBoxDecodeTestSpec:
    """bounding_box_decode 算子测试规范（kernel 流程）

    golden 和 third_party 均用 torch 算子拼接实现，供 cross_check 交叉比对
    """

    def golden(
        anchor_box,
        deltas,
        *,
        max_shape,
        means=(0.0, 0.0, 0.0, 0.0),
        stds=(1.0, 1.0, 1.0, 1.0),
        wh_ratio_clip=0.016,
        **kwargs,
    ):
        x_dtype = anchor_box.dtype

        means = [float(v) for v in means]
        stds = [float(v) for v in stds]
        max_h = float(max_shape[0])
        max_w = float(max_shape[1])
        # 裁剪到 [0, W-1]/[0, H-1]（最后有效像素），与 CANN 内置版及 CheckValid 契约一致
        clip_h = max_h - 1.0
        clip_w = max_w - 1.0

        a = torch.from_numpy(anchor_box.astype(np.float32))
        d = torch.from_numpy(deltas.astype(np.float32))

        a_x1, a_y1, a_x2, a_y2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        dx_in, dy_in, dw_in, dh_in = d[:, 0], d[:, 1], d[:, 2], d[:, 3]

        pw = a_x2 - a_x1 + 1.0
        ph = a_y2 - a_y1 + 1.0
        pcx = (a_x1 + a_x2) * 0.5
        pcy = (a_y1 + a_y2) * 0.5

        dx = dx_in * stds[0] + means[0]
        dy = dy_in * stds[1] + means[1]
        dw = dw_in * stds[2] + means[2]
        dh = dh_in * stds[3] + means[3]

        # A2 基线 canndev tbe clamp_denorm_detal：反标准化后 dw/dh 裁剪到
        # ±|ln(wh_ratio_clip)| 再进入 exp（默认 0.016 → ±4.1352）。
        # torch.clamp 对 NaN 原样传播，与 kernel 的 NaN 处理路径一致。
        max_ratio = abs(float(np.log(float(wh_ratio_clip))))
        dw = torch.clamp(dw, -max_ratio, max_ratio)
        dh = torch.clamp(dh, -max_ratio, max_ratio)

        gw = pw * torch.exp(dw)
        gh = ph * torch.exp(dh)
        gx = pcx + pw * dx
        gy = pcy + ph * dy

        ox1 = gx - gw * 0.5 + 0.5
        oy1 = gy - gh * 0.5 + 0.5
        ox2 = gx + gw * 0.5 - 0.5
        oy2 = gy + gh * 0.5 - 0.5

        ox1 = torch.nan_to_num(ox1, nan=0.0, posinf=clip_w, neginf=0.0)
        oy1 = torch.nan_to_num(oy1, nan=0.0, posinf=clip_h, neginf=0.0)
        ox2 = torch.nan_to_num(ox2, nan=0.0, posinf=clip_w, neginf=0.0)
        oy2 = torch.nan_to_num(oy2, nan=0.0, posinf=clip_h, neginf=0.0)
        ox1 = torch.clamp(ox1, 0.0, clip_w)
        oy1 = torch.clamp(oy1, 0.0, clip_h)
        ox2 = torch.clamp(ox2, 0.0, clip_w)
        oy2 = torch.clamp(oy2, 0.0, clip_h)

        boxes = torch.stack([ox1, oy1, ox2, oy2], dim=1)
        return [boxes.numpy().astype(x_dtype)]

    class TorchImpl:
        """torch 算子拼接实现：用 torch.exp + torch.clamp + torch.nan_to_num 拼接解码公式。"""

        def __init__(
            self,
            *,
            max_shape,
            means=(0.0, 0.0, 0.0, 0.0),
            stds=(1.0, 1.0, 1.0, 1.0),
            wh_ratio_clip=0.016,
            **kwargs,
        ):
            self.max_h = float(max_shape[0])
            self.max_w = float(max_shape[1])
            self.clip_h = self.max_h - 1.0
            self.clip_w = self.max_w - 1.0
            self.means = [float(v) for v in means]
            self.stds = [float(v) for v in stds]
            # A2 基线 clamp_denorm_detal 阈值：|ln(wh_ratio_clip)|
            self.wh_max_ratio = abs(float(np.log(float(wh_ratio_clip))))

        def __call__(self, anchor_box, deltas, **kwargs):
            a = anchor_box.to(torch.float32)
            d = deltas.to(torch.float32)

            a_x1, a_y1, a_x2, a_y2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
            dx_in, dy_in, dw_in, dh_in = d[:, 0], d[:, 1], d[:, 2], d[:, 3]

            pw = a_x2 - a_x1 + 1.0
            ph = a_y2 - a_y1 + 1.0
            pcx = (a_x1 + a_x2) * 0.5
            pcy = (a_y1 + a_y2) * 0.5

            dx = dx_in * self.stds[0] + self.means[0]
            dy = dy_in * self.stds[1] + self.means[1]
            dw = dw_in * self.stds[2] + self.means[2]
            dh = dh_in * self.stds[3] + self.means[3]

            # A2 基线 clamp_denorm_detal：dw/dh 裁剪到 ±|ln(wh_ratio_clip)|
            dw = torch.clamp(dw, -self.wh_max_ratio, self.wh_max_ratio)
            dh = torch.clamp(dh, -self.wh_max_ratio, self.wh_max_ratio)

            gw = pw * torch.exp(dw)
            gh = ph * torch.exp(dh)
            gx = pcx + pw * dx
            gy = pcy + ph * dy

            ox1 = gx - gw * 0.5 + 0.5
            oy1 = gy - gh * 0.5 + 0.5
            ox2 = gx + gw * 0.5 - 0.5
            oy2 = gy + gh * 0.5 - 0.5

            ox1 = torch.nan_to_num(ox1, nan=0.0, posinf=self.clip_w, neginf=0.0)
            oy1 = torch.nan_to_num(oy1, nan=0.0, posinf=self.clip_h, neginf=0.0)
            ox2 = torch.nan_to_num(ox2, nan=0.0, posinf=self.clip_w, neginf=0.0)
            oy2 = torch.nan_to_num(oy2, nan=0.0, posinf=self.clip_h, neginf=0.0)
            ox1 = torch.clamp(ox1, 0.0, self.clip_w)
            oy1 = torch.clamp(oy1, 0.0, self.clip_h)
            ox2 = torch.clamp(ox2, 0.0, self.clip_w)
            oy2 = torch.clamp(oy2, 0.0, self.clip_h)

            boxes = torch.stack([ox1, oy1, ox2, oy2], dim=1)
            return [boxes.to(anchor_box.dtype)]

    third_party = {
        "torch": TorchImpl,
    }

    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
    }
