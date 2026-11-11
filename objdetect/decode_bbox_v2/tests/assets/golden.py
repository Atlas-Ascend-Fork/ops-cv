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
    "DecodeBboxV2": "DecodeBboxV2TestSpec",
    "decode_bbox_v2": "DecodeBboxV2TestSpec",
}


class DecodeBboxV2TestSpec:
    """decode_bbox_v2 算子测试规范（kernel 流程）

    golden 和 third_party 均用 torch 算子拼接实现，供 cross_check 交叉比对
    """

    def golden(
        boxes,
        anchors,
        *,
        scales=None,
        decode_clip=0.0,
        reversed_box=False,
        **kwargs,
    ):
        scales = scales if scales is not None else [1.0, 1.0, 1.0, 1.0]
        sy, sx, sh, sw = [float(s) for s in scales]

        out_dtype = torch.from_numpy(boxes).dtype

        b_in = torch.from_numpy(boxes.astype(np.float32))
        a_in = torch.from_numpy(anchors.astype(np.float32))

        if not reversed_box:
            b = b_in.transpose(0, 1)
            a = a_in.transpose(0, 1)
        else:
            b = b_in
            a = a_in

        ty, tx, th, tw = b[0], b[1], b[2], b[3]
        aymin, axmin, aymax, axmax = a[0], a[1], a[2], a[3]

        ah = aymax - aymin
        aw = axmax - axmin

        tys = ty / sy
        txs = tx / sx
        ths = th / sh
        tws = tw / sw

        if decode_clip != 0:
            clip_val = torch.tensor(float(decode_clip), dtype=torch.float32)
            h = torch.exp(torch.minimum(ths, clip_val)) * ah
            w = torch.exp(torch.minimum(tws, clip_val)) * aw
        else:
            h = torch.exp(ths) * ah
            w = torch.exp(tws) * aw

        cy = tys * ah + aymin + ah / 2
        cx = txs * aw + axmin + aw / 2

        ymin = cy - h / 2
        xmin = cx - w / 2
        ymax = cy + h / 2
        xmax = cx + w / 2

        y4 = torch.stack([ymin, xmin, ymax, xmax], dim=0)

        if not reversed_box:
            y = y4.transpose(0, 1)
        else:
            y = y4

        return [y.to(out_dtype).numpy()]

    class TorchImpl:
        """torch 算子拼接实现：用 torch.exp + torch.minimum 拼接解码公式。"""

        def __init__(
            self,
            *,
            scales=None,
            decode_clip=0.0,
            reversed_box=False,
            **kwargs,
        ):
            self.scales = scales if scales is not None else [1.0, 1.0, 1.0, 1.0]
            self.decode_clip = float(decode_clip)
            self.reversed_box = reversed_box

        def __call__(self, boxes, anchors, **kwargs):
            sy, sx, sh, sw = [float(s) for s in self.scales]

            b_in = boxes.to(torch.float32)
            a_in = anchors.to(torch.float32)

            if not self.reversed_box:
                b = b_in.transpose(0, 1)
                a = a_in.transpose(0, 1)
            else:
                b = b_in
                a = a_in

            ty, tx, th, tw = b[0], b[1], b[2], b[3]
            aymin, axmin, aymax, axmax = a[0], a[1], a[2], a[3]

            ah = aymax - aymin
            aw = axmax - axmin

            tys = ty / sy
            txs = tx / sx
            ths = th / sh
            tws = tw / sw

            if self.decode_clip != 0:
                clip_val = torch.scalar_tensor(
                    self.decode_clip, dtype=torch.float32, device=ths.device
                )
                h = torch.exp(torch.minimum(ths, clip_val)) * ah
                w = torch.exp(torch.minimum(tws, clip_val)) * aw
            else:
                h = torch.exp(ths) * ah
                w = torch.exp(tws) * aw

            cy = tys * ah + aymin + ah / 2
            cx = txs * aw + axmin + aw / 2

            ymin = cy - h / 2
            xmin = cx - w / 2
            ymax = cy + h / 2
            xmax = cx + w / 2

            y4 = torch.stack([ymin, xmin, ymax, xmax], dim=0)

            if not self.reversed_box:
                y = y4.transpose(0, 1)
            else:
                y = y4

            return [y.to(boxes.dtype)]

    third_party = {
        "torch": TorchImpl,
    }

    tolerance = {
        "float32": {"standard": "mix_tolerance", "rtol": 1.0e-05, "atol": 1.0e-05},
        "float16": {"standard": "mix_tolerance", "rtol": 1.0e-03, "atol": 1.0e-03},
    }
