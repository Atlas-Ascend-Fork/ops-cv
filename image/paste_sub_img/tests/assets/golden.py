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
import math

import numpy
import torch

__spec__ = {"paste_sub_img": "PasteSubImgTestSpec"}


class PasteSubImgTestSpec:
    @staticmethod
    def golden(
        patch_img, patch_coord, core_area_coord, combine_img, *, scale=1.0, **kwargs
    ):
        is_uint8 = patch_img.dtype == numpy.uint8

        patch = torch.from_numpy(patch_img)
        result = torch.from_numpy(combine_img).clone()
        if is_uint8:
            patch = patch.to(torch.float16)
            result = result.to(torch.float16)

        px1, py1 = int(patch_coord[0]), int(patch_coord[1])
        cx1, cy1, cx2, cy2 = (
            int(core_area_coord[0]),
            int(core_area_coord[1]),
            int(core_area_coord[2]),
            int(core_area_coord[3]),
        )

        s_cy1 = math.floor(cy1 * scale)
        s_cy2 = math.floor(cy2 * scale)
        s_cx1 = math.floor(cx1 * scale)
        s_cx2 = math.floor(cx2 * scale)
        d_cy1 = math.floor((cy1 + py1) * scale)
        d_cx1 = math.floor((cx1 + px1) * scale)

        dh = s_cy2 - s_cy1
        dw = s_cx2 - s_cx1

        result[d_cy1 : d_cy1 + dh, d_cx1 : d_cx1 + dw, :] += patch[
            s_cy1:s_cy2, s_cx1:s_cx2, :
        ]

        if is_uint8:
            result = result.clamp(0, 255).to(torch.uint8)
        return [result.numpy()]

    class ThirdPartyImpl:
        def __init__(self, *, scale=1.0, **kwargs):
            self.scale = scale

        def __call__(
            self, patch_img, patch_coord, core_area_coord, combine_img, **kwargs
        ):
            is_uint8 = patch_img.dtype == torch.uint8

            result = combine_img.clone()
            if is_uint8:
                patch = patch_img.to(torch.float16)
                result = result.to(torch.float16)
            else:
                patch = patch_img

            px1, py1 = int(patch_coord[0]), int(patch_coord[1])
            cx1, cy1, cx2, cy2 = (
                int(core_area_coord[0]),
                int(core_area_coord[1]),
                int(core_area_coord[2]),
                int(core_area_coord[3]),
            )

            s_cy1 = math.floor(cy1 * self.scale)
            s_cy2 = math.floor(cy2 * self.scale)
            s_cx1 = math.floor(cx1 * self.scale)
            s_cx2 = math.floor(cx2 * self.scale)
            d_cy1 = math.floor((cy1 + py1) * self.scale)
            d_cx1 = math.floor((cx1 + px1) * self.scale)

            dh = s_cy2 - s_cy1
            dw = s_cx2 - s_cx1

            result[d_cy1 : d_cy1 + dh, d_cx1 : d_cx1 + dw, :] += patch[
                s_cy1:s_cy2, s_cx1:s_cx2, :
            ]

            if is_uint8:
                result = result.clamp(0, 255).to(torch.uint8)
            return [result]

    third_party = {
        "torch": ThirdPartyImpl,
    }

    tolerance = {
        "uint8": {"standard": "binary_equal"},
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }
