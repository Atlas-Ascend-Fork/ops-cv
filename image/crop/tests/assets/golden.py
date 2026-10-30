#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

__spec__ = {"crop": "CropTestSpec"}

import numpy as np
import torch


class CropTestSpec:
    """Crop golden — torch.as_strided_copy 单算子实现多维裁剪。"""

    @staticmethod
    def _normalize(x_torch, size_shape, axis, offsets):
        """归一化 axis/offsets，将多维 offsets 折算为一维 storage_offset。"""
        x_torch = x_torch.contiguous()
        rank = len(x_torch.shape)

        if axis < 0:
            axis += rank

        if offsets is None:
            offsets = [0] * rank
        elif len(offsets) == 1:
            offsets = [0] * axis + [offsets[0]] * (rank - axis)
        else:
            offsets = [0] * axis + list(offsets)

        stride = x_torch.stride()
        storage_offset = sum(map(lambda o, s: o * s, offsets, stride))
        return x_torch, size_shape, stride, storage_offset

    @staticmethod
    def _output_shape(x_shape, size_shape, axis):
        rank = len(x_shape)
        if axis < 0:
            axis += rank
        return list(x_shape[:axis]) + list(size_shape[axis:])

    @staticmethod
    def _apply_crop(x_torch, size_shape, axis, offsets):
        """golden 路径：normalize + as_strided_copy + contiguous（转 numpy）。"""
        x_torch, size_shape, stride, storage_offset = CropTestSpec._normalize(
            x_torch, size_shape, axis, offsets
        )
        return torch.as_strided_copy(
            x_torch,
            size=size_shape,
            stride=stride,
            storage_offset=storage_offset,
        ).contiguous()

    @staticmethod
    def golden(x, size, *, axis=2, offsets=None, **kwargs):
        """Crop golden：numpy -> torch -> _apply_crop -> numpy。"""
        x_torch = torch.from_numpy(np.ascontiguousarray(x))
        out_shape = CropTestSpec._output_shape(
            list(x_torch.shape), list(size.shape), axis
        )
        result = CropTestSpec._apply_crop(x_torch, out_shape, axis, offsets)
        return [result.numpy()]

    class _TorchCropImpl:
        """third_party 性能对比实现：__call__ 仅 as_strided_copy，零元数据操作。"""

        def __init__(self, x, size, *, axis=2, offsets=None, **kwargs):
            out_shape = CropTestSpec._output_shape(
                list(x.shape), list(size.shape), axis
            )
            x, self._size, self._stride, self._storage_offset = CropTestSpec._normalize(
                x, out_shape, axis, offsets
            )
            self._x = x

        def __call__(self, **kwargs):
            return [
                torch.as_strided_copy(
                    self._x,
                    size=self._size,
                    stride=self._stride,
                    storage_offset=self._storage_offset,
                )
            ]

    # crop 为纯数据搬运算子，所有 dtype 逐位一致
    tolerance = {
        "float16": {"standard": "binary_equal"},
        "float32": {"standard": "binary_equal"},
        "int8": {"standard": "binary_equal"},
        "int16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "int64": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
        "uint16": {"standard": "binary_equal"},
        "uint32": {"standard": "binary_equal"},
        "uint64": {"standard": "binary_equal"},
    }

    third_party = {"torch": _TorchCropImpl}
