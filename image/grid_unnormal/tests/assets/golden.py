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

from typing import ClassVar

import numpy as np
import torch

__spec__ = {
    "grid_unnormal": "GridUnnormalKernelSpec",
}
__golden__ = {
    "kernel": {"grid_unnormal": "grid_unnormal_golden"},
}

_TOL = {
    "float32": {"standard": "cross_check", "level": "L1"},
    "float16": {"standard": "cross_check", "level": "L1"},
    "int32": {"standard": "binary_equal"},
}


def _normalize_attr_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def _validate_grid_unnormal_inputs(grid, assist):
    """Enforce the spec contract and prevent accidental broadcasting."""
    grid_shape = tuple(grid.shape)
    assist_shape = tuple(assist.shape)
    if len(grid_shape) != 4 or grid_shape[-1] != 2:
        raise ValueError("grid must be rank-4 with last dimension 2")
    if assist_shape != grid_shape:
        raise ValueError("assist must have the same shape as grid")
    if np.dtype(grid.dtype) != np.dtype(assist.dtype):
        raise TypeError("grid and assist must have identical dtype")
    if np.dtype(grid.dtype) not in (
        np.dtype(np.float16),
        np.dtype(np.float32),
        np.dtype(np.float64),
    ):
        raise TypeError("grid and assist dtype must be float16, float32, or float64")


def _grid_unnormal_torch(grid, assist, align_corners):
    ori_dtype = grid.dtype
    torch_dtype = (
        torch.float16 if ori_dtype == np.float16 else torch.from_numpy(grid).dtype
    )
    grid_tensor = torch.from_numpy(grid).to(torch_dtype)
    assist_tensor = torch.from_numpy(assist).to(torch_dtype)
    diff, position = _grid_unnormal_golden_compute(
        grid_tensor, assist_tensor, align_corners
    )
    return [
        diff.cpu().numpy().astype(ori_dtype, copy=False),
        position.cpu().numpy(),
    ]


def _prepare_grid_unnormal_inputs(grid_tensor, assist_tensor):
    out_dtype = grid_tensor.dtype
    compute_dtype = (
        torch.float32
        if grid_tensor.dtype in (torch.float16, torch.bfloat16)
        else grid_tensor.dtype
    )
    grid_compute = grid_tensor.to(compute_dtype)
    assist_compute = assist_tensor.to(compute_dtype)
    return out_dtype, grid_compute, assist_compute


def _grid_unnormal_native_position(grid_tensor, assist_tensor, align_corners):
    grid_compute = grid_tensor.to(torch.float32)
    assist_compute = assist_tensor.to(torch.float32)
    normalized = torch.mul(torch.add(grid_compute, 1.0), 0.5)
    if align_corners:
        pos_base = torch.mul(normalized, torch.sub(assist_compute, 1.0))
    else:
        pos_base = torch.sub(torch.mul(normalized, assist_compute), 0.5)
    return torch.floor(pos_base).to(torch.int32)


def _grid_unnormal_golden_compute(grid_tensor, assist_tensor, align_corners):
    out_dtype, grid_compute, assist_compute = _prepare_grid_unnormal_inputs(
        grid_tensor, assist_tensor
    )
    if align_corners:
        pos_base = (grid_compute + 1.0) * (assist_compute - 1.0) / 2.0
    else:
        pos_base = ((grid_compute + 1.0) * assist_compute - 1.0) / 2.0
    floor = torch.floor(pos_base)
    diff = (pos_base - floor).to(out_dtype)

    # TTK promotes fp32 golden inputs to fp64 for the continuous cross-check
    # output. Position is a discrete output whose contract uses the native
    # fp16/fp32 operator domain (FP32 arithmetic) before floor. Computing floor
    # in promoted fp64 changes the adjacent integer at rounding boundaries.
    position = _grid_unnormal_native_position(grid_tensor, assist_tensor, align_corners)
    return [diff, position]


def _grid_unnormal_third_party_compute(grid_tensor, assist_tensor, align_corners):
    # Keep the GPU benchmark implementation independent from the CPU golden.
    out_dtype = grid_tensor.dtype
    compute_dtype = (
        torch.float32
        if grid_tensor.dtype in (torch.float16, torch.bfloat16)
        else grid_tensor.dtype
    )
    grid_compute = grid_tensor.to(compute_dtype)
    assist_compute = assist_tensor.to(compute_dtype)
    normalized = (grid_compute + 1.0) * 0.5
    if align_corners:
        pos_base = normalized * (assist_compute - 1.0)
    else:
        pos_base = normalized * assist_compute - 0.5
    floor = torch.floor(pos_base)
    position = floor.to(torch.int32)
    diff = (pos_base - floor).to(out_dtype)
    return [diff, position]


class _GridUnnormalCompose:
    def __init__(self, align_corners=False, **kwargs):
        self.align_corners = _normalize_attr_bool(align_corners)

    def __call__(self, grid, assist, **kwargs):
        grid_tensor = (
            grid
            if isinstance(grid, torch.Tensor)
            else torch.from_numpy(np.asarray(grid))
        )
        assist_tensor = (
            assist
            if isinstance(assist, torch.Tensor)
            else torch.from_numpy(np.asarray(assist))
        )
        if grid_tensor.ndim != 4 or grid_tensor.shape[-1] != 2:
            raise ValueError("grid must be rank-4 with last dimension 2")
        if tuple(assist_tensor.shape) != tuple(grid_tensor.shape):
            raise ValueError("assist must have the same shape as grid")
        if assist_tensor.dtype != grid_tensor.dtype:
            raise TypeError("grid and assist must have identical dtype")
        if grid_tensor.dtype not in (torch.float16, torch.float32):
            raise TypeError("grid and assist dtype must be float16 or float32")
        return _grid_unnormal_third_party_compute(
            grid_tensor, assist_tensor, self.align_corners
        )


class _GridUnnormalTfCompose:
    def __init__(self, align_corners=False, **kwargs):
        self.align_corners = _normalize_attr_bool(align_corners)

    def __call__(self, grid, assist, **kwargs):
        import tensorflow as tf

        grid_tensor = tf.convert_to_tensor(grid)
        assist_tensor = tf.convert_to_tensor(assist)
        if grid_tensor.shape.rank != 4 or grid_tensor.shape[-1] != 2:
            raise ValueError("grid must be rank-4 with last dimension 2")
        if assist_tensor.shape != grid_tensor.shape:
            raise ValueError("assist must have the same shape as grid")
        if assist_tensor.dtype != grid_tensor.dtype:
            raise TypeError("grid and assist must have identical dtype")
        out_dtype = grid_tensor.dtype
        if out_dtype not in (tf.float16, tf.float32):
            raise TypeError("grid and assist dtype must be float16 or float32")
        compute_dtype = (
            tf.float32 if out_dtype in (tf.float16, tf.bfloat16) else out_dtype
        )
        grid_compute = tf.cast(grid_tensor, compute_dtype)
        assist_compute = tf.cast(assist_tensor, compute_dtype)
        normalized = (grid_compute + 1.0) * 0.5
        if self.align_corners:
            pos_base = normalized * (assist_compute - 1.0)
        else:
            pos_base = normalized * assist_compute - 0.5
        floor = tf.floor(pos_base)
        return [tf.cast(pos_base - floor, out_dtype), tf.cast(floor, tf.int32)]


class GridUnnormalKernelSpec:
    @staticmethod
    def golden(grid, assist, *, align_corners=False, **kwargs):
        grid_array = np.asarray(grid)
        assist_array = np.asarray(assist)
        _validate_grid_unnormal_inputs(grid_array, assist_array)
        return _grid_unnormal_torch(
            grid_array, assist_array, _normalize_attr_bool(align_corners)
        )

    third_party: ClassVar = {
        "torch": _GridUnnormalCompose,
        "tf": _GridUnnormalTfCompose,
    }
    tolerance = _TOL


def grid_unnormal_golden(grid, assist, align_corners=False, **kwargs):
    return GridUnnormalKernelSpec.golden(
        grid, assist, align_corners=align_corners, **kwargs
    )


# Not registered in __spec__:
# - aclnn/e2e: OpDef is aclnn_exclude and no torch_npu binding is delivered.
# - TensorFlow/ONNX/fusion: no parser or graph pass is delivered for this op.
