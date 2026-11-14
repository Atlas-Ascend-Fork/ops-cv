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

"""crop_and_resize 在 Kernel/GEIR 测试路径下的 golden。

语义：
    - x 为 ND/NHWC 时按 (N, H, W, C) 解析，y 为 (num_boxes, crop_h, crop_w, C)
    - x 为 NCHW 时按 (N, C, H, W) 解析，y 为 (num_boxes, C, crop_h, crop_w)
    - method 支持 bilinear / nearest
    - boxes 非有限行先置 0，计算后整行覆盖为 extrapolation_value
    - 输出 dtype 由 boxes dtype 决定：FP16 -> FP16，FP32/其他 -> FP32
"""

__spec__ = {
    # key 必须与 CSV op_name 完全一致。
    "crop_and_resize": "CropAndResizeKernelSpec",
}

import numpy as np
import tensorflow as tf

_SUPPORTED_METHODS = ("bilinear", "nearest")


def _validate_method(method):
    if method not in _SUPPORTED_METHODS:
        raise ValueError(
            f"Unsupported method: {method}, supported methods are {_SUPPORTED_METHODS}"
        )


def _is_nchw_input(input_formats):
    """Return True iff the x input format (first entry) is NCHW."""
    if not input_formats:
        return False
    if isinstance(input_formats, str):
        x_format = input_formats
    else:
        try:
            x_format = input_formats[0]
        except (TypeError, IndexError, KeyError):
            return False
    return str(x_format).strip().upper() == "NCHW"


def _split_nonfinite_boxes(boxes):
    boxes_np = np.asarray(boxes)
    bad = ~np.isfinite(boxes_np).all(axis=1)
    boxes_for_tf = boxes_np.astype(np.float32, copy=True)
    if bad.any():
        boxes_for_tf[bad] = 0.0
    return boxes_for_tf, bad


def _to_tensor(value, dtype, like=None):
    """按 like 的 device 创建 tensor；like 为 None 时保持默认 CPU。"""
    device = getattr(like, "device", None) if like is not None else None
    if device:
        with tf.device(device):
            return tf.convert_to_tensor(value, dtype=dtype)
    return tf.convert_to_tensor(value, dtype=dtype)


def _prepare_inputs(x, boxes, box_index, kwargs, preserve_device=False):
    is_nchw = _is_nchw_input(kwargs.get("input_formats"))
    boxes_tf_dtype = tf.dtypes.as_dtype(boxes.dtype)
    out_dtype = (
        boxes_tf_dtype if boxes_tf_dtype in (tf.float16, tf.float32) else tf.float32
    )

    if is_nchw:
        x_tf = tf.cast(tf.transpose(x, (0, 2, 3, 1)), tf.float32)
    else:
        x_tf = tf.cast(x, tf.float32)

    boxes_np = boxes.numpy() if hasattr(boxes, "numpy") else np.asarray(boxes)
    boxes_for_tf, bad_boxes = _split_nonfinite_boxes(boxes_np)
    like = boxes if preserve_device else None
    boxes_tf = _to_tensor(boxes_for_tf, tf.float32, like)
    box_index_tf = _to_tensor(box_index, tf.int32, like)
    return x_tf, boxes_tf, box_index_tf, out_dtype, is_nchw, bad_boxes


def _run_crop_and_resize(
    x_tf,
    boxes_tf,
    box_index_tf,
    crop_size,
    out_dtype,
    is_nchw,
    bad_boxes,
    method,
    extrapolation_value,
):
    crop_size_list = [int(crop_size[0]), int(crop_size[1])]
    extrapolation_value_f = float(extrapolation_value)
    y_tf = tf.image.crop_and_resize(
        x_tf,
        boxes_tf,
        box_index_tf,
        crop_size_list,
        method=method,
        extrapolation_value=extrapolation_value_f,
    )
    y = np.array(y_tf, copy=True)
    if bad_boxes.any():
        y[bad_boxes] = np.float32(extrapolation_value_f)
    if is_nchw:
        y = np.transpose(y, (0, 3, 1, 2))
    return y.astype(out_dtype.as_numpy_dtype, copy=False)


class CropAndResizeKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray。"""

    @staticmethod
    def golden(
        x,
        boxes,
        box_index,
        crop_size,
        *,
        extrapolation_value=0.0,
        method="bilinear",
        **kwargs,
    ):
        _validate_method(method)
        x_tf, boxes_tf, box_index_tf, out_dtype, is_nchw, bad_boxes = _prepare_inputs(
            x, boxes, box_index, kwargs
        )
        y = _run_crop_and_resize(
            x_tf,
            boxes_tf,
            box_index_tf,
            crop_size,
            out_dtype,
            is_nchw,
            bad_boxes,
            method,
            extrapolation_value,
        )
        return [y]

    class ThirdPartyImpl:
        """TF vendor 小算子拼接；__init__ 预处理，__call__ 只做计算。"""

        def __init__(
            self,
            *,
            extrapolation_value=0.0,
            method="bilinear",
            x=None,
            boxes=None,
            box_index=None,
            crop_size=None,
            **kwargs,
        ):
            _validate_method(method)
            self.method = method
            self.crop_size = crop_size
            self.extrapolation_value = extrapolation_value
            (
                self.x_tf,
                self.boxes_tf,
                self.box_index_tf,
                self.out_dtype,
                self.is_nchw,
                self.bad_boxes,
            ) = _prepare_inputs(x, boxes, box_index, kwargs, preserve_device=True)

        def __call__(
            self, x=None, boxes=None, box_index=None, crop_size=None, **kwargs
        ):
            y = _run_crop_and_resize(
                self.x_tf,
                self.boxes_tf,
                self.box_index_tf,
                self.crop_size,
                self.out_dtype,
                self.is_nchw,
                self.bad_boxes,
                self.method,
                self.extrapolation_value,
            )
            return [y]

    third_party = {
        "tf": ThirdPartyImpl,
    }

    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
    }
