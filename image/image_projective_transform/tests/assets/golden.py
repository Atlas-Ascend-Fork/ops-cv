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

"""TensorFlow V2 TestSpec for image_projective_transform Kernel/GEIR tests.

Golden main path: ``tf.raw_ops.ImageProjectiveTransformV2`` (3 inputs, no fill_value),
which natively covers the v2.0 enhancement scope (SE §7):
- fill_mode four modes CONSTANT/REFLECT/WRAP/NEAREST (NEAREST is undocumented in TF
  but supported by the TF 2.21.0 kernel, matching SE §5.2-A formulas point-by-point);
- one_trans broadcast: transforms ``[1, 8]`` is broadcast to every batch natively
  (bit-equivalent to explicitly tiling it to ``[N, 8]``, SE §5.2-B.4).
No numpy computation is performed; numpy is only the container boundary.
"""

import numpy as np
import tensorflow as tf

__spec__ = {
    "image_projective_transform": "ImageProjectiveTransformTestSpec",
    "ImageProjectiveTransform": "ImageProjectiveTransformTestSpec",
}


def _output_hw(output_shape):
    """Read [HOut, WOut] from the output_shape input container."""
    shape_hw = np.asarray(output_shape).reshape(-1)
    h_out = int(shape_hw[0]) if shape_hw.size > 0 else 0
    w_out = int(shape_hw[1]) if shape_hw.size > 1 else 0
    return h_out, w_out


def _is_empty_output(output_shape):
    h_out, w_out = _output_hw(output_shape)
    return h_out == 0 or w_out == 0


def _empty_output_tensor(images, output_shape):
    """Build the README-declared empty output (output_shape element == 0).

    README states that a zero element in output_shape yields an empty tensor and the
    operator returns without computing. TF 2.21 raises InvalidArgumentError for
    H/W == 0, so this legal empty output is produced with the framework's own
    ``tf.zeros`` (not a NumPy array) using the input dtype.
    """
    images_t = tf.convert_to_tensor(images)
    h_out, w_out = _output_hw(output_shape)
    rank = len(images_t.shape)
    n = int(images_t.shape[0]) if rank >= 1 else 0
    c = int(images_t.shape[3]) if rank >= 4 else 0
    return tf.zeros((n, h_out, w_out, c), dtype=images_t.dtype)


def _tf_call(images, transforms, output_shape, *, interpolation, fill_mode):
    """Call the TensorFlow V2 reference API with the actual OpDef signature."""
    return tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.convert_to_tensor(images),
        transforms=tf.cast(tf.convert_to_tensor(transforms), tf.float32),
        output_shape=tf.cast(tf.convert_to_tensor(output_shape), tf.int32)[:2],
        interpolation=interpolation,
        fill_mode=fill_mode,
    )


class ImageProjectiveTransformTestSpec:
    """Golden receives numpy arrays; the tf provider receives tf.Tensor inputs."""

    def golden(
        images,
        transforms,
        output_shape,
        *,
        interpolation,
        fill_mode="CONSTANT",
        **kwargs,
    ):
        if _is_empty_output(output_shape):
            return [_empty_output_tensor(images, output_shape).numpy()]
        result = _tf_call(
            images,
            transforms,
            output_shape,
            interpolation=interpolation,
            fill_mode=fill_mode,
        )
        return [result.numpy()]

    class TfThirdPartyImpl:
        def __init__(self, *, interpolation, fill_mode="CONSTANT", **kwargs):
            self.interpolation = interpolation
            self.fill_mode = fill_mode

        def __call__(self, images, transforms, output_shape, **kwargs):
            if _is_empty_output(output_shape):
                return [_empty_output_tensor(images, output_shape)]
            return [
                _tf_call(
                    images,
                    transforms,
                    output_shape,
                    interpolation=self.interpolation,
                    fill_mode=self.fill_mode,
                )
            ]

    third_party = {"tf": TfThirdPartyImpl}

    # 比较标准说明：cross_check 需要第三方（TensorFlow XPU）参考并要求可用的 XPU/GPU endpoint。
    # 目标验证环境为 Ascend NPU、无 CPU 之外的 XPU provider，cross_check 只能得到 GOLDEN_FAILURE，
    # 无法建立任何数值结论。因此声明 TTK 官方标准 mix_tolerance（仅需本交付 CPU golden）完成浮点数值
    # 比较；third_party 继续交付并在具备可用 XPU provider 时可切回 cross_check。
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
        "uint8": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
    }
