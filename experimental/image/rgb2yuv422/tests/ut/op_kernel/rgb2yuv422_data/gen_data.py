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

import sys
import os
import numpy as np


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip("(").strip(")")
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list)


def golden_rgb2yuv422(x, data_format="NHWC"):
    input_dtype = x.dtype
    x_f64 = np.asarray(x, dtype=np.float64)

    if data_format not in ("NHWC", "NCHW"):
        raise ValueError(f"Unsupported data_format: {data_format}")
    channel_axis = -1 if data_format == "NHWC" else -3
    if x_f64.shape[channel_axis] != 3:
        raise ValueError(
            f"Input channel dimension must be 3, got {x_f64.shape[channel_axis]}"
        )

    x_hwc = np.moveaxis(x_f64, channel_axis, -1)
    orig_shape = x_hwc.shape
    w = orig_shape[-2]

    r = x_hwc[..., 0]
    g = x_hwc[..., 1]
    b = x_hwc[..., 2]

    if input_dtype == np.uint8:
        y_full = 0.29900 * r + 0.58700 * g + 0.11400 * b
        u_full = -0.16874 * r - 0.33126 * g + 0.50000 * b + 128.0
        v_full = 0.50000 * r - 0.41869 * g - 0.08131 * b + 128.0
    else:
        y_full = 0.29900 * r + 0.58700 * g + 0.11400 * b
        u_full = -0.16874 * r - 0.33126 * g + 0.50000 * b
        v_full = 0.50000 * r - 0.41869 * g - 0.08131 * b

    yuv422 = np.zeros(list(orig_shape[:-1]) + [2], dtype=np.float64)

    for j in range(0, w, 2):
        yuv422[..., j, 0] = y_full[..., j]
        yuv422[..., j, 1] = u_full[..., j]

        if j + 1 < w:
            yuv422[..., j + 1, 0] = y_full[..., j + 1]
            yuv422[..., j + 1, 1] = v_full[..., j]

    if data_format == "NCHW":
        yuv422 = np.moveaxis(yuv422, -1, -3)

    if input_dtype == np.uint8:
        yuv422 = np.clip(np.round(yuv422), 0, 255).astype(np.uint8)
    elif input_dtype == np.float16:
        yuv422 = yuv422.astype(np.float16)
    else:
        yuv422 = yuv422.astype(np.float32)

    return yuv422


def gen_data_and_golden(shape_str, d_type="float32"):
    d_type_dict = {"float32": np.float32, "float16": np.float16, "uint8": np.uint8}
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)
    shape_tuple = tuple(int(s) for s in shape)
    size = np.prod(shape)

    if d_type == "uint8":
        tmp_input = np.random.randint(0, 256, size=size, dtype=np.uint8)
    else:
        tmp_input = np.random.uniform(0.0, 1.0, size=size).astype(np_type)
    tmp_input = tmp_input.reshape(shape_tuple)

    tmp_golden = golden_rgb2yuv422(tmp_input, "NHWC")
    tmp_golden = tmp_golden.astype(np_type)

    tmp_input.astype(np_type).tofile(f"{d_type}_input_rgb2yuv422.bin")
    tmp_golden.astype(np_type).tofile(f"{d_type}_golden_rgb2yuv422.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2])
