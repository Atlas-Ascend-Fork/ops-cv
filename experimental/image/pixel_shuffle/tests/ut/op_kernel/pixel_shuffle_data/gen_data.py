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


def golden_pixel_shuffle(x, scale_factor=2):
    """与 tests/golden.py 的 pixel_shuffle_golden 保持一致的 numpy 参考实现。"""
    r = int(scale_factor)
    input_dtype = x.dtype
    dim = x.ndim
    if dim == 4:
        n, c, h, w = x.shape
        c_out = c // (r * r)
        out = x.reshape(n, c_out, r, r, h, w)
        out = out.transpose(0, 1, 4, 2, 5, 3)
        out = out.reshape(n, c_out, h * r, w * r)
    elif dim == 5:
        n, c, d, h, w = x.shape
        c_out = c // (r * r * r)
        out = x.reshape(n, c_out, r, r, r, d, h, w)
        out = out.transpose(0, 1, 5, 2, 6, 3, 7, 4)
        out = out.reshape(n, c_out, d * r, h * r, w * r)
    else:
        raise ValueError(
            "pixel_shuffle only supports 4D or 5D input, got rank %d" % dim
        )
    return out.astype(input_dtype)


def gen_data_and_golden(shape_str, d_type="float32", scale_factor=2):
    d_type_dict = {"float32": np.float32, "float16": np.float16}
    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)
    shape_tuple = tuple(int(s) for s in shape)
    size = int(np.prod(shape))

    # 逐元素递增的唯一值：置换/索引/分段任何错位都会在比对中暴露
    tmp_input = np.arange(size, dtype=np.int64).reshape(shape_tuple).astype(np_type)
    tmp_golden = golden_pixel_shuffle(tmp_input, scale_factor).astype(np_type)

    tmp_input.tofile(f"{d_type}_input_pixel_shuffle.bin")
    tmp_golden.tofile(f"{d_type}_golden_pixel_shuffle.bin")
    print(
        f"shape_in={shape_tuple} shape_out={tmp_golden.shape} dtype={d_type} elems={tmp_golden.size}"
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Param num must be at least 3: <shape> <dtype> [scale_factor]")
        exit(1)
    os.system("rm -rf *.bin")
    scale = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    gen_data_and_golden(sys.argv[1], sys.argv[2], scale)
