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

"""PixelShuffle kernel UT 输出比对脚本。

把数据目录下 gen_data.py 产出的 *golden*.bin 与 kernel 落盘的 *output*.bin 逐元素比对，
判定容差：float32 用 np.isclose(atol=1e-5)、float16 用相对误差 1e-3。

用法: python3 compare_data.py <dtype>     # dtype 取 float32 或 float16
退出码: 0 表示全部一致，1 表示存在超差元素或文件缺失。
"""

import argparse
import glob
import os
import sys

import numpy as np

DATA_DIR = os.path.dirname(os.path.realpath(__file__))
SUPPORTED_DTYPE = {"float32": np.float32, "float16": np.float16}
FP32_ABS_TOL = 1e-5
FP16_REL_TOL = 1e-3
REPORT_LIMIT = 5


def collect_bins(kind):
    """按文件名收集 golden / output 二进制，排序后两侧下标一一对应。"""
    pattern = os.path.join(DATA_DIR, "*{}*.bin".format(kind))
    return sorted(glob.glob(pattern))


def pick_bad_indices(actual, expected, dtype_name):
    """返回超差元素下标；返回空数组表示全部在容差内。"""
    got = actual.astype(np.float64)
    want = expected.astype(np.float64)
    if dtype_name == "float16":
        # 半精度用相对误差；分母补极小量，避免 golden 为 0 时除零
        rel_err = np.abs(got - want) / (np.abs(want) + 1e-10)
        return np.flatnonzero(rel_err > FP16_REL_TOL)
    ok = np.isclose(got, want, rtol=0.0, atol=FP32_ABS_TOL, equal_nan=True)
    return np.flatnonzero(np.logical_not(ok))


def compare_one_pair(golden_path, output_path, dtype_name):
    np_dtype = SUPPORTED_DTYPE[dtype_name]
    golden = np.fromfile(golden_path, np_dtype)
    output = np.fromfile(output_path, np_dtype)
    if golden.size != output.size:
        print(
            "FAILED! element count mismatch: golden={} output={}".format(
                golden.size, output.size
            )
        )
        return False

    bad = pick_bad_indices(output, golden, dtype_name)
    if bad.size == 0:
        print("PASSED!")
        return True

    print("FAILED! {}/{} elements out of tolerance".format(bad.size, golden.size))
    for idx in bad[:REPORT_LIMIT]:
        print("  idx={} output={} golden={}".format(idx, output[idx], golden[idx]))
    return False


def compare_all(dtype_name):
    goldens = collect_bins("golden")
    outputs = collect_bins("output")
    if not goldens or not outputs:
        print(
            'No golden"/"output bin found under {}, skip comparison.'.format(DATA_DIR)
        )
        return 0
    if len(goldens) != len(outputs):
        print(
            "FAILED! file count mismatch: golden={} output={}".format(
                len(goldens), len(outputs)
            )
        )
        return 1

    results = [compare_one_pair(g, o, dtype_name) for g, o in zip(goldens, outputs)]
    print("compare result: {}".format(all(results)))
    return 0 if all(results) else 1


def main():
    parser = argparse.ArgumentParser(
        description="compare PixelShuffle kernel output with golden"
    )
    parser.add_argument(
        "dtype", choices=sorted(SUPPORTED_DTYPE), help="float32 or float16"
    )
    args = parser.parse_args()
    return compare_all(args.dtype)


if __name__ == "__main__":
    sys.exit(main())
