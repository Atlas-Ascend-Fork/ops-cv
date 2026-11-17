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

"""GuidedFilter kernel UT 输出比对脚本。

把数据目录下 gen_data.py 产出的 *golden*.bin 与 kernel 落盘的 *output*.bin 逐元素比对。

判定容差：
  float32: np.isclose(rtol=2e-4, atol=1e-5)

  参考实现是 numpy float64、内核是 float32，量级 O(1) 时正常相对误差约 1e-6，
  2e-4 的余量足以覆盖 float32 累加差异，同时能拦住任何结构性错误
  （guided filter 的输出与输入同量级，索引/通道/边界错位都会远超这个量级）。

另外**显式检查 NaN/Inf**：np.isclose 默认把 NaN 判为不等（正确），但 Inf==Inf 会
判为相等，所以非有限值要单独报出来，避免「全 Inf 也 PASS」。

用法: python3 compare_data.py <dtype>     # dtype 取 float32 或 float16
退出码: 0 表示全部一致，1 表示存在超差元素、非有限值或文件缺失。
"""

import argparse
import glob
import os
import sys

import numpy as np

DATA_DIR = os.path.dirname(os.path.realpath(__file__))
SUPPORTED_DTYPE = {"float32": np.float32, "float16": np.float16}
FP32_RTOL = 2e-4
FP32_ATOL = 1e-5
FP16_RTOL = 2e-3
FP16_ATOL = 1e-2
REPORT_LIMIT = 5


def collect_bins(kind):
    """按文件名收集 golden / output 二进制，排序后两侧下标一一对应。"""
    pattern = os.path.join(DATA_DIR, "*{}*.bin".format(kind))
    return sorted(glob.glob(pattern))


def pick_bad_indices(actual, expected, dtype_name):
    """返回超差元素下标；返回空数组表示全部在容差内。"""
    got = actual.astype(np.float64)
    want = expected.astype(np.float64)
    rtol = FP16_RTOL if dtype_name == "float16" else FP32_RTOL
    atol = FP16_ATOL if dtype_name == "float16" else FP32_ATOL
    ok = np.isclose(got, want, rtol=rtol, atol=atol, equal_nan=False)
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

    # 非有限值单独检查：Inf==Inf 会被 isclose 判成相等，必须显式拦住
    bad_finite = np.flatnonzero(np.logical_not(np.isfinite(output.astype(np.float64))))
    if bad_finite.size != 0:
        print(
            "FAILED! {}/{} elements are NaN/Inf (kernel output is not finite)".format(
                bad_finite.size, output.size
            )
        )
        for idx in bad_finite[:REPORT_LIMIT]:
            print("  idx={} output={} golden={}".format(idx, output[idx], golden[idx]))
        return False

    bad = pick_bad_indices(output, golden, dtype_name)
    if bad.size == 0:
        print("PASSED!")
        return True

    print("FAILED! {}/{} elements out of tolerance".format(bad.size, golden.size))
    for idx in bad[:REPORT_LIMIT]:
        got = float(output[idx])
        want = float(golden[idx])
        rel = abs(got - want) / max(abs(want), 1e-12)
        print(
            "  idx={} output={:.9g} golden={:.9g} abs={:.3e} rel={:.3e}".format(
                idx, got, want, abs(got - want), rel
            )
        )
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
        description="compare GuidedFilter kernel output with golden"
    )
    parser.add_argument(
        "dtype", choices=sorted(SUPPORTED_DTYPE), help="float32 or float16"
    )
    args = parser.parse_args()
    return compare_all(args.dtype)


if __name__ == "__main__":
    sys.exit(main())
