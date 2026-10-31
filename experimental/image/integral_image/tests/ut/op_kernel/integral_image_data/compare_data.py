#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""IntegralImage op_kernel UT result checker.

用法: python3 compare_data.py <run_dir> <tag> <sat_dtype> [sqsum_enabled] [tilted_enabled]

逐元素比较 kernel 输出（<tag>_output_*.bin）与 golden（<tag>_golden_*.bin）。
sat 为必选输出，始终比对；sqsum / tilted 为可选输出（形态 1/2/3），
enabled 为 0 时调用方未提供该输出、kernel 已跳过对应分支，本脚本同样跳过比对。
整型输出要求完全一致；浮点输出按 dtype 量化误差给出容差。
"""

import os
import sys

import numpy as np

SAT_DTYPES = {"int32": np.int32, "float32": np.float32}
# 浮点结果容差 (rtol, atol)：kernel 与 golden 的累加顺序不同，且 sqsum 的累加量级可达 1e8，
# float32 在 2^24 以上无法表示所有整数，因此使用相对与绝对误差阈值。
FLOAT_TOL = {"sat": (1e-5, 1e-3), "sqsum": (1e-5, 1e-2), "tilted": (1e-5, 1e-3)}


def compare_one(run_dir, tag, name, dtype, tol):
    out_path = os.path.join(run_dir, "%s_output_%s.bin" % (tag, name))
    gold_path = os.path.join(run_dir, "%s_golden_%s.bin" % (tag, name))
    if not os.path.exists(out_path):
        print("FAILED! missing output file: %s" % out_path)
        return False
    got = np.fromfile(out_path, dtype)
    want = np.fromfile(gold_path, dtype)
    if got.size != want.size:
        print(
            "FAILED! %s size mismatch: output=%d golden=%d"
            % (name, got.size, want.size)
        )
        return False
    if np.issubdtype(dtype, np.floating):
        output_finite = np.isfinite(got)
        golden_finite = np.isfinite(want)
        if not output_finite.all() or not golden_finite.all():
            print(
                "FAILED! %s non-finite values: output=%d golden=%d"
                % (
                    name,
                    np.count_nonzero(~output_finite),
                    np.count_nonzero(~golden_finite),
                )
            )
            return False
    gold = want.astype(np.float64)
    diff = np.abs(got.astype(np.float64) - gold)
    denom = np.maximum(np.abs(gold), 1.0)
    max_diff = float(diff.max()) if diff.size else 0.0
    max_rel = float((diff / denom).max()) if diff.size else 0.0
    if np.issubdtype(dtype, np.integer):
        bad = np.nonzero(diff > 0)[0]
    else:
        rtol, atol = tol
        bad = np.nonzero(diff > atol + rtol * np.abs(gold))[0]
    if bad.size == 0:
        print("PASSED! %s max_diff=%.6g max_rel=%.3g" % (name, max_diff, max_rel))
        return True
    print(
        "FAILED! %s max_diff=%.6g max_rel=%.3g bad_count=%d"
        % (name, max_diff, max_rel, bad.size)
    )
    for idx in bad[:5]:
        print("  index=%d output=%.6g golden=%.6g" % (idx, got[idx], want[idx]))
    return False


def main():
    if len(sys.argv) not in (4, 6):
        print(__doc__)
        return 1
    run_dir, tag, sat_dtype = sys.argv[1:4]
    sqsum_enabled = int(sys.argv[4]) if len(sys.argv) == 6 else 1
    tilted_enabled = int(sys.argv[5]) if len(sys.argv) == 6 else 1
    if sat_dtype not in SAT_DTYPES:
        print("unsupported sat dtype: %s" % sat_dtype)
        return 1
    ok = True
    # sat 必选输出，始终比对
    ok = (
        compare_one(run_dir, tag, "sat", SAT_DTYPES[sat_dtype], FLOAT_TOL["sat"]) and ok
    )
    if sqsum_enabled:
        ok = compare_one(run_dir, tag, "sqsum", np.float32, FLOAT_TOL["sqsum"]) and ok
    else:
        print("SKIPPED! sqsum (optional output disabled, form 1)")
    if tilted_enabled:
        ok = (
            compare_one(
                run_dir, tag, "tilted", SAT_DTYPES[sat_dtype], FLOAT_TOL["tilted"]
            )
            and ok
        )
    else:
        print("SKIPPED! tilted (optional output disabled)")
    print("compare result:", ok, "(tag=%s)" % tag)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
