#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compare aclnn Scharr example dumps with OpenCV (cv2) reference output.

The example (tests/../examples/test_aclnn_scharr.cpp) writes
``scharr_input_<dtype>_<direction>_<border>.bin`` and
``scharr_output_<dtype>_<direction>_<border>.bin`` files. This script rebuilds
the same Scharr response with cv2 and checks the NPU result for bit-exactness.

Usage:
    python3 compare_with_cv2.py --dump-dir <dir> --shape 65,64 [--rtol 0]
"""

import argparse
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:
    print("[ERROR] cv2 is required: pip install opencv-python-headless")
    sys.exit(2)

DTYPE_MAP = {
    "uint8": np.uint8,
    "float16": np.float16,
    "float32": np.float32,
}
BORDER_MAP = {
    "REFLECT_101": cv2.BORDER_REFLECT_101,
    "REPLICATE": cv2.BORDER_REPLICATE,
}
# Scharr 3x3 kernels used by OpenCV (X direction; Y is the transpose).
KERNEL_X = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)


def reference(src32, direction, border):
    if border in BORDER_MAP:
        dx, dy = (1, 0) if direction == "X" else (0, 1)
        return cv2.Scharr(src32, cv2.CV_32F, dx, dy, borderType=BORDER_MAP[border])
    kernel = KERNEL_X if direction == "X" else KERNEL_X.T
    return cv2.filter2D(src32, cv2.CV_32F, kernel, borderType=cv2.BORDER_CONSTANT)


def parse_case(name):
    # scharr_input_<dtype>_<direction>_<border>.bin
    body = name[len("scharr_input_") : -len(".bin")]
    for border in ("REFLECT_101", "REPLICATE", "CONSTANT"):
        suffix = "_" + border
        if body.endswith(suffix):
            head = body[: -len(suffix)]
            dtype, direction = head.rsplit("_", 1)
            return dtype, direction, border
    raise ValueError("unexpected dump name: %s" % name)


def run_one(dump_dir, dtype, direction, border, shape, rtol):
    in_path = os.path.join(
        dump_dir, "scharr_input_%s_%s_%s.bin" % (dtype, direction, border)
    )
    out_path = os.path.join(
        dump_dir, "scharr_output_%s_%s_%s.bin" % (dtype, direction, border)
    )
    if not (os.path.isfile(in_path) and os.path.isfile(out_path)):
        print("[SKIP] %s_%s_%s: dumps not found" % (dtype, direction, border))
        return None

    src = np.fromfile(in_path, dtype=DTYPE_MAP[dtype]).reshape(shape)
    out = np.fromfile(out_path, dtype=np.float32).reshape(shape)
    ref = reference(src.astype(np.float32), direction, border).reshape(shape)

    equal = bool(np.array_equal(out, ref))
    diff = np.abs(out.astype(np.float64) - ref.astype(np.float64))
    max_diff = float(diff.max()) if diff.size else 0.0
    bad = int(np.count_nonzero(diff > 0))
    ok = (
        equal
        if rtol == 0
        else bool(np.allclose(out, ref, rtol=rtol, atol=0, equal_nan=True))
    )
    print(
        "[%s] %-8s %-4s %-12s exact=%s max_abs_diff=%g mismatched=%d/%d"
        % (
            "PASS" if ok else "FAIL",
            dtype,
            direction,
            border,
            equal,
            max_diff,
            bad,
            out.size,
        )
    )
    return ok


def main():
    parser = argparse.ArgumentParser(
        description="Compare Scharr aclnn dumps with cv2 reference."
    )
    parser.add_argument(
        "--dump-dir", required=True, help="directory holding the dumped .bin files"
    )
    parser.add_argument("--shape", required=True, help="input shape as H,W")
    parser.add_argument(
        "--rtol", type=float, default=0.0, help="0 (default) requires bit-exact match"
    )
    args = parser.parse_args()

    h, w = (int(v) for v in args.shape.split(","))
    names = sorted(
        n
        for n in os.listdir(args.dump_dir)
        if n.startswith("scharr_input_") and n.endswith(".bin")
    )
    if not names:
        print("[ERROR] no scharr_input_*.bin found in %s" % args.dump_dir)
        return 2

    results = []
    for name in names:
        dtype, direction, border = parse_case(name)
        results.append(
            run_one(args.dump_dir, dtype, direction, border, (h, w), args.rtol)
        )

    checked = [r for r in results if r is not None]
    failed = [r for r in checked if not r]
    print(
        "\nsummary: %d checked, %d passed, %d failed"
        % (len(checked), len(checked) - len(failed), len(failed))
    )
    return 0 if checked and not failed else 1


if __name__ == "__main__":
    sys.exit(main())
