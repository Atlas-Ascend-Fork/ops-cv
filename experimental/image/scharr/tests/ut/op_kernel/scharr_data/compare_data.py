#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Bitwise compare of Scharr kernel output against golden (float32)."""

import sys
from pathlib import Path

import numpy as np

EXPECTED = float("0")


def main():
    output = np.fromfile(Path("output.bin"), dtype=np.float32)
    golden = np.fromfile(Path("output") / "golden.bin", dtype=np.float32)
    if output.size != golden.size:
        print(f"size mismatch: output={output.size}, golden={golden.size}")
        return 1
    neq = output.view(np.int32) != golden.view(np.int32)
    neq_cnt = int(neq.sum())
    if neq_cnt == 0:
        print(f"bit-exact: {golden.size}/{golden.size} matched")
        return 0
    idx = np.where(neq)[0]
    print(f"mismatch: {neq_cnt} / {golden.size}")
    for i in idx[:5]:
        print(f"  index {i}: output={output[i]!r}, golden={golden[i]!r}")
    max_abs = float(np.max(np.abs(output[idx] - golden[idx])))
    print(f"  max abs error: {max_abs}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
