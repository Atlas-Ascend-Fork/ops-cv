#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import numpy as np
import glob
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))


def compare_data(golden_file_lists, output_file_lists, d_type):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "int32": np.int32,  # LBP 输出统一为 int32（UINT8/FLOAT 输入都是）
    }
    np_dtype = d_type_dict[d_type]

    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out, np_dtype)
        tmp_gold = np.fromfile(gold, np_dtype)

        # 元素个数先对上，否则 fromfile 静默截断会造成"假通过"
        if tmp_out.size != tmp_gold.size:
            print("FAILED! (size mismatch)")
            print(f"  golden: {gold} -> {tmp_gold.size} elements")
            print(f"  output: {out} -> {tmp_out.size} elements")
            data_same = False
            continue

        # int32 输出用零容差全等比较（isclose 的 atol=0, rtol=0）
        diff_res = np.isclose(tmp_out, tmp_gold, 0, 0, True)
        diff_idx = np.where(~diff_res)[0]
        if len(diff_idx) == 0:
            print("PASSED!")
            print(
                f"  {os.path.basename(gold)} vs {os.path.basename(out)}: "
                f"{tmp_gold.size} elements, all equal"
            )
        else:
            print("FAILED!")
            print(f"  差异元素个数: {len(diff_idx)} / {tmp_gold.size}")
            for idx in diff_idx[:5]:
                print(f"index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False
    return data_same


def get_file_lists(dtype):
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    return golden_file_lists, output_file_lists


def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)

    if len(golden_file_lists) == 0 or len(output_file_lists) == 0:
        print("FAILED! (file not found)")
        print(f"  golden files: {golden_file_lists}")
        print(f"  output files: {output_file_lists}")
        return False

    if len(golden_file_lists) != len(output_file_lists):
        print("FAILED! (file count mismatch)")
        print(
            f"  golden: {len(golden_file_lists)} 个, output: {len(output_file_lists)} 个"
        )
        print("  疑似上一次用例的旧 bin 未清理，检查文件配对")
        return False

    result = compare_data(golden_file_lists, output_file_lists, d_type)
    print("compare result:", result)
    return result


if __name__ == "__main__":
    ret = process(sys.argv[1])
    exit(0 if ret else 1)
