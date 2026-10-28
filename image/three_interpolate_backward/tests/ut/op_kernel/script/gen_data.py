# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy as np
import sys


def FourDToFiveD(data):
    b, c, n = data.shape
    dtype = data.dtype
    data = data.reshape(b, c, n, 1)
    c0 = 16
    padding_c = c0 - c % c0 if c % c0 else 0

    if padding_c:
        padding_shape = [b, padding_c, n, 1]
        data = np.concatenate((data, np.zeros(padding_shape, dtype)), axis=1)

    new_shape = data.shape
    c1 = new_shape[1] // c0
    data = data.reshape((new_shape[0], c1, c0, new_shape[2], new_shape[3]))
    data = data.transpose(0, 1, 3, 4, 2)
    return data


def FiveDToFourD(data, cs):
    data = data.transpose(0, 1, 4, 2, 3)
    shape = data.shape
    data = data.reshape((shape[0], shape[1] * shape[2], shape[3], shape[4]))
    data = np.squeeze(data, axis=3)
    return data[:, 0:cs, :]


def three_interpolate_backward_golden(bs, cs, ms, ns, dtype, invalid_idx=0):
    grad_x = np.ones((bs, cs, ns)).astype(dtype)
    idx = np.random.randint(0, ms, size=(bs, ns, 3), dtype=np.int32)
    if invalid_idx:
        # issue #1405 回归：混入越界取值（等于 ms、负数、大于 ms），golden 同步丢弃其贡献
        idx[0][0][0] = ms
        idx[0][1][1] = -1
        idx[0][2][2] = ms + 1
    weight = np.random.uniform(-10, 10, (bs, ns, 3)).astype(dtype)
    grad_y = np.zeros((bs, cs, ms), dtype=dtype)

    for b in range(bs):
        for c in range(cs):
            for n in range(ns):
                # 与 kernel 拦截语义一致：越界取值（负数或 >= ms）的贡献不参与 grad_y 累加（issue #1405）
                if 0 <= idx[b][n][0] < ms:
                    grad_y[b][c][idx[b][n][0]] = (
                        grad_y[b][c][idx[b][n][0]] + grad_x[b][c][n] * weight[b][n][0]
                    )
                if 0 <= idx[b][n][1] < ms:
                    grad_y[b][c][idx[b][n][1]] = (
                        grad_y[b][c][idx[b][n][1]] + grad_x[b][c][n] * weight[b][n][1]
                    )
                if 0 <= idx[b][n][2] < ms:
                    grad_y[b][c][idx[b][n][2]] = (
                        grad_y[b][c][idx[b][n][2]] + grad_x[b][c][n] * weight[b][n][2]
                    )

    grad_x = FourDToFiveD(grad_x)

    grad_x.tofile("./grad_x.bin")
    idx.tofile("./idx.bin")
    weight.tofile("./weight.bin")
    grad_y.tofile("./golden.bin")


if __name__ == "__main__":
    bs = int(sys.argv[1])
    cs = int(sys.argv[2])
    ms = int(sys.argv[3])
    ns = int(sys.argv[4])
    dtype = int(sys.argv[5])
    invalid_idx = int(sys.argv[6]) if len(sys.argv) > 6 else 0

    data_type = np.float32
    if dtype == 1:
        data_type = np.float16
    three_interpolate_backward_golden(bs, cs, ms, ns, data_type, invalid_idx)
