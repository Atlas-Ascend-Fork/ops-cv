# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
import numpy as np

loss = 1e-3
minimum = 10e-10
c0 = 16


def FiveDToFourD(data, cs):
    data = data.transpose(0, 1, 4, 2, 3)
    shape = data.shape
    data = data.reshape((shape[0], shape[1] * shape[2], shape[3], shape[4]))
    data = np.squeeze(data, axis=3)
    return data[:, 0:cs, :]


def verify_result(real_result, golden, bs, cs, ms, ns, dtype):
    real_result = np.fromfile(real_result, dtype=dtype)
    golden = np.fromfile(golden, dtype=dtype)
    golden = golden.reshape(bs, cs, ms)
    c1 = (cs + c0 - 1) // c0
    real_result = real_result.reshape(bs, c1, ms, 1, c0).astype(dtype)
    real_result = FiveDToFourD(real_result, cs)

    # print("real_result:")
    # print(real_result)
    # print("golden:")
    # print(golden)

    return np.allclose(golden, real_result, 1e-3, 1e-4)


if __name__ == "__main__":
    data_type = int(sys.argv[7])
    dtype = np.float32
    if data_type == 1:
        dtype = np.float16

    if verify_result(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]),
        int(sys.argv[5]),
        int(sys.argv[6]),
        dtype,
    ):
        exit(0)
    else:
        exit(-1)
