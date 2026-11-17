#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.
# You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""stack_group_points 在 kernel/GEIR 测试路径下的 golden 编写。

Kernel/GEIR 的 golden 收到 numpy.ndarray，直接以 numpy 向量化原语拼接计算，
特征值原 dtype 进出、零 cast 纯位搬运。
stack_group_points 是点云特征分组算子，无直接同接口的框架 API，oracle 由
numpy.cumsum + searchsorted + gather 拼接实现。

取数合法域与设备 kernel 逐条对齐：
- 合法位置须同时满足 batch 局部双检查 0 <= cin < features_batch_cnt[bs_idx]
  与全局行号 fstart + cin ∈ [0, N)；
- 非法位置（负索引 / 越界索引）保持输出的零值，不报错。
"""

__spec__ = {
    "stack_group_points": "StackGroupPointsKernelSpec",
    "StackGroupPoints": "StackGroupPointsKernelSpec",
}

import numpy as np

_TOLERANCE = {
    "float16": {"standard": "binary_equal"},
    "float32": {"standard": "binary_equal"},
}


class StackGroupPointsKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray

    用 numpy.cumsum + searchsorted + fancy indexing 向量化实现，
    特征值零 cast 原样搬运。

    tolerance: 搬运类算子，特征值原样复制无精度损失，输出值域为输入值
    子集并上零，全 dtype binary_equal 逐位比对。
    """

    tolerance = _TOLERANCE

    def golden(features, features_batch_cnt, indices, indices_batch_cnt, **kwargs):
        # 计数升 int64：kernel 批扫描以 int64 前缀累加计数，numpy cumsum
        # 沿用输入 dtype，故必须显式升宽后再累加
        fbc = features_batch_cnt.astype(np.int64)
        ibc = indices_batch_cnt.astype(np.int64)

        N, C = features.shape
        M, nsample = indices.shape
        B = ibc.shape[0]

        output = np.zeros((M, C, nsample), dtype=features.dtype)
        if M == 0 or C == 0 or nsample == 0 or B == 0 or fbc.shape[0] == 0 or N == 0:
            return [output]

        # 批次归属：searchsorted right 语义 + 上界钳位 B-1，非负计数域上与
        # kernel 的线性扫描逐位一致
        ibc_cumsum = np.cumsum(ibc)
        bs_idx = np.searchsorted(ibc_cumsum, np.arange(M, dtype=np.int64), side="right")
        bs_idx = np.minimum(bs_idx, B - 1)

        fbc_padded = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(fbc)))
        fbs = fbc_padded[bs_idx][:, None, None]
        fbcnt = fbc[bs_idx][:, None, None]

        # cin 提升至 int64 再做索引算术：kernel 以 static_cast<int64_t>(cin) 计算
        # fs_idx（stack_group_points.h），int32 乘 C 会回绕出假合法索引
        cin = indices.astype(np.int64)[:, None, :]
        c_idx = np.arange(C, dtype=np.int64)[None, :, None]
        in_idx = cin * C + c_idx

        feat_len = N * C

        # 合法域 = kernel 双重判定: batch 局部双检查 0 <= cin < 本 batch 容量
        # （c_idx ∈ [0, C) 与 cin 单调对应，故 in_idx >= 0 等价 cin >= 0，
        #   in_idx < fbcnt * C 等价 cin < fbcnt）
        mask = (in_idx >= 0) & (in_idx < fbcnt * C)
        # 且全局行号 fstart + cin ∈ [0, N)（fs_idx 逐元素落在 [0, feat_len) 内）
        fs_idx = in_idx + fbs * C
        mask = mask & (fs_idx >= 0) & (fs_idx < feat_len)

        gathered = features.reshape(-1)[np.clip(fs_idx, 0, feat_len - 1)]

        out_flat = output.reshape(-1)
        m_flat = mask.reshape(-1)
        out_flat[m_flat] = gathered.reshape(-1)[m_flat]

        return [output]
