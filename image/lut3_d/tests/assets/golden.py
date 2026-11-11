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

import numpy as np

__spec__ = {
    "lut3_d": "Lut3DTestSpec",
}

__golden__ = {
    "kernel": {"lut3_d": "lut3_d_golden"},
}


def _lut3_d_compute(img, lut_table):
    """LUT3D 三线性插值核心计算（Torch 基础算子组合）。"""
    import torch

    # 统一转为 float32 计算
    img_f32 = torch.from_numpy(np.ascontiguousarray(img)).to(torch.float32)
    lut_table_f32 = torch.from_numpy(np.ascontiguousarray(lut_table)).to(torch.float32)

    lut_n = lut_table_f32.shape[0]  # 查找表边长 N

    # 获取图像像素数和通道数
    orig_shape = img_f32.shape
    if len(orig_shape) == 3:
        H, W, C = orig_shape
        N_batch = 1
    elif len(orig_shape) == 4:
        N_batch, H, W, C = orig_shape
    else:
        raise ValueError(f"img dims should be 3 or 4, got {len(orig_shape)}")

    assert C == 3, f"img last dim must be 3 (RGB), got {C}"
    assert lut_table_f32.shape == (lut_n, lut_n, lut_n, 3), (
        f"lut_table shape must be [N,N,N,3], got {tuple(lut_table_f32.shape)}"
    )

    # 展平为 [num_pixels, 3] 以便向量化计算
    pixels = img_f32.reshape(-1, 3)  # [num_pixels, 3]

    # 步骤1: 像素值缩放 [0, 255] → [0, N-1]
    # 强制 fp32 标量除法, 避免默认提升为 fp64 导致舍入不一致。
    scale = torch.tensor(lut_n - 1, dtype=torch.float32) / torch.tensor(
        255.0, dtype=torch.float32
    )
    img_scaled = pixels * scale  # [num_pixels, 3], 通道顺序 B, G, R

    # 步骤2: 计算 floor/ceil 索引和插值权重
    # 直接取整不做 clip; 输入域 [0,255] 内 scaled ∈ [0, N-1],
    # floor/ceil 索引天然落在 [0, N-1]。
    floor_idx = torch.floor(img_scaled).to(torch.int32)
    ceil_idx = torch.ceil(img_scaled).to(torch.int32)

    fract = img_scaled - floor_idx.to(torch.float32)
    # fract_1 = abs(fract - 1.0), 对 fract ∈ [0,1] 与 1.0 - fract 逐位等价。
    fract_1 = torch.abs(fract - 1.0)

    b_floor = floor_idx[:, 0]
    g_floor = floor_idx[:, 1]
    r_floor = floor_idx[:, 2]
    b_ceil = ceil_idx[:, 0]
    g_ceil = ceil_idx[:, 1]
    r_ceil = ceil_idx[:, 2]

    fract_b = fract[:, 0]
    fract_g = fract[:, 1]
    fract_r = fract[:, 2]
    fract_b_1 = fract_1[:, 0]
    fract_g_1 = fract_1[:, 1]
    fract_r_1 = fract_1[:, 2]

    # 步骤3: 计算 8 个角点的 LUT 线性索引
    # 在 fp32 下计算 b*N*N + g*N + r 再 round 转 int32 (N ≤ 20, 索引为精确整数)。
    lut_n_fp32 = torch.tensor(lut_n, dtype=torch.float32)
    b_floor_f = b_floor.to(torch.float32) * (lut_n_fp32 * lut_n_fp32)
    g_floor_f = g_floor.to(torch.float32) * lut_n_fp32
    b_ceil_f = b_ceil.to(torch.float32) * (lut_n_fp32 * lut_n_fp32)
    g_ceil_f = g_ceil.to(torch.float32) * lut_n_fp32
    r_floor_f = r_floor.to(torch.float32)
    r_ceil_f = r_ceil.to(torch.float32)

    def _mk_idx(b_f, g_f, r_f):
        bg = b_f + g_f
        bgr = bg + r_f
        return torch.round(bgr).to(torch.int32)

    idx_0 = _mk_idx(b_floor_f, g_floor_f, r_floor_f)
    idx_1 = _mk_idx(b_ceil_f, g_floor_f, r_floor_f)
    idx_2 = _mk_idx(b_floor_f, g_ceil_f, r_floor_f)
    idx_3 = _mk_idx(b_ceil_f, g_ceil_f, r_floor_f)
    idx_4 = _mk_idx(b_floor_f, g_floor_f, r_ceil_f)
    idx_5 = _mk_idx(b_ceil_f, g_floor_f, r_ceil_f)
    idx_6 = _mk_idx(b_floor_f, g_ceil_f, r_ceil_f)
    idx_7 = _mk_idx(b_ceil_f, g_ceil_f, r_ceil_f)

    # 步骤4: 查表获取 8 个角点的值
    lut_flat = lut_table_f32.reshape(-1, 3)

    def _gather(idx):
        return torch.index_select(lut_flat, 0, idx.to(torch.long))

    v0 = _gather(idx_0)
    v1 = _gather(idx_1)
    v2 = _gather(idx_2)
    v3 = _gather(idx_3)
    v4 = _gather(idx_4)
    v5 = _gather(idx_5)
    v6 = _gather(idx_6)
    v7 = _gather(idx_7)

    # 步骤5: 三线性插值
    # 顺序 R->G->B, 每步显式拆成两次独立乘法加一次加法, 避免 FMA 融合。
    r1 = v0 * fract_r_1.unsqueeze(1)
    r2 = v4 * fract_r.unsqueeze(1)
    vr0 = r1 + r2
    r1 = v1 * fract_r_1.unsqueeze(1)
    r2 = v5 * fract_r.unsqueeze(1)
    vr1 = r1 + r2
    r1 = v2 * fract_r_1.unsqueeze(1)
    r2 = v6 * fract_r.unsqueeze(1)
    vr2 = r1 + r2
    r1 = v3 * fract_r_1.unsqueeze(1)
    r2 = v7 * fract_r.unsqueeze(1)
    vr3 = r1 + r2

    g1 = vr0 * fract_g_1.unsqueeze(1)
    g2 = vr2 * fract_g.unsqueeze(1)
    vg0 = g1 + g2
    g1 = vr1 * fract_g_1.unsqueeze(1)
    g2 = vr3 * fract_g.unsqueeze(1)
    vg1 = g1 + g2

    b1 = vg0 * fract_b_1.unsqueeze(1)
    b2 = vg1 * fract_b.unsqueeze(1)
    lut_img_flat = b1 + b2

    # 恢复原始 shape
    lut_img = lut_img_flat.reshape(tuple(orig_shape))

    return lut_img.contiguous().cpu().numpy()


def _lut3_d_torch_compute(img, lut_table, device=None):
    """LUT3D 三线性插值的独立 Torch 竞品实现（供 third_party cross_check 使用）。

    采用教科书带 clamp 的实现路径，与 golden 的 fp32-round 索引路径相互独立：
    - floor/ceil 后 clamp 到 [0, N-1]（越界像素收敛到边界角点）；
    - fract_1 = 1.0 - fract（golden 用 abs(fract - 1)，[0,1] 上数值一致但写法独立）；
    - 角点索引用 long 整数算术（golden 用 fp32 乘加后 round）。
    对合法输入（uint8/float32 ∈ [0,255]）与 golden 结果一致，但代码路径独立，避免同源自证。

    设备可移植：跟随输入 Tensor 的 device（xpu_server 沙箱已将输入放到
    cpu 或 cuda:N），不再硬编码 cuda；仅当输入为 numpy 且框架显式注入
    device 时才落指定设备。
    """
    import torch

    def _to_f32(x):
        if isinstance(x, torch.Tensor):
            return x.detach().to(dtype=torch.float32)
        t = torch.from_numpy(np.ascontiguousarray(x))
        if device is not None:
            t = t.to(device)
        return t.to(dtype=torch.float32)

    img_f32 = _to_f32(img)
    lut_table_f32 = _to_f32(lut_table)

    lut_n = lut_table_f32.shape[0]  # 查找表边长 N
    orig_shape = img_f32.shape
    if img_f32.ndim not in (3, 4):
        raise ValueError(f"img dims should be 3 or 4, got {img_f32.ndim}")
    if orig_shape[-1] != 3:
        raise ValueError(f"img last dim must be 3 (RGB), got {orig_shape[-1]}")

    pixels = img_f32.reshape(-1, 3)  # [num_pixels, 3]

    scale = torch.tensor(lut_n - 1, dtype=torch.float32) / torch.tensor(
        255.0, dtype=torch.float32
    )
    img_scaled = pixels * scale

    # floor 后 clamp 到 [0, N-1]，ceil 由 clamp 后 floor + 1 再 clamp 得到
    floor_idx = torch.clamp(torch.floor(img_scaled).to(torch.long), 0, lut_n - 1)
    ceil_idx = torch.clamp(floor_idx + 1, 0, lut_n - 1)

    fract = img_scaled - floor_idx.to(torch.float32)
    fract_1 = 1.0 - fract

    b_floor = floor_idx[:, 0]
    g_floor = floor_idx[:, 1]
    r_floor = floor_idx[:, 2]
    b_ceil = ceil_idx[:, 0]
    g_ceil = ceil_idx[:, 1]
    r_ceil = ceil_idx[:, 2]

    fract_b = fract[:, 0]
    fract_g = fract[:, 1]
    fract_r = fract[:, 2]
    fract_b_1 = fract_1[:, 0]
    fract_g_1 = fract_1[:, 1]
    fract_r_1 = fract_1[:, 2]

    # 8 个角点的 LUT 线性索引（long 整数算术，不经 fp32 舍入）
    nsq = lut_n * lut_n
    idx_0 = b_floor * nsq + g_floor * lut_n + r_floor
    idx_1 = b_ceil * nsq + g_floor * lut_n + r_floor
    idx_2 = b_floor * nsq + g_ceil * lut_n + r_floor
    idx_3 = b_ceil * nsq + g_ceil * lut_n + r_floor
    idx_4 = b_floor * nsq + g_floor * lut_n + r_ceil
    idx_5 = b_ceil * nsq + g_floor * lut_n + r_ceil
    idx_6 = b_floor * nsq + g_ceil * lut_n + r_ceil
    idx_7 = b_ceil * nsq + g_ceil * lut_n + r_ceil

    lut_flat = lut_table_f32.reshape(-1, 3)

    def _gather(idx):
        return torch.index_select(lut_flat, 0, idx)

    v0 = _gather(idx_0)
    v1 = _gather(idx_1)
    v2 = _gather(idx_2)
    v3 = _gather(idx_3)
    v4 = _gather(idx_4)
    v5 = _gather(idx_5)
    v6 = _gather(idx_6)
    v7 = _gather(idx_7)

    # 三线性插值（R → G → B）
    vr0 = v0 * fract_r_1.unsqueeze(1) + v4 * fract_r.unsqueeze(1)
    vr1 = v1 * fract_r_1.unsqueeze(1) + v5 * fract_r.unsqueeze(1)
    vr2 = v2 * fract_r_1.unsqueeze(1) + v6 * fract_r.unsqueeze(1)
    vr3 = v3 * fract_r_1.unsqueeze(1) + v7 * fract_r.unsqueeze(1)

    vg0 = vr0 * fract_g_1.unsqueeze(1) + vr2 * fract_g.unsqueeze(1)
    vg1 = vr1 * fract_g_1.unsqueeze(1) + vr3 * fract_g.unsqueeze(1)

    lut_img_flat = vg0 * fract_b_1.unsqueeze(1) + vg1 * fract_b.unsqueeze(1)

    # 恢复原始 shape
    lut_img = lut_img_flat.reshape(tuple(orig_shape))

    return lut_img.contiguous().cpu().numpy()


class Lut3DTestSpec:
    """lut3_d kernel/geir 测试通路的 golden 规范入口。

    ``__spec__`` 注册（key=op_name, value=类名）供 TTK Loader 发现与绑定，
    保留 ``__golden__`` 作为历史命名回退，两者指向同一计算实现。
    """

    @staticmethod
    def golden(img, lut_table, **kwargs):
        return _lut3_d_compute(img, lut_table)

    class ThirdPartyImpl:
        """Torch 竞品实现（独立于 golden 的教科书路径）。

        ``device`` 为框架保留注入参数（cpu 或 cuda:N），供 numpy 输入落设备；
        Tensor 输入随其自身 device 执行。
        """

        def __call__(self, img, lut_table, device=None, **kwargs):
            return [_lut3_d_torch_compute(img, lut_table, device=device)]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
    }


def lut3_d_golden(img, lut_table, **kwargs):
    """
    Golden function for lut3_d.
    All the parameters (names and order) follow @lut3_d_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        img: numpy.ndarray, input image, shape [H,W,3] or [N,H,W,3],
             dtype uint8 or float32, value range [0, 255]
        lut_table: numpy.ndarray, 3D lookup table, shape [lut_n, lut_n, lut_n, 3],
                   dtype uint8 or float32
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor (numpy.ndarray, dtype float32)
    """
    return _lut3_d_compute(img, lut_table)
