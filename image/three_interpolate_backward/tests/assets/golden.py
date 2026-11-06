#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""three_interpolate_backward 的 Torch CPU golden 与 Torch 三方标杆。

参考语义（README / aclnnThreeInterpolateBackward 文档）：

    grad_y[b, c, idx[b, n, i]] += grad_x[b, c, n] * weight[b, n, i],  i in [0, 3)

先对 grad_y 整行（长度为 m）清零，再按 i 外层、n 内层散射累加；n == 0 时输出为全零。
参考计算全程由 Torch 基础算子组合承担，numpy 仅作为容器边界。
"""

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

__spec__ = {
    "three_interpolate_backward": "ThreeInterpolateBackwardKernelSpec",
    "ThreeInterpolateBackward": "ThreeInterpolateBackwardKernelSpec",
    "aclnnThreeInterpolateBackward": "ThreeInterpolateBackwardAclnnSpec",
}

__golden__ = {
    "kernel": {
        "three_interpolate_backward": "golden_three_interpolate_backward",
        "ThreeInterpolateBackward": "golden_three_interpolate_backward",
    },
}


def _three_interpolate_backward_compute(grad_x, idx, weight, m):
    """仅用 Torch 基础算子复现散射累加语义。

    输入可为 numpy 或 torch.Tensor；输出为与 grad_x 同 dtype 的 torch.Tensor。
    累加语义与 kernel 一致：单项乘积与逐次累加都在 float32 中完成，仅行末把
    fp32 累加结果一次性 cast 回输出 dtype（与 kernel 相同的唯一舍入点）；
    累加顺序为 k 外层（0..2）、n 内层升序，与 kernel 行私有串行累加顺序一致。
    注意：不能使用 scatter_add_ 汇总（其内部累加/舍入行为不可控），故按步
    gather/scatter 保持与 kernel 完全相同的 fp32 乘加序列。
    """
    grad_x_t = torch.as_tensor(grad_x)
    out_dtype = grad_x_t.dtype
    gx = grad_x_t.to(dtype=torch.float32).contiguous()
    ix = torch.as_tensor(idx).to(dtype=torch.long).contiguous()
    wt = torch.as_tensor(weight).to(dtype=torch.float32).contiguous()

    m_int = int(m)
    if gx.dim() != 3:
        return torch.zeros((0, 0, max(m_int, 0)), dtype=out_dtype)

    b, c, n = int(gx.shape[0]), int(gx.shape[1]), int(gx.shape[2])
    acc = torch.zeros((b, c, m_int), dtype=torch.float32)
    if b == 0 or c == 0 or n == 0 or m_int == 0:
        return acc.to(dtype=out_dtype)

    # k 外层、n 内层升序；每个 (k, n) 步对每个目标元素恰好一次乘加，
    # 全程 float32 累加（fp16 输入转 float32 无损，乘积精确）。
    for k in range(3):
        for ni in range(n):
            j = ix[:, ni, k]  # (b,)
            prod = gx[:, :, ni] * wt[:, ni, k].unsqueeze(1)  # (b, c) float32 乘
            jf = j.view(b, 1, 1).expand(b, c, 1)
            cur = acc.gather(2, jf).squeeze(2)  # (b, c)
            acc.scatter_(2, jf, (cur + prod).unsqueeze(2))  # float32 加
    return acc.to(dtype=out_dtype)


class ThreeInterpolateBackwardKernelSpec:
    """three_interpolate_backward 的 Kernel/GEIR TestSpec。"""

    @staticmethod
    def golden(grad_x, idx, weight, *, m, **kwargs):
        """CPU 真值：numpy 输入转换，计算仅使用 Torch 基础小算子组合。"""
        del kwargs
        return [_three_interpolate_backward_compute(grad_x, idx, weight, m).numpy()]

    class ThirdPartyImpl:
        """Torch 三方标杆：以 gather 组合构造可微前向 three_interpolate，再用
        autograd 求反向（grad_points 即本算子输出），与 golden 的 scatter 实现
        互为独立印证。前向/反向在 float32 中计算后回原 dtype；可在 CPU 或
        可用 XPU 设备上执行。cross_check 标准需要 XPU provider 时由框架在
        设备上调用；本验收环境未启用 cross_check，该实现不被比较。
        """

        def __init__(self, *, m, **kwargs):
            self.m = int(m)
            self.device = torch.device(kwargs.get("device") or "cpu")

        def __call__(self, grad_x=None, idx=None, weight=None, **kwargs):
            if grad_x is None or idx is None or weight is None:
                raise RuntimeError(
                    "three_interpolate_backward third-party reference requires grad_x, idx and weight"
                )
            gx = torch.as_tensor(grad_x).to(device=self.device)
            ix = torch.as_tensor(idx).to(device=self.device, dtype=torch.long)
            wt = torch.as_tensor(weight).to(device=self.device)
            out_dtype = gx.dtype
            if gx.dim() != 3:
                return [torch.zeros((0, 0, max(self.m, 0)), dtype=out_dtype)]
            b, c, n = int(gx.shape[0]), int(gx.shape[1]), int(gx.shape[2])
            if b == 0 or c == 0 or n == 0 or self.m <= 0:
                return [torch.zeros((b, c, self.m), dtype=out_dtype)]

            points = torch.zeros(
                (b, c, self.m),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            forward = torch.zeros((b, c, n), dtype=torch.float32, device=self.device)
            for k in range(3):
                gathered = torch.gather(
                    points, 2, ix[:, :, k].unsqueeze(1).expand(b, c, n)
                )
                forward = forward + gathered * wt[:, :, k].unsqueeze(1).to(
                    dtype=torch.float32
                )
            forward.backward(gradient=gx.to(dtype=torch.float32))
            return [points.grad.to(dtype=out_dtype)]

    third_party = {"torch": ThirdPartyImpl}

    # 比较标准按 TTK 官方 binary_equal 声明（同仓已验证的散射累加类算子 arg_max_grad、
    # scatter_list 同款约定）：本算子 kernel 以 user workspace 做 fp32 行累加、行末一次性
    # cast 回输出 dtype（k 外层、n 内层升序），golden 以上文逐步 gather/scatter 复现同一
    # fp32 乘加序列与唯一舍入点，与设备输出逐位一致（含 NaN/Inf/±0/次正规数与空 tensor）。
    # 不声明 mix_tolerance：该标准会把 golden 输入提升到 fp32/f64 按“真值”比较，
    # 而散射累加在大数抵消/±FLT_MAX 溢出场景下，fp32 kernel 与提升精度真值的偏差
    # 必然超过阈值（fp32 kernel 对 ±FLT_MAX 权重的中间积溢出为 inf/NaN，
    # 任何 fp32 实现都无法通过该比较）。三方对比另由 --compare cross_check 单独执行。
    tolerance = {
        "float16": {"standard": "binary_equal"},
        "float32": {"standard": "binary_equal"},
    }


class ThreeInterpolateBackwardAclnnSpec:
    """three_interpolate_backward 的 ACLNN TestSpec（torch 输入/输出，含 out 位）。"""

    @staticmethod
    def golden(grad_x, idx, weight, m, grad_y=None, **kwargs):
        del grad_y, kwargs
        return [_three_interpolate_backward_compute(grad_x, idx, weight, m)]

    third_party = {"torch": ThreeInterpolateBackwardKernelSpec.ThirdPartyImpl}

    # 与 Kernel Spec 同一参考实现与比较标准（binary_equal，理由见上）。
    tolerance = {
        "float16": {"standard": "binary_equal"},
        "float32": {"standard": "binary_equal"},
    }


def golden_three_interpolate_backward(grad_x, idx, weight, *, m, **kwargs):
    """TTK __golden__ 兼容入口 — 接收 numpy，返回 numpy 列表。"""
    return ThreeInterpolateBackwardKernelSpec.golden(grad_x, idx, weight, m=m, **kwargs)
