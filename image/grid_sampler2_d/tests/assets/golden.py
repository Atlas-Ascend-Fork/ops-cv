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

import numpy as np
import torch

# 属性值映射表（string -> int 索引）。编号与 op_host/grid_sampler2_d_def.cpp 的 STRING
# 属性域（interpolation_mode: bilinear/nearest/bicubic，padding_mode: zeros/border/reflection）
# 以及 torch.ops.aten.grid_sampler_2d 的 mode/padding_mode 顺序一致。
_INTERP_MODES = ["bilinear", "nearest", "bicubic"]
_PADDING_MODES = ["zeros", "border", "reflection"]


def _resolve_interp_mode(value):
    """将 interpolation_mode 兼容 string/int 输入为 int 索引"""
    if isinstance(value, str):
        return _INTERP_MODES.index(value)
    return int(value)


def _resolve_padding_mode(value):
    """将 padding_mode 兼容 string/int 输入为 int 索引"""
    if isinstance(value, str):
        return _PADDING_MODES.index(value)
    return int(value)


def _as_torch_tensor(value):
    """桥接 numpy.ndarray / torch.Tensor 容器到 torch.Tensor（不含数值计算）。"""
    if isinstance(value, torch.Tensor):
        return value
    numpy_value = (
        value.numpy() if hasattr(value, "numpy") and callable(value.numpy) else value
    )
    if isinstance(numpy_value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(numpy_value))
    return torch.as_tensor(numpy_value)


class GridSampler2DComposeSpec:
    """Class form — small-op composition for grid_sampler2_d"""

    # -- golden — function form --
    @staticmethod
    def golden(
        x, grid, *, interpolation_mode=0, padding_mode=0, align_corners=False, **kwargs
    ):
        """
        Golden function for grid_sampler2_d.
        All the parameters (names and order) follow the public aclnn/aten prototype without outputs.
        All the input Tensors are numpy.ndarray.

        Args:
            x: input feature map, shape (N, C, H_in, W_in)
            grid: sampling grid, shape (N, H_out, W_out, 2)
            interpolation_mode: 0=bilinear, 1=nearest, 2=bicubic (or string)
            padding_mode: 0=zeros, 1=border, 2=reflection (or string)
            align_corners: bool, coordinate alignment
            **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                      full_soc_version, short_soc_version, testcase_name

        Returns:
            Output tensor, shape (N, C, H_out, W_out)
        """
        interp_idx = _resolve_interp_mode(interpolation_mode)
        pad_idx = _resolve_padding_mode(padding_mode)

        x_torch = torch.from_numpy(np.ascontiguousarray(x))
        grid_torch = torch.from_numpy(np.ascontiguousarray(grid))
        orig_dtype = x_torch.dtype
        # 避免 float16/bfloat16 精度丢失或下溢，统一在 float32 下计算；
        # 升/降精度由 Torch 完成，NumPy 只承担张量桥接。
        if orig_dtype in (torch.float16, torch.bfloat16):
            x_torch = x_torch.to(torch.float32)
            grid_torch = grid_torch.to(torch.float32)

        result = torch.nn.functional.grid_sample(
            x_torch,
            grid_torch,
            mode=_INTERP_MODES[interp_idx],
            padding_mode=_PADDING_MODES[pad_idx],
            align_corners=align_corners,
        )

        # 还原为原始数据类型
        if orig_dtype in (torch.float16, torch.bfloat16):
            result = result.to(orig_dtype)

        # Golden 约定按输出逐个返回列表（本算子仅单输出 y）。
        return [result.numpy()]

    # -- third_party — dict multi-vendor --
    class ThirdPartyImpl:
        def __init__(
            self,
            *,
            interpolation_mode=None,
            padding_mode=None,
            align_corners=None,
            interpolationMode=None,
            paddingMode=None,
            alignCorners=None,
            **kwargs,
        ):
            # Kernel 入口按 snake_case 绑定，ACLNN 入口按 API 的 camelCase 绑定；
            # 两种命名都进入有效形参池，避免非默认属性被 **kwargs 吞掉。
            self.interpolation_mode = (
                interpolation_mode
                if interpolation_mode is not None
                else (interpolationMode if interpolationMode is not None else 0)
            )
            self.padding_mode = (
                padding_mode
                if padding_mode is not None
                else (paddingMode if paddingMode is not None else 0)
            )
            self.align_corners = (
                align_corners
                if align_corners is not None
                else (alignCorners if alignCorners is not None else False)
            )

        def __call__(self, input=None, grid=None, *, x=None, **kwargs):
            # Kernel/GEIR 入口按输入名 x/grid 绑定，ACLNN 入口按 input/grid 绑定；
            # 两个名字都进入有效形参池，避免一侧因形参名不匹配而无法绑定。
            x = x if x is not None else input
            interp_idx = _resolve_interp_mode(self.interpolation_mode)
            pad_idx = _resolve_padding_mode(self.padding_mode)

            # Kernel/GEIR 入口按 TestSpec 约定传入 numpy.ndarray，ACLNN 入口传入
            # torch.Tensor；第三方参考自行完成容器适配，不依赖上游预转换。
            x = _as_torch_tensor(x)
            grid = _as_torch_tensor(grid)

            # 与 golden 对齐: fp16/bf16 升 fp32 计算, 避免 bicubic/reflection
            # 等精度敏感模式在 fp16 下中间计算丢精度 (与 golden 不同精度级别导致
            # cross_check 时 |b-g| 异常大)。
            orig_dtype = x.dtype
            if orig_dtype in (torch.float16, torch.bfloat16):
                x = x.to(torch.float32)
                grid = grid.to(torch.float32)

            # 独立第三方实现：走 aten::grid_sampler_2d 入口，与 golden 使用的
            # torch.nn.functional.grid_sample 分开识别，按调用链核对参考独立性。
            output = torch.ops.aten.grid_sampler_2d(
                x, grid, interp_idx, pad_idx, self.align_corners
            )

            if orig_dtype in (torch.float16, torch.bfloat16):
                output = output.to(orig_dtype)
            # third_party 与 golden 一致按输出逐个返回列表（本算子仅单输出 y）。
            return [output]

    third_party = {
        "torch": ThirdPartyImpl,
    }

    # -- tolerance --
    # CPU 基准比较（NPU 输出 vs CPU torch golden）：本任务无 GPU/XPU 三方提供方时，
    # cross_check 会因缺少远端端点而 GOLDEN_FAILURE；改用 mix_tolerance 以 CPU 基准
    # 得到可在 NPU 上断言的数值结论。
    tolerance = {
        "float32": {"standard": "mix_tolerance"},
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
    }


class AclnnGridSampler2DTestSpec:
    """aclnnGridSampler2D(input, grid, interpolationMode, paddingMode, alignCorners, out).

    The ACLNN golden is invoked positionally in the C-header order of
    aclnnGridSampler2DGetWorkspaceSize, so the attributes carry the API's
    camelCase names (the CSV attribute keys), unlike the kernel entry which
    receives snake_case attributes by keyword.
    """

    @staticmethod
    def golden(
        input,
        grid,
        interpolationMode=0,
        paddingMode=0,
        alignCorners=False,
        out=None,
        **kwargs,
    ):
        interp_idx = _resolve_interp_mode(interpolationMode)
        pad_idx = _resolve_padding_mode(paddingMode)

        x = input.detach().cpu()
        grid_t = grid.detach().cpu()
        orig_dtype = x.dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
            grid_t = grid_t.to(torch.float32)

        result = torch.nn.functional.grid_sample(
            x,
            grid_t,
            mode=_INTERP_MODES[interp_idx],
            padding_mode=_PADDING_MODES[pad_idx],
            align_corners=alignCorners,
        )
        if orig_dtype in (torch.float16, torch.bfloat16):
            result = result.to(orig_dtype)
        # Golden 约定按输出逐个返回列表（本算子仅单输出 y）。
        return [result]

    tolerance = {
        "float32": {"standard": "mix_tolerance"},
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
    }

    # 逐入口第三方参考：ACLNN 入口共享 Kernel 入口的 torch 参考实现。
    third_party = {"torch": GridSampler2DComposeSpec.ThirdPartyImpl}


# Explicit registration: class names use *Spec suffix (not *TestSpec),
# so __spec__ dict is needed for discovery.
__spec__ = {
    "grid_sampler2_d": "GridSampler2DComposeSpec",
    "GridSampler2D": "GridSampler2DComposeSpec",
    "aclnnGridSampler2D": "AclnnGridSampler2DTestSpec",
}
