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

"""TTK golden for ResizeLinearGrad (1D linear backward, NCL)
+ aclnnUpsampleLinear1dBackward.

标杆标准: ops-precision-standard 浮点计算类 · 单标杆比对 · 优先级-1 高精度
CPU 实现 (PyTorch CPU aten 原生). 详见仓外 BENCHMARK_STANDARD.md.

两个入口:
  * ``resize_linear_grad``             —— kernel 路径 (Ascend IR, 属性 scale)
  * ``aclnnUpsampleLinear1dBackward``  —— ACLNN 路径 (ATen 语义)

正式 golden 直接调用 ``torch.ops.aten.upsample_linear1d_backward`` —— 前向
``upsample_linear1d`` 的原生反向, **使用实际上游梯度 grads**(非 ones_like),
不复刻 kernel、不依赖被测算子。

分支表 (kernel 入口, 单分支):
    条件      : 全部 (Linear 隐含 half_pixel = not align_corners)
    接口      : upsample_linear1d_backward(grad_NCL, [Lo], [N,C,Li], align_corners, scales)
    grad      : 实际上游梯度 grads (grad_output)
    input_size: 由 original_image 提供 [N,C,Li] (仅用形状)
    scales    : CANN ``scale`` 属性>0 时映射到 torch scales; 否则 None
    dtype     : float64 计算 → 回原 dtype
    实证偏差  : ≤6.66e-16 (adjoint 恒等 <fwd(x),g>==<x,bwd(g)> 成立, 含非均匀梯度)
    ALL_COPY  : L_out==L_in → 直返 grads(拷贝的反向仍是拷贝), 对齐 kernel
                ComputeKey ComputeDataCopy(忽略 scale)

不捕获异常静默回退: torch 调用失败即直接抛出。
"""

import numpy as np
import torch

__spec__ = {
    "resize_linear_grad": "ResizeLinearGradTestSpec",
    "aclnnUpsampleLinear1dBackward": "AclnnUpsampleLinear1dBackwardTestSpec",
    "torch.ops.aten.upsample_linear1d_backward": (
        "E2eUpsampleLinear1dBackwardTestSpec"
    ),
}


def _scales_arg(scale):
    if scale is None:
        return None
    try:
        v = float(scale)
    except (TypeError, ValueError):
        return None
    return v if v > 0.0 else None


# --------------------------------------------- 零长度轴口径 (HPTG #147 FR-07)
_ZERO_SPATIAL_REJECT = (
    "零长度空间轴 (L_in == 0 或 L_out == 0): 线性插值的源/目标坐标映射在该身份上"
    "无定义, 算子 tiling 侧亦拒绝该身份。golden 按接口合同显式标注为**不可比较**"
    "(返回 None → TTK 记 SUPPRESSED), 既不留作 ATen 的 "
    "'Input and output sizes should be greater than 0' 静默异常, 也不计为通过。"
)


def _zero_axis_verdict(n, c, l_in, l_out):
    """零轴分类: "reject" 非法零空间轴 / "empty" 合法空张量 / None 常规。

    按接口合同区分两类零:
      * N == 0 或 C == 0 —— 只是批/通道维为空, 形状语义完整, 反向结果是同形空
        张量(ATen 本身也支持, 实测 N=0 -> (0,2,4)、C=0 -> (1,0,4));
      * L_in == 0 或 L_out == 0 —— 空间轴退化, 插值无定义。

    不把两者混为一谈: 合法空输入照常给出参考, 非法身份显式拒绝。
    """
    if int(l_in) == 0 or int(l_out) == 0:
        return "reject"
    if int(n) == 0 or int(c) == 0:
        return "empty"
    return None


def _linear_grad(grads, input_size, align_corners, scale):
    """PyTorch CPU 高精度标杆: aten.upsample_linear1d_backward, float64。"""
    n, c, lo = grads.shape
    li = int(input_size[2])
    # kernel ComputeKey: lenSrcLOrUb_==lenDesL_ 时 ComputeDataCopy() 无条件纯拷贝
    # (resize_linear_grad_tiling_arch35.cpp:192-195)。拷贝前向的反向仍是拷贝,
    # grad 原样回传, 忽略 scale / align_corners。
    if int(lo) == li:
        return np.ascontiguousarray(grads, dtype=np.float64)
    g = torch.from_numpy(np.ascontiguousarray(grads, dtype=np.float64))
    y = torch.ops.aten.upsample_linear1d_backward(
        g,
        [int(lo)],
        [int(input_size[0]), int(input_size[1]), int(input_size[2])],
        bool(align_corners),
        _scales_arg(scale),
    )
    return y.numpy()


class ResizeLinearGradTestSpec:
    """TestSpec golden for ResizeLinearGrad."""

    @staticmethod
    def golden(grads, original_image, *, align_corners=False, scale=0.0, **kwargs):
        grads = np.ascontiguousarray(grads)
        original_image = np.ascontiguousarray(original_image)
        assert grads.ndim == 3 and original_image.ndim == 3, (
            "ResizeLinearGrad expects 3-D NCL tensors"
        )
        verdict = _zero_axis_verdict(
            original_image.shape[0],
            original_image.shape[1],
            original_image.shape[2],
            grads.shape[2],
        )
        if verdict == "reject":
            return [None]  # 见 _ZERO_SPATIAL_REJECT
        if verdict == "empty":
            return [np.zeros(original_image.shape, dtype=original_image.dtype)]
        y = _linear_grad(grads, original_image.shape, bool(align_corners), scale)
        return [y.astype(original_image.dtype, copy=False)]

    class ThirdPartyImpl:
        """``torch.ops.aten.upsample_linear1d_backward`` 三方标杆。

        ResizeLinearGrad 的 half_pixel 由 ``align_corners`` 完全确定
        (kernel ``isHalfPiex_ = alignCorners ? 0 : 1``), 与 ATen 的单个
        ``align_corners`` 开关一一对应, 故两种 ac 取值都可表达。

        **显式正 scale 已纳入合法域**: 入口从 ``F.interpolate`` + autograd 换成
        底层 ``aten.upsample_linear1d_backward(grad_output, output_size,
        input_size, align_corners, scales)``。``F.interpolate`` 的
        ``size``/``scale_factor`` 互斥与 ``recompute_scale_factor`` 重算属
        **包装层**限制, ATen 低阶反向算子**同时接受 output_size / input_size
        与 scales**, 且 scales 真参与坐标映射(实测 4→3 显式 scale=2 与
        size-derived 结果不同 ⇒ 生效) —— 与 IR 语义一致。

        该算子就是前向的精确转置(ATen 内部同一套坐标与权重), 故不再绕
        autograd: 少一层包装, 且显式 scales 可直通。

        ``align_corners=True`` 时 ATen 忽略 scales —— kernel / aclnn / ATen
        三方一致, 直通不特判。

        **仍然拒绝的象限(有据, 非回避)**: ``L_src == L_dst`` 且给了显式正 scale
        —— ATen 对同尺寸 early-exit 返回拷贝并忽略 scales, 复现同一 early-exit
        的标杆无交叉校验价值; 该象限由文件内 golden 覆盖。

        布局: 1D 是 NCL/ND, 直接对应 ATen 的 3-D 输入, 无需转置。
        """

        def __init__(self, *, align_corners=False, scale=0.0, device=None, **kwargs):
            # 阶段 1: 参数准备 / 可用性判定 (不触碰数据, 不迁移设备)
            self.align_corners = bool(align_corners)
            # ac=True 时 ATen 与 kernel/aclnn 一致忽略 scales
            self.scale = None if self.align_corners else _scales_arg(scale)
            # device 必须**显式声明为形参**才会被注入: ttk 的
            # bind_params 只对 _FRAMEWORK_RESERVED={"device"} 这个**形参名**
            # 注入设备, 它不在 inputs/attrs 里, 因此 `kwargs.get("device")`
            # 恒为空 —— 实测会静默退回 cpu, 配置的设备根本不生效。
            self.device = device or "cpu"

        def _prepare(self, t):
            # 阶段 2: 设备迁移 (1D NCL 无需布局归一)
            v = t if torch.is_tensor(t) else torch.from_numpy(np.ascontiguousarray(t))
            return v.to(self.device).contiguous()

        def __call__(self, grads, original_image, **kwargs):
            # 阶段 3: 计算 (ATen 低阶反向, 即前向的精确转置)
            g = self._prepare(grads)
            img = self._prepare(original_image)
            if self.scale is not None and int(g.shape[-1]) == int(img.shape[-1]):
                raise NotImplementedError(
                    "L_src == L_dst 且给了显式正 scale: ATen upsample_linear1d_backward "
                    "对同尺寸 early-exit 返回拷贝并忽略 scales, 无法作为该象限的独立"
                    "标杆; 由文件内 golden 覆盖"
                )
            y = torch.ops.aten.upsample_linear1d_backward(
                g,
                [int(g.shape[-1])],
                list(int(v) for v in img.shape),
                self.align_corners,
                self.scale,
            )
            return [y]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


# ---------------------------------------------------------------- ACLNN 入口
_ONE_D_FORMATS = ("NCL", "ND")


def _opt_scale(v):
    """aclnn OUT/IN scale → ATen Optional[float]; <=0 表示未指定。

    直通不反转: ACLNN 边界恒为 OUT/IN, ATen ``compute_scales_value_backwards``
    同样按 ``scale > 0 → 1/scale`` 解读, 架构差异在 ``GetWorkspaceSize`` 内部
    消化。

    反向的"未指定"哨兵在 aclnn 侧是 0.0 (``ComputeCheckScale`` 仅当
    ``std::abs(scales) > 1e-9`` 时才校验 ``floor(inL*scales)==outL``, 0.0 合法);
    前向的哨兵是负数(判据 ``scale >= 0``, 0.0 会被判冲突)。对负数与 0 都返回
    None 即兼容两者。
    """
    if v is None:
        return None
    s = float(v.reshape(-1)[0]) if torch.is_tensor(v) else float(v)
    return s if s > 0.0 else None


def _aclnn_format(kwargs):
    """aclnn / e2e 路径的首输入 format。两条通路的键名**不同**, 必须都读:

      * ``input_formats`` —— 远端三方执行注入(``executor._compose_kwargs``,
        仅类形态实现), 即 ``third_party`` provider 能看到的唯一 format 来源;
      * ``tensor_formats`` —— 本地 golden 调用注入
        (``ttk/core_modules/aclnn/golden.py``), provider 侧**拿不到**。

    只读 ``tensor_formats`` 会让 provider 侧的布局校验恒按缺省 ND 通过, 非法
    布局用例被静默放行。
    """
    for key in ("input_formats", "tensor_formats"):
        fmts = kwargs.get(key)
        if not fmts:
            continue
        v = fmts if isinstance(fmts, str) else (fmts[0] if len(fmts) > 0 else None)
        if v:
            return str(v).strip().upper()
    return "ND"


# ----------------------------------------------- third_party 共享数学核 (1D)
#
# ACLNN / E2E 两个入口的 third_party 共用下面两个核, 只是形参名不同
# (aclnn 是 camel ``gradOut``/``outputSize``/``inputSize``/``alignCorners``/
# ``scales``, E2E 是 ATen snake ``grad_output``/``output_size``/``input_size``/
# ``align_corners``/``scales``) —— 需求允许共享数学核, 但**声明必须各按自己的
# 签名**, 否则 ttk 的 bind_params 会以 UnknownParamError 拒绝。
#
# ``device`` 必须**显式声明为形参**才会被注入: bind_params 只对
# _FRAMEWORK_RESERVED={"device"} 这个形参名注入, 它不在 inputs/attrs 里,
# 所以 ``kwargs.get("device")`` 恒为空(实测会静默退回 cpu)。


def _tp_linear_grad_aten(grads, input_size, align_corners, scale, device):
    """ATen 低阶 1D 线性反向。与文件内 golden 同源, 见 ``_tp_linear_aten``。"""
    g = (
        grads
        if torch.is_tensor(grads)
        else torch.from_numpy(np.ascontiguousarray(grads))
    )
    g = g.to(device or "cpu")
    orig = g.dtype
    if orig in (torch.float16, torch.bfloat16):
        g = g.to(torch.float32)
    y = torch.ops.aten.upsample_linear1d_backward(
        g.contiguous(),
        [int(g.shape[-1])],
        [int(v) for v in input_size],
        bool(align_corners),
        scale,
    )
    if orig in (torch.float16, torch.bfloat16):
        y = y.to(orig)
    return y


class AclnnUpsampleLinear1dBackwardTestSpec:
    """aclnnUpsampleLinear1dBackward(gradOut, outputSize, inputSize, alignCorners,
    scales, out).

    单分支 ``aten::upsample_linear1d_backward`` —— aclnn 的坐标语义被
    ``alignCorners`` 完全确定, ATen 原生覆盖, 不引入自实现分支。

    * 形参名严格对齐 C 头 ``aclnnUpsampleLinear1dBackwardGetWorkspaceSize``:
      反向是**复数 ``scales``**(前向才是单数 ``scale``)。
    * scales: ``<=0 → None``(ATen 退回 in/out); ``alignCorners=True`` 时统一传
      None —— ATen / kernel / aclnn 三方一致忽略 scales。
    * dtype: 入参抬 fp64 计算(ATen CPU 原生支持 double, 取最高精度标杆),
      结果回原 dtype。``--golden-mode`` 默认不抬精度, 抬升必须在入口内部完成。
    * 布局: 1D 是 NCL/ND, ``gradOut`` 与 ``inputSize``(3 元 {N,C,Li}) 都直接
      对应 ATen 1D 接口, 无需转置/维序重排(2D 反向才需要把 {N,H,W,C} 重排成
      {N,C,H,W}); 仍读 ``tensor_formats`` 做校验, 出现 channels-last 标记即报错。
    * **不抄 kernel 的两处捷径**:
      - 同尺寸 ALL_COPY: 不需要。ATen 同尺寸反向本身就精确恒等 —— 实测
        (1,2,8)/(2,3,6)/(1,4,1)/(2,8,4096)/(1,1,5) × ac∈{F,T} × scales∈{None,1.0}
        共 20 组 fp64 maxdiff 全为 0 且与非均匀上游梯度逐位相等(torch 2.7.1 CPU),
        fp32/fp16/bf16 抬 fp64 回原 dtype 后同样逐位相等。
      - "同尺寸忽略 scale": 在 aclnn 侧不可达 —— ``ComputeCheckScale`` 对非 0
        scales 强校验 ``floor(inL*scales) == outL``, 矛盾的 scales 直接被拒,
        属未经验证的 kernel 实现细节。
    """

    @staticmethod
    def golden(
        gradOut, outputSize, inputSize, alignCorners, scales, out=None, **kwargs
    ):
        fmt = _aclnn_format(kwargs)
        assert fmt in _ONE_D_FORMATS, (
            "aclnnUpsampleLinear1dBackward is a 3-D NCL/ND interface, "
            f"got format {fmt!r}"
        )
        g = (
            gradOut.detach().cpu()
            if torch.is_tensor(gradOut)
            else torch.as_tensor(gradOut)
        )
        orig_dtype = g.dtype
        assert g.dim() == 3, (
            "aclnnUpsampleLinear1dBackward expects a 3-D NCL gradOut, "
            f"got shape {tuple(g.shape)}"
        )
        g = g.contiguous().to(torch.float64)

        osize = [int(v) for v in np.asarray(outputSize).reshape(-1)]
        isize = [int(v) for v in np.asarray(inputSize).reshape(-1)]
        assert len(osize) >= 1, "outputSize must carry [L_out]"
        assert len(isize) == 3, "inputSize must carry 3 dims [N, C, L_in]"

        ac = bool(alignCorners)
        s = None if ac else _opt_scale(scales)

        verdict = _zero_axis_verdict(isize[0], isize[1], isize[2], int(g.shape[2]))
        if verdict == "reject":
            return [None]  # 见 _ZERO_SPATIAL_REJECT
        if verdict == "empty":
            return [torch.zeros(tuple(isize), dtype=orig_dtype)]

        y = torch.ops.aten.upsample_linear1d_backward(g, osize[:1], isize, ac, s)
        return [y.contiguous().to(orig_dtype)]

    class ThirdPartyImpl:
        """ATen 适配器, 形参按 **aclnn C 头**命名(camel, 反向是复数 ``scales``)。

        数学核与文件内 golden 同源, 覆盖参数适配/绑定路径。按算子原型本入口
        仅对标 PyTorch, 故不设 TF 同类; 独立数值参考为文件内 fp64 golden。
        """

        def __init__(self, *, device=None, **kwargs):
            self.device = device or "cpu"

        def __call__(
            self, gradOut, outputSize, inputSize, alignCorners, scales, **kwargs
        ):
            isz = [int(v) for v in np.asarray(inputSize).reshape(-1)]
            ac = bool(alignCorners)
            return [
                _tp_linear_grad_aten(
                    gradOut, isz, ac, None if ac else _opt_scale(scales), self.device
                )
            ]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


# ------------------------------------------------------- Torch E2E (框架) 入口
class E2eUpsampleLinear1dBackwardTestSpec:
    """``torch.ops.aten.upsample_linear1d_backward`` 的框架 E2E 入口 (NCL)。

    被测对象是**框架调用链**: ttk 把上游梯度搬到 ``npu:0`` 后调用
    ``torch.ops.aten.upsample_linear1d_backward``, 经 torch_npu 适配层落到
    ``aclnnUpsampleLinear1dBackward`` → 设备 kernel ``ResizeLinearGrad``。
    本 golden 只在 **CPU** 上用 fp64 算参考值, 与被测路径完全分离。

    **重载**: 绑定 ``.default``
    ``aten::upsample_linear1d_backward(Tensor grad_output, SymInt[1]
    output_size, SymInt[3] input_size, bool align_corners, float?
    scales=None) -> Tensor``。

    * **绝不绑 ``.grad_input``**: 该重载在本版本 **PrivateUse1 无 kernel**,
      会**静默回落 CPU** —— 跑过也不构成 NPU 取证。ttk 的 ``match_overload``
      在 params 不含 ``grad_input`` 时选 ``oidx 0``(即 ``.default``), 故 csv
      **不得**出现 ``grad_input`` 键。
    * **不注册 ``.vec``**: composite + PrivateUse1 无 kernel, 会被分解。
    * **形参名**: 反向的 ATen 形参是**复数 ``scales``**, 与 aclnn 反向同名(前向
      才是单数 ``scale``)。csv 按 ATen schema 写 ``scales``。
    * **``input_size`` 为 3 元 ``[N, C, L_in]``**: 1-D 无维序重排问题(2-D 反向
      在 NHWC 下才需把 ``{N,H,W,C}`` 重排成 ``{N,C,H,W}``)。csv 的
      ``tensor_view_shapes`` 第二元必须与 ``input_size`` 一致 —— 被测侧据此
      分配输出, 不一致会直接报 shape 错而非静默算错。
    * **上游梯度是真实非均匀随机值**(``input_data_ranges`` 为 ``(-3.0, 5.0)``),
      不是全 1 / 常量 —— 常量上游梯度会让散射错误相互掩盖。
    * **dtype**: 入参抬 fp64 计算后回原 dtype(``--golden-mode Promote`` 对 e2e
      未实现)。与本文件 aclnn 入口同口径。
    * tolerance 沿用 kernel/aclnn 入口同值, 无放宽理由。
    """

    @staticmethod
    def golden(
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales=None,
        out=None,
        **kwargs,
    ):
        g = (
            grad_output.detach().cpu()
            if torch.is_tensor(grad_output)
            else torch.as_tensor(grad_output)
        )
        orig_dtype = g.dtype
        assert g.dim() == 3, (
            f"aten.upsample_linear1d_backward expects a 3-D NCL grad_output, "
            f"got {tuple(g.shape)}"
        )
        g = g.contiguous().to(torch.float64)

        osize = [int(v) for v in np.asarray(output_size).reshape(-1)]
        isize = [int(v) for v in np.asarray(input_size).reshape(-1)]
        assert len(osize) >= 1, "output_size must carry [L_out]"
        assert len(isize) == 3, "input_size must carry 3 dims [N, C, L_in]"

        verdict = _zero_axis_verdict(isize[0], isize[1], isize[2], int(g.shape[2]))
        if verdict == "reject":
            return [None]  # 见 _ZERO_SPATIAL_REJECT
        if verdict == "empty":
            return [torch.zeros(tuple(isize), dtype=orig_dtype)]

        y = torch.ops.aten.upsample_linear1d_backward(
            g, osize[:1], isize, bool(align_corners), _opt_scale(scales)
        )
        return [y.contiguous().to(orig_dtype)]

    class ThirdPartyImpl:
        """ATen 适配器, 形参按 **ATen schema** 命名(snake, 反向是复数 ``scales``)。

        数学核与文件内 golden 同源, 覆盖参数适配/绑定路径。按算子原型本入口
        仅对标 PyTorch, 故不设 TF 同类; 独立数值参考为文件内 fp64 golden。
        """

        def __init__(self, *, device=None, **kwargs):
            self.device = device or "cpu"

        def __call__(
            self,
            grad_output,
            output_size,
            input_size,
            align_corners,
            scales=None,
            **kwargs,
        ):
            isz = [int(v) for v in np.asarray(input_size).reshape(-1)]
            ac = bool(align_corners)
            return [
                _tp_linear_grad_aten(
                    grad_output,
                    isz,
                    ac,
                    None if ac else _opt_scale(scales),
                    self.device,
                )
            ]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
