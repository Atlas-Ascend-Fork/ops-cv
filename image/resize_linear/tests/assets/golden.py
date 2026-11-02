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

"""TTK golden for ResizeLinear (1D linear forward, NCL) + aclnnUpsampleLinear1d.

标杆标准: ops-precision-standard 浮点计算类 · 单标杆比对 · 优先级-1 高精度
CPU 实现 (PyTorch CPU aten 原生). 详见仓外 BENCHMARK_STANDARD.md.

两个入口:
  * ``resize_linear``          —— kernel 路径 (Ascend IR, 属性 size/scale)
  * ``aclnnUpsampleLinear1d``  —— ACLNN 路径 (ATen upsample_linear1d 语义)

正式 golden 直接调用 ``torch.ops.aten.upsample_linear1d`` —— PyTorch 原生 1D
线性插值高精度 CPU 实现, 不复刻 kernel、不依赖被测算子。

分支表 (kernel 入口, 单分支):
    条件      : 全部 (Linear 隐含 half_pixel = not align_corners)
    接口      : torch.ops.aten.upsample_linear1d(x_NCL, [Lo], align_corners, scales)
    scales    : CANN ``scale`` 属性>0 时映射到 torch scales(=scale_attr); 否则 None
    布局      : NCL 原生, 无需转置
    dtype     : float64 计算 → 回原 dtype
    实证偏差  : ≤4.44e-16 (含 src=1 / dst=1 / 同尺寸 / 非整数缩放)
    ALL_COPY  : L_out==L_in → 直返输入, 对齐 kernel schId-1(DIM_1, 忽略 scale)

torch align_corners=False 恒采用 half_pixel 坐标, 与 Linear kernel
``isHalfPiex_ = alignCorners ? 0 : 1`` 一致, 故无需独立 half_pixel 开关。
不捕获异常静默回退: torch 调用失败即直接抛出。
"""

import numpy as np
import torch

__spec__ = {
    "resize_linear": "ResizeLinearTestSpec",
    "aclnnUpsampleLinear1d": "AclnnUpsampleLinear1dTestSpec",
    "torch.ops.aten.upsample_linear1d": "E2eUpsampleLinear1dTestSpec",
}


# ttk 的 XPU 绑定器按**被测 API 自己的参数名**匹配形参
# (``ttk.core_modules.common.xpu.bind_params``), 而 aclnnUpsampleLinear1d 的
# C 头首参名与 ATen schema 首参名都是 ``self``。所以三方 provider 必须把该形参
# **逐字**声明为 ``self``、并把实例槽改名为 ``impl`` —— 写成 ``x`` 会在计算
# 开始前就抛 ``UnknownParamError: parameter 'x' ... has no value``, 整个三方
# 标杆静默不执行。本仓已合入的 ``image/resize_nearest_neighbor_v2`` /
# ``image/upsample_nearest3d`` 同此写法。


def _scales_arg(scale):
    """CANN scale 属性 → torch scales; <=0/缺省返回 None(用 size 隐含 src/dst)。"""
    if scale is None:
        return None
    try:
        v = float(scale)
    except (TypeError, ValueError):
        return None
    return v if v > 0.0 else None


def _linear_forward(x, lo, align_corners, scale):
    """PyTorch CPU 高精度标杆: aten.upsample_linear1d, float64 计算。"""
    li = int(x.shape[2])
    # kernel ComputeKey schId-1(DIM_1): L_out==L_in 时无条件纯搬运,
    # 忽略 scale / align_corners(resize_linear_tiling_arch35.cpp:73-74)。
    if int(lo) == li:
        return np.ascontiguousarray(x, dtype=np.float64)
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))
    y = torch.ops.aten.upsample_linear1d(
        t, [int(lo)], bool(align_corners), _scales_arg(scale)
    )
    return y.numpy()


class ResizeLinearTestSpec:
    """TestSpec golden for ResizeLinear."""

    @staticmethod
    def golden(x, size, *, align_corners=False, scale=0.0, **kwargs):
        x = np.ascontiguousarray(x)
        assert x.ndim == 3, "ResizeLinear expects a 3-D NCL input"
        dst = np.asarray(size).reshape(-1)
        assert dst.size >= 1, "size input must carry [L_out]"
        lo = int(dst[0])
        y = _linear_forward(x, lo, bool(align_corners), scale)
        return [y.astype(x.dtype, copy=False)]

    class ThirdPartyImpl:
        """``torch.ops.aten.upsample_linear1d`` 三方标杆。

        ResizeLinear 的 half_pixel 由 ``align_corners`` 完全确定
        (kernel ``isHalfPiex_ = alignCorners ? 0 : 1``), 与 ATen 的单个
        ``align_corners`` 开关一一对应, 故两种 ac 取值都可表达。

        **显式正 scale 已纳入合法域**: 入口从 ``F.interpolate`` 换成底层
        ``aten.upsample_linear1d(self, output_size, align_corners, scales)``。
        ``F.interpolate`` 的 ``size``/``scale_factor`` 互斥与
        ``recompute_scale_factor`` 重算属**包装层**限制, ATen 低阶算子**同时
        接受 output_size 与 scales**, 且 scales 真参与坐标映射 —— 与 IR 的
        "scale>0 → 坐标尺度取 1/scale, 输出尺寸由 size 输入独立给定" 完全一致
        (实测 3→4 显式 scale=2 得 ``[1, 1.25, 1.75, 2.25]``, 与 size-derived
        的 ``[1, 1.625, 2.375, 3]`` 不同 ⇒ scales 生效)。

        ``align_corners=True`` 时 ATen 的 ``area_pixel_compute_scale`` 直接
        忽略 scales —— kernel / aclnn / ATen 三方一致, 故直通不特判。

        **仍然拒绝的象限(有据, 非回避)**: ``L_in == L_out`` 且给了显式正 scale。
        ATen 对同尺寸有 early-exit 直接返回拷贝、忽略 scales(实测
        ``upsample_linear1d(3→3, scales=2.0)`` 返回原值, 而按坐标映射应随
        scale 变化), 复现同一 early-exit 的标杆无交叉校验价值; 该象限由文件内
        golden 覆盖。

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

        def _prepare(self, x):
            # 阶段 2: 设备迁移 (1D NCL 无需布局归一)
            t = x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x))
            return t.to(self.device).contiguous()

        def __call__(self, x, size, **kwargs):
            # 阶段 3: 计算
            dst = np.asarray(size).reshape(-1)
            t = self._prepare(x)
            lo = int(dst[0])
            if self.scale is not None and int(t.shape[-1]) == lo:
                raise NotImplementedError(
                    "L_in == L_out 且给了显式正 scale: ATen upsample_linear1d 对同尺寸 "
                    "early-exit 返回拷贝并忽略 scales(实测 3->3 scale=2.0 原值返回), "
                    "无法作为该象限的独立标杆; 由文件内 golden 覆盖"
                )
            y = torch.ops.aten.upsample_linear1d(
                t, [lo], self.align_corners, self.scale
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

    直通不反转: ACLNN 边界恒为 OUT/IN, ATen ``compute_scales_value`` 同样按
    ``scale > 0 → 1/scale`` 解读, 架构差异在 ``GetWorkspaceSize`` 内部消化。

    ``aclnnUpsampleLinear1d`` 的"未指定"哨兵在 aclnn 侧**必须是负数**, 这是七个
    入口里唯一的例外, csv 用 -1.0。依据
    ``image/upsample_linear1d/op_host/op_api/aclnn_upsample_linear_1d.cpp``:

      * ``CheckScale`` 只要求 ``std::isfinite(scale)`` —— 负数合法;
      * ``CheckShape`` 的冲突校验闸门是 ``if (scale >= 0)``, 命中后要求
        ``int64(inL * scale) == outL`` —— 所以 ``0.0`` 会被当成"已指定 0 倍"
        并以 "scale conflicts with outputSize" **报错**;
      * 真正消费 scale 的 ``CheckLinear1dScales`` 用的是 ``scale > 0``,
        即 ``<=0`` 一律按"未指定, 退回 in/out"处理。

    即只有 ``0.0`` 这一个值既被当作"已指定"又必然冲突。其余六个入口的闸门是
    ``scales <= 0 → 视为未指定``(bicubic/AA/三个反向)或显式拒绝负数
    (``aclnnUpsampleBilinear2d::CheckScalesValid``), 故那些用 0.0。
    本函数对负数与 0 都返回 None, 两种哨兵都兼容。
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
# (aclnn 是 camel ``outputSize``/``alignCorners``/``scale``, E2E 是 ATen
# snake ``output_size``/``align_corners``/``scales``) —— 需求允许共享数学核,
# 但**声明必须各按自己的签名**, 否则 ttk 的 bind_params 会以
# UnknownParamError 拒绝(参数名不在 inputs ∪ attrs 内)。
#
# ``device`` 必须**显式声明为形参**才会被注入: bind_params 只对
# _FRAMEWORK_RESERVED={"device"} 这个形参名注入, 它不在 inputs/attrs 里,
# 所以 ``kwargs.get("device")`` 恒为空(实测会静默退回 cpu)。


def _tp_linear_aten(x, lo, align_corners, scale, device):
    """ATen 低阶 1D 线性。与文件内 golden **同源**(都走 aten.upsample_linear1d)。

    交叉校验价值: 覆盖的是**参数适配与绑定路径**(camel/snake 形参名、scale
    哨兵、osize 切片、dtype 处理), 数学与 golden 同源, 不构成独立数值标杆。
    不设 TF provider: 算子原型 ``op_graph/resize_linear_proto.h`` 声明
    "Compatible with PyTorch upsample_linear1d operator.", 仅对标 PyTorch,
    独立数值参考由文件内 fp64 golden 承担。
    """
    t = x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x))
    t = t.to(device or "cpu")
    if t.dtype in (torch.float16, torch.bfloat16):
        t = t.to(torch.float32)
    y = torch.ops.aten.upsample_linear1d(
        t.contiguous(), [int(lo)], bool(align_corners), scale
    )
    if torch.is_tensor(x) and x.dtype in (torch.float16, torch.bfloat16):
        y = y.to(x.dtype)
    return y


class AclnnUpsampleLinear1dTestSpec:
    """aclnnUpsampleLinear1d(self, outputSize, alignCorners, scale, out).

    单分支 ``aten::upsample_linear1d`` —— aclnn 的坐标语义被 ``alignCorners``
    完全确定(True → align_corners, False → half_pixel), ATen 原生覆盖, 不引入
    自实现分支。

    * 形参名严格对齐 C 头 ``aclnnUpsampleLinear1dGetWorkspaceSize``: 前向是
      **单数 ``scale``**(反向才是复数 ``scales``)。framework 自动 torch 映射会把
      camel ``scale`` 转成 snake ``scale`` 后喂给 ATen 的 ``scales`` 形参从而
      schema 不匹配(notes/PROBE.md: `failed to match any schema`), 这正是本入口
      必须显式注册的原因。
    * scales: ``<=0 → None``(ATen 退回 in/out); ``alignCorners=True`` 时统一传
      None —— ATen / kernel / aclnn 三方一致忽略 scales。
    * dtype: 入参抬 fp64 计算(ATen CPU 原生支持 double, 取最高精度标杆),
      结果回原 dtype。``--golden-mode`` 默认不抬精度, 抬升必须在入口内部完成。
    * 布局: 1D 是 NCL/ND, 直接对应 ATen 1D 接口, 无需转置; 仍读
      ``tensor_formats`` 做校验, 出现 2D/channels-last 标记即报错而不是误算。
    * **不抄 kernel 的两处捷径**:
      - 同尺寸 ALL_COPY: 不需要。ATen 同尺寸本身就精确恒等 —— 实测
        (1,2,8)/(2,3,6)/(1,4,1)/(2,8,4096)/(1,1,5) × ac∈{F,T} × scale∈{None,1.0}
        共 20 组 fp64 maxdiff 全为 0 且逐位相等(torch 2.7.1 CPU),
        fp32/fp16/bf16 抬 fp64 回原 dtype 后同样逐位相等。
      - "同尺寸忽略 scale": 在 aclnn 侧不可达 —— ``CheckShape`` 强校验
        ``floor(inL*scale) == outL``, 矛盾的 scale 直接被拒,
        属未经验证的 kernel 实现细节。
    """

    @staticmethod
    def golden(x, outputSize, alignCorners, scale, out=None, **kwargs):
        fmt = _aclnn_format(kwargs)
        assert fmt in _ONE_D_FORMATS, (
            f"aclnnUpsampleLinear1d is a 3-D NCL/ND interface, got format {fmt!r}"
        )
        t = x.detach().cpu() if torch.is_tensor(x) else torch.as_tensor(x)
        orig_dtype = t.dtype
        assert t.dim() == 3, (
            f"aclnnUpsampleLinear1d expects a 3-D NCL tensor, got shape {tuple(t.shape)}"
        )
        t = t.contiguous().to(torch.float64)

        osize = [int(v) for v in np.asarray(outputSize).reshape(-1)]
        assert len(osize) >= 1, "outputSize must carry [L_out]"
        ac = bool(alignCorners)
        s = None if ac else _opt_scale(scale)

        y = torch.ops.aten.upsample_linear1d(t, osize[:1], ac, s)
        return [y.contiguous().to(orig_dtype)]

    class ThirdPartyImpl:
        """ATen 适配器, 形参按 **aclnn C 头**命名(camel, 前向是单数 ``scale``)。

        数学核与文件内 golden 同源, 覆盖参数适配/绑定路径。按算子原型本入口
        仅对标 PyTorch, 故不设 TF 同类; 独立数值参考为文件内 fp64 golden。
        """

        def __init__(self, *, device=None, **kwargs):
            self.device = device or "cpu"

        def __call__(impl, self, outputSize, alignCorners, scale, **kwargs):
            osize = [int(v) for v in np.asarray(outputSize).reshape(-1)]
            ac = bool(alignCorners)
            return [
                _tp_linear_aten(
                    self, osize[0], ac, None if ac else _opt_scale(scale), impl.device
                )
            ]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


# ------------------------------------------------------- Torch E2E (框架) 入口
class E2eUpsampleLinear1dTestSpec:
    """``torch.ops.aten.upsample_linear1d`` 的框架 E2E 入口 (NCL)。

    被测对象是**框架调用链**: ttk 把输入搬到 ``npu:0`` 后调用
    ``torch.ops.aten.upsample_linear1d``, 经 torch_npu 适配层落到
    ``aclnnUpsampleLinear1d`` → 设备 kernel ``ResizeLinear``。
    本 golden 只在 **CPU** 上用 fp64 算参考值, 与被测路径完全分离。

    **重载**: 绑定 ``.default``
    ``aten::upsample_linear1d(Tensor self, SymInt[1] output_size, bool
    align_corners, float? scales=None) -> Tensor``。

    * **不注册 ``.vec``**: ``.vec`` 是 ``CompositeImplicitAutograd`` 且
      PrivateUse1 无 kernel, 会被分解, **根本不落目标 kernel**。
    * **不传 ``out``**: ``.out`` 重载虽有 PrivateUse1 kernel, 但走显式 out 形参
      会改变重载选择; 保持 ``.default`` 以确保落到被验证的实现。
      (反向的 ``.grad_input`` 重载更是 **CPU fallback**, 见 resize_linear_grad。)
    * **形参名**: 1-D 的 ATen 形参是**复数 ``scales``**, 而 aclnn 前向是单数
      ``scale`` —— 两者不同名, csv ``attributes`` 必须按 ATen schema 写
      ``scales``。框架按**位置**绑定, 形参名仅作可读性。
    * **scales 哨兵**: ATen 的 ``scale <= 0`` 与 ``None`` 等价(实测
      ``scales=-1.0``/``0.0`` 与 ``None`` 结果逐位相同, 内部按 out/in 比例计算),
      故复用 ``_opt_scale``; csv 直接写 ``None`` 表示未指定, 写正值表示显式指定。
    * **dtype**: 入参抬 fp64 计算后回原 dtype —— ``--golden-mode Promote`` 对
      e2e 路径未实现, 抬升必须在入口内部完成。
    * **不抄 kernel 的 ALL_COPY 捷径**: ATen 同尺寸本身精确恒等(见 aclnn 入口
      docstring 的 20 组实证), 无需特判。
    * tolerance 沿用 kernel/aclnn 入口同值: 同一数学语义、同一后端精度, 无放宽理由。
    """

    @staticmethod
    def golden(x, output_size, align_corners, scales=None, out=None, **kwargs):
        t = x.detach().cpu() if torch.is_tensor(x) else torch.as_tensor(x)
        orig_dtype = t.dtype
        assert t.dim() == 3, (
            f"aten.upsample_linear1d expects a 3-D NCL tensor, got {tuple(t.shape)}"
        )
        t = t.contiguous().to(torch.float64)

        osize = [int(v) for v in np.asarray(output_size).reshape(-1)]
        assert len(osize) >= 1, "output_size must carry [L_out]"

        y = torch.ops.aten.upsample_linear1d(
            t, osize[:1], bool(align_corners), _opt_scale(scales)
        )
        return [y.contiguous().to(orig_dtype)]

    class ThirdPartyImpl:
        """ATen 适配器, 形参按 **ATen schema** 命名(snake, 1D 前向是复数
        ``scales``)。数学核与文件内 golden 同源, 覆盖参数适配/绑定路径。
        按算子原型本入口仅对标 PyTorch, 故不设 TF 同类; 独立数值参考为
        文件内 fp64 golden。
        """

        def __init__(self, *, device=None, **kwargs):
            self.device = device or "cpu"

        def __call__(impl, self, output_size, align_corners, scales=None, **kwargs):
            osize = [int(v) for v in np.asarray(output_size).reshape(-1)]
            ac = bool(align_corners)
            return [
                _tp_linear_aten(
                    self, osize[0], ac, None if ac else _opt_scale(scales), impl.device
                )
            ]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
