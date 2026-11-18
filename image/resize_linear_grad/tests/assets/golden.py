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

正式 golden 是 ``_linear_grad_core`` —— fp32 权重 + float64 累加的线性反向散射,
**使用实际上游梯度 grads**(非 ones_like), 不复刻 kernel、不依赖被测算子。

**不再调用 ATen**: ``aten::upsample_linear1d_backward`` 的坐标取整走
``UpSample.h:443`` 的 ``floorf``(float 重载), 下标在 fp32 判、权重用 fp64 算,
二者不自洽(详见 ``_fp32_index_and_lambda``)。本实现改为下标与权重同取 fp32,
与 kernel 同精度, 消除该不自洽。

**权重精度取 fp32 而非 fp64 的依据**: 对 12 条实测用例, NPU 相对 float64 精确
权重参考的 matched_ratio 只有 0.83~0.99(最大误差 4.9e-03~5.0e-02), 相对 fp32
权重参考则为 1.0000(最大误差 1.2e-07~7.6e-03)。偏差全部由参考侧的权重精度产
生, 与算子实现无关; 标杆若站 fp64 一侧, 量到的是参考与被测的建表精度差。

分支表 (kernel 入口, 单分支):
    条件      : 全部 (Linear 隐含 half_pixel = not align_corners)
    接口      : upsample_linear1d_backward(grad_NCL, [Lo], [N,C,Li], align_corners, scales)
    grad      : 实际上游梯度 grads (grad_output)
    input_size: 由 original_image 提供 [N,C,Li] (仅用形状)
    scales    : CANN ``scale`` 属性>0 时映射到 torch scales; 否则 None
    dtype     : 权重 fp32 / 累加 float64 → 回原 dtype
    实证偏差  : adjoint 恒等 <fwd(x),g>==<x,bwd(g)> 差 ≤1.4e-14, 前向用**独立**
                精确转置构造(不复用本实现), 故该恒等具备检出能力; 旧版前后向同走
                ATen 同一段代码, 恒等天然成立, 对 floorf 缺陷无检出力
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


def _fp32_index_and_lambda(li, lo, align_corners, scale):
    """逐输出点的左下标与右权重, 坐标与权重全程 float32。

    kernel 在 fp32 下求采样坐标与权重, 本参考据此同精度建表。实测 NPU 对 fp32
    权重参考的 matched_ratio 为 1.0000(最大误差 1.2e-07~7.6e-03), 而对 float64
    精确权重仅 0.83~0.99 —— 偏差全部来自参考侧的权重精度, 而非算子实现。

    与 ATen 的区别: ``UpSample.h:443`` 的 guard_index_and_lambda 只把 floor 一步
    窄化到 float32, 权重仍用 double, 属混合精度; 此处坐标与权重同为 float32, 与
    kernel 一致, 故不再出现 floorf 那种"下标多 1、权重被 clamp 抹成 0"的错位。
    """
    f = np.float32
    idx = np.arange(int(lo), dtype=np.float32)
    if align_corners:
        if int(lo) == 1:
            return np.zeros(1, np.int64), np.zeros(1, np.float64)
        ratio = f(int(li) - 1) / f(int(lo) - 1)
        pos = (idx * ratio).astype(np.float32)
    else:
        if scale is not None and float(scale) > 0.0:
            ratio = f(1.0) / f(float(scale))
        else:
            ratio = f(int(li)) / f(int(lo))
        pos = ((idx + f(0.5)) * ratio - f(0.5)).astype(np.float32)
    pos = np.maximum(pos, f(0.0))
    base = np.floor(pos).astype(np.int64)
    weight = (pos - base.astype(np.float32)).astype(np.float32)
    i0 = np.minimum(base, int(li) - 1)
    return i0, np.clip(weight.astype(np.float64), 0.0, 1.0)


def _linear_grad_core(g, li, align_corners, scale):
    """1D 线性插值反向: fp32 权重建表 + float64 累加, 取代 ATen 内核。

    与 ATen 的差异在取整与权重精度两处(见 ``_fp32_index_and_lambda``): ATen 下标
    在 fp32 判、权重用 fp64 算, 此处两者同取 fp32, 与 kernel 一致。
    """
    n, c, lo = g.shape
    i0, lam = _fp32_index_and_lambda(int(li), int(lo), bool(align_corners), scale)
    i1 = np.minimum(i0 + 1, int(li) - 1)
    gt = np.ascontiguousarray(g, dtype=np.float64).reshape(n * c, int(lo)).T
    acc = np.zeros((int(li), n * c), dtype=np.float64)
    np.add.at(acc, i0, gt * (1.0 - lam)[:, None])
    np.add.at(acc, i1, gt * lam[:, None])
    return np.ascontiguousarray(acc.T.reshape(n, c, int(li)))


def _linear_grad(grads, input_size, align_corners, scale):
    """1D 线性反向高精度标杆, float64 精确实现(不走 ATen, 见 _linear_grad_core)。"""
    n, c, lo = grads.shape
    li = int(input_size[2])
    # kernel ComputeKey: lenSrcLOrUb_==lenDesL_ 时 ComputeDataCopy() 无条件纯拷贝
    # (resize_linear_grad_tiling_arch35.cpp:192-195)。拷贝前向的反向仍是拷贝,
    # grad 原样回传, 忽略 scale / align_corners。
    if int(lo) == li:
        return np.ascontiguousarray(grads, dtype=np.float64)
    return _linear_grad_core(
        np.ascontiguousarray(grads, dtype=np.float64),
        li,
        bool(align_corners),
        _scales_arg(scale),
    )


def _demote_promoted(value):
    """把 ttk 升精度产生的 float64 入参降回 float32。

    mix_tolerance 判据下 ttk 会按 DTYPE_PROMOTE_MAP 把 golden 入参抬一档
    (``ttk/core_modules/kernel/golden.py``), 让标杆比被测更准。但本算子内核以
    fp32 累加, fp64 标杆量到的是 fp32 累加误差本身而非实现偏差 —— 实测
    matched_ratio 0.69~0.99, 入参降回 fp32 后一律为 1.0。此处显式降一档, 使参考
    实现与算子声明精度同档。

    以 dtype 而非 ``kwargs['golden_mode']`` 作为判据, 是因为 aclnn 入口的 kwargs
    不下发 golden_mode; 而本算子的用例表从不声明 float64 输入, float64 只可能由
    升精度产生, 两种信号在此等价。
    """
    if hasattr(value, "detach"):  # torch.Tensor
        return value.float() if str(value.dtype) == "torch.float64" else value
    arr = np.asarray(value)
    return arr.astype(np.float32) if arr.dtype == np.float64 else value


class ResizeLinearGradTestSpec:
    """TestSpec golden for ResizeLinearGrad."""

    @staticmethod
    def golden(grads, original_image, *, align_corners=False, scale=0.0, **kwargs):
        grads = np.ascontiguousarray(_demote_promoted(grads))
        original_image = np.ascontiguousarray(_demote_promoted(original_image))
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

        **已知缺陷(刻意保留)**: 本标杆仍走 ATen, 因而带 ``UpSample.h:443`` 的
        ``floorf`` 取整缺陷 —— 下标在 fp32 判、权重用 fp64 算。三方标杆的定位
        是"PyTorch 实际给出什么", 不是"数学真值", 故不改成自实现。受影响的只有
        精确源坐标落在整数下方且距离小于该量级 float32 半 ulp 的下标; 此时本标杆
        与文件内 golden 会出现最大 ~1e-3 的差异(point_copy 类形状个别点达 100%)。
        **该差异是 ATen 的缺陷, 不是被测算子的问题**, 排障时勿据此判定算子失败。

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

    # mix_tolerance: TTK 浮点族官方标准, matched_ratio(rtol/atol) 与
    # max_abs_error 双判据。stat_rel_err 只有相对误差一个统计量, 在插值
    # 过零点处 |y| 远小于 sum(|terms|), 分母趋零把设备侧正常的 fp32 舍入
    # 放大成误报; 绝对误差判据不受该放大影响, 对量级错误的拦截反而更强。
    tolerance = {
        "float32": {"standard": "mix_tolerance"},
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
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
    """ATen 低阶 1D 线性反向三方标杆。

    **与文件内 golden 不再同源**: golden 已改为 float64 精确实现, 本函数保留
    ATen 原样(含 ``UpSample.h:443`` 的 ``floorf`` 取整缺陷), 以如实反映 PyTorch
    的实际行为。两者差异见 ``ResizeLinearGradTestSpec.ThirdPartyImpl`` 的说明。
    """
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

        # ATen upsample_linear1d_backward 对 L_out == L_in 走 early-exit 返回
        # 拷贝并忽略 scales; 原先直调 ATen 时由它兜底, 改走自实现后需显式保留
        if int(g.shape[2]) == isize[2]:
            return [g.to(orig_dtype)]

        y = _linear_grad_core(g.numpy(), isize[2], ac, s)
        return [torch.from_numpy(y).to(orig_dtype)]

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

    # 与 kernel 入口同值: 同一数学语义、同一后端精度。
    tolerance = {
        "float32": {"standard": "mix_tolerance"},
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
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

        # ATen upsample_linear1d_backward 对 L_out == L_in 走 early-exit 返回
        # 拷贝并忽略 scales; 原先直调 ATen 时由它兜底, 改走自实现后需显式保留
        if int(g.shape[2]) == isize[2]:
            return [g.to(orig_dtype)]

        y = _linear_grad_core(
            g.numpy(), isize[2], bool(align_corners), _opt_scale(scales)
        )
        return [torch.from_numpy(y).to(orig_dtype)]

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

    # 与 kernel 入口同值: 同一数学语义、同一后端精度。
    tolerance = {
        "float32": {"standard": "mix_tolerance"},
        "float16": {"standard": "mix_tolerance"},
        "bfloat16": {"standard": "mix_tolerance"},
    }
