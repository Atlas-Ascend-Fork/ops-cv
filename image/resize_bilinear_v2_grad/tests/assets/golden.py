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

"""TTK golden for ResizeBilinearV2Grad (2D bilinear backward, NCHW / NHWC).

标杆标准: ops-precision-standard 浮点计算类 · 单标杆比对 · 优先级-1 高精度 CPU
实现. 按 (align_corners, half_pixel_centers, scales) 在调用框架**之前**做确定性
分支选择, 不捕获框架异常静默回退.

反向是前向 gather 的转置(scatter): 对每个上游位置 (oh,ow), 用与前向一致的 4 邻
点 / 双线性权重把 **实际上游梯度 grads[oh,ow]**(非 ones_like) 散射回源网格,
与 kernel SimtCompute (op_kernel/arch35/resize_bilinear_v2_grad_simt.h) 逐行一致。

语义真值源是 kernel 实现(非 proto 注释):
  * scale 取值 (tiling SetScales / docs/aclnnUpsampleBilinear2dBackward.md):
      align_corners=1            -> (src-1)/(dst-1), dst>1; 否则 src/dst; 忽略 scales
      align_corners=0 & attr>0   -> 1/attr                 (scales 仅在 ac=0 生效)
      align_corners=0            -> src/dst
  * 源坐标 orig = (idx+off)*scale - off, off = half_pixel ? 0.5 : 0, 再 clamp 到 >=0
  * left = min(floor(orig), len-1); right = left<len-1 ? left+1 : left
    delta = clamp(orig-left, 0, 1)
  * align_corners=1 且 half_pixel_centers=1 被 tiling GetAttrInfo 拒绝 -> 本 golden
    同样 raise, 不产出“看似正常”的结果。

默认后端路由 (ref_backend="auto"), **默认路径全部按 float64 计算**:

| ac | hp | scales       | 后端                                      | 精度    |
|----|----|--------------|-------------------------------------------|---------|
| 1  | 0  | 忽略         | torch aten.upsample_bilinear2d_backward   | float64 |
| 0  | 1  | 无等长轴冲突 | torch 同上(sh/sw = scales>0 时的 attr)    | float64 |
| 0  | 1  | 等长轴带显式 | NumPy kernel 转写(torch 会丢 scale)       | float64 |
| 0  | 0  | 任意         | NumPy kernel 转写                         | float64 |
| 1  | 1  | --           | raise ValueError (kernel 无该分支)        | --      |

**等长轴 + 显式 scale 必须绕开 torch**: torch CPU 的
upsample_bilinear2d_backward 对"输入输出等长"的空间轴走短路实现(aten
UpSample.h compute_index_ranges_weights / UpSampleMoreKernel.cpp), 直接用
同位置索引与 1/0 权重, 完全不看该轴传入的 scale; 而 kernel 没有这个短路,
ac=0 下仍按 orig = (idx+off)*(1/attr) - off 做真实散射。故"某一轴等长且该轴
带显式 scale>0"时(见 _torch_drops_scale)即便 hp=1 也必须交给 NumPy 转写,
否则会拿到未缩放的梯度 —— 这是 ALL_COPY 之外第二条会漏算的同尺寸路径,
单轴等长(仅 H 或仅 W)同样命中。

ALL_COPY(grads.dtype == original_image.dtype 且 H/W 均相等): 直返 grads, 严格对齐
kernel IsMatchAllCopy (tiling arch35 310-312 行: lenSrcH==lenDesH && lenSrcW==lenDesW
&& gradsDtype==yDtype), 该分支**不检查** ac/hp/scales。混合 dtype(grads=float32,
original_image=float16/bfloat16)不命中 ALL_COPY, kernel 走 SIMT 真实散射, 故 golden
也必须真实计算 —— 这是本文件修复的核心点。

torch align_corners=False 恒为 half_pixel, 故 (ac=0, hp=0) 无法用 torch 表达, 由与
kernel 逐行一致的 NumPy float64 转写承担。

精度例外(显式声明): ref_backend="tf" 调用 tf.raw_ops.ResizeBilinearGrad, TF 仅注册
float32, 即该后端**实际为 float32 计算**, 不是 float64; 它也不接受 scales 参数。该
后端仅用于 legacy TF 兼容性交叉核对(opt-in), 不是默认高精度标杆; 传入显式 scales 时
直接 raise 而非静默忽略参数。实测 (ac=0,hp=0, 7 组形状) NumPy float64 与 TF float32
最大相对偏差 4.1e-7, 符合单精度量级。

布局: 从 TTK 注入的 input_formats kwarg 读取(NCHW/NHWC/ND, 缺省与 ND 均按 NCHW
解释); tiling 判的是**存储格式**(GetFormat().GetStorageFormat()), 故 input_formats
优先于 input_ori_formats。内部统一转 NCHW 计算, 结果按原布局转回。输出 shape/dtype
恒等于 original_image。

dtype 校验按 TTK 注入的 golden_mode kwarg 分流: --golden-mode Promote 会在调用本
golden 之前把输入数组与 input_dtypes/output_dtypes 元数据一起按 DTYPE_PROMOTE_MAP
升精度(float16/bfloat16 -> float32, float32 -> float64, 见 ttk _promote_dtype),
升精度后的数组已无法还原设备 dtype; 而模式本身由
_collect_dynamic_golden_kwargs 以 golden_mode 下发(本 golden 声明了 **kwargs,
accepts_var_keyword 为真 => kwargs 不裁剪), 故 Promote 下按"升精度像"校验、其余
模式下按设备组合校验, 不必放宽成两者的并集。只在拿不到该 kwarg 时(不下发 mode
的旧版 TTK)才退回并集, 以免把合法用例判为非法, 也不为过校验把计算降精度。详见
_check_dtypes 处的推导与 Promote 下已知的 float16/bfloat16 信息损失。
"""

import math

import numpy as np

__spec__ = {
    "resize_bilinear_v2_grad": "ResizeBilinearV2GradTestSpec",
}

_SUPPORTED_LAYOUTS = ("NCHW", "NHWC", "ND")
_SCALES_NUM = 2  # 对齐 tiling SCALES_NUM


# ---------------------------------------------------------------- 布局辅助
def _resolve_layout(kwargs):
    """Read the tensor layout from the TTK-injected format kwargs.

    TTK passes input_formats / input_ori_formats into the golden
    (ops-test-kit ttk/core_modules/kernel/golden.py). input_formats wins:
    tiling reads the STORAGE format (GetFormat().GetStorageFormat(), see
    GetTensorInfo) and input_ori_formats is only the original declaration,
    which TTK defaults to input_formats and then to "ND" -- preferring it would
    route an NHWC storage case through NCHW whenever a case declares both.
    Both grads and original_image must carry the same layout (tiling
    CheckFormatValid), so the first entry decides. Absent / "ND" is interpreted
    as NCHW, matching the historical NCHW-only cases of this operator.
    """
    formats = kwargs.get("input_formats") or kwargs.get("input_ori_formats")
    if not formats:
        return "NCHW"
    if isinstance(formats, str):
        entries = [formats]
    else:
        try:
            entries = [f for f in formats if f is not None]
        except TypeError:
            return "NCHW"
    if not entries:
        return "NCHW"
    names = [str(f).strip().upper() for f in entries[:2]]
    for name in names:
        if name not in _SUPPORTED_LAYOUTS:
            raise ValueError(
                f"ResizeBilinearV2Grad supports NCHW/NHWC only, got format {name!r} "
                "(tiling CheckFormatValid rejects everything else)."
            )
    if len(names) == _SCALES_NUM and names[0] != names[1]:
        raise ValueError(
            "The formats of grads and original_image must be the same, got "
            f"{names[0]!r} and {names[1]!r}."
        )
    return "NCHW" if names[0] == "ND" else names[0]


def _to_nchw(arr, layout):
    return (
        np.ascontiguousarray(np.transpose(arr, (0, 3, 1, 2)))
        if layout == "NHWC"
        else arr
    )


def _from_nchw(arr, layout):
    return (
        np.ascontiguousarray(np.transpose(arr, (0, 2, 3, 1)))
        if layout == "NHWC"
        else arr
    )


# ---------------------------------------------------------------- 分支选择辅助
def _scale_attr(scales, idx):
    """scales[idx] as positive float, or None when absent/<=0."""
    if scales is None:
        return None
    try:
        v = float(scales[idx])
    except (IndexError, TypeError, ValueError):
        return None
    return v if v > 0.0 else None


def _has_explicit_scales(scales):
    return _scale_attr(scales, 0) is not None or _scale_attr(scales, 1) is not None


def _torch_drops_scale(hi, wi, ho, wo, align_corners, scales):
    """True when torch would silently ignore an explicit scale on an equal axis.

    torch's CPU upsample_bilinear2d_backward short-circuits any spatial axis
    whose input and output lengths are equal: it uses the identity index with
    weights 1/0 and never consults the passed scale (aten UpSample.h
    compute_index_ranges_weights, UpSampleMoreKernel.cpp backward). The kernel
    has no such short-circuit -- with scales=[a, b] and align_corners=false it
    still maps orig = (idx + off) * (1 / attr) - off, so an equal-length axis
    carrying attr != 1 really does scatter. Routing such a case to torch
    returns the unscaled gradient, which is why ALL_COPY-style wrong results
    survived on the half_pixel path even after the dtype guard was added.

    align_corners ignores scales entirely (tiling SetScales), and an axis with
    no explicit scale degenerates to src/dst == 1 on equal lengths, i.e. the
    identity torch already computes -- neither case diverges.
    """
    if align_corners:
        return False
    return (hi == ho and _scale_attr(scales, 0) is not None) or (
        wi == wo and _scale_attr(scales, 1) is not None
    )


# ---------------------------------------------------------------- torch 分支 A/B
def _torch_grad(grads, hi, wi, align_corners, scales):
    import torch

    n, c, ho, wo = grads.shape
    g = torch.from_numpy(np.ascontiguousarray(grads, dtype=np.float64))
    if align_corners:
        sh = sw = None  # align_corners 忽略 scales(与 kernel SetScales 一致)
    else:
        sh = _scale_attr(scales, 0)
        sw = _scale_attr(scales, 1)
    y = torch.ops.aten.upsample_bilinear2d_backward(
        g,
        [int(ho), int(wo)],
        [int(n), int(c), int(hi), int(wi)],
        bool(align_corners),
        sh,
        sw,
    )
    return y.numpy()


# ---------------------------------------------------------------- TF 交叉核对后端
def _tf_grad(grads, hi, wi, align_corners, half_pixel, scales):
    """Legacy-TF cross-check backend. float32 compute (documented exception)."""
    if _has_explicit_scales(scales):
        raise ValueError(
            "tf.raw_ops.ResizeBilinearGrad has no scales argument; the TF backend "
            "cannot express explicit scales. Use ref_backend='numpy' (or 'auto')."
        )
    import tensorflow as tf

    n, c, ho, wo = grads.shape
    g_nhwc = np.ascontiguousarray(np.transpose(grads.astype(np.float32), (0, 2, 3, 1)))
    # original_image 仅提供形状/dtype, 内容不参与梯度; float32 NHWC 零张量即可
    img_nhwc = np.zeros((n, hi, wi, c), np.float32)
    y = tf.raw_ops.ResizeBilinearGrad(
        grads=tf.constant(g_nhwc),
        original_image=tf.constant(img_nhwc),
        align_corners=bool(align_corners),
        half_pixel_centers=bool(half_pixel),
    ).numpy()
    return np.ascontiguousarray(np.transpose(y, (0, 3, 1, 2))).astype(np.float64)


# ---------------------------------------------------------------- NumPy kernel 转写
def _compute_scale(src, dst, align_corners, scale_attr):
    if align_corners:
        return (src - 1.0) / (dst - 1.0) if dst > 1 else float(src) / float(dst)
    if scale_attr is not None and scale_attr > 0.0:
        return 1.0 / float(scale_attr)
    return float(src) / float(dst)


def _src_coord(dst_idx, scale, half_pixel):
    v = (dst_idx + 0.5) * scale - 0.5 if half_pixel else dst_idx * scale
    return v if v > 0.0 else 0.0


def _axis_weight(orig, src_len):
    left = int(math.floor(orig))
    if left > src_len - 1:
        left = src_len - 1
    right = left + 1
    if right > src_len - 1:
        right = src_len - 1
    delta = orig - left
    if delta < 0.0:
        delta = 0.0
    elif delta > 1.0:
        delta = 1.0
    return left, right, delta


def _numpy_grad(grads, hi, wi, align_corners, half_pixel, scales):
    n, c, ho, wo = grads.shape
    gf = grads.astype(np.float64)
    sh = _compute_scale(hi, ho, align_corners, _scale_attr(scales, 0))
    sw = _compute_scale(wi, wo, align_corners, _scale_attr(scales, 1))
    wmap = [_axis_weight(_src_coord(ow, sw, half_pixel), wi) for ow in range(wo)]
    y = np.zeros((n, c, hi, wi), np.float64)
    for oh in range(ho):
        ty, by, dy = _axis_weight(_src_coord(oh, sh, half_pixel), hi)
        wy_top = 1.0 - dy
        wy_bot = dy
        for ow in range(wo):
            lx, rx, dx = wmap[ow]
            wx_left = 1.0 - dx
            wx_right = dx
            g = gf[:, :, oh, ow]
            y[:, :, ty, lx] += g * (wy_top * wx_left)
            y[:, :, ty, rx] += g * (wy_top * wx_right)
            y[:, :, by, lx] += g * (wy_bot * wx_left)
            y[:, :, by, rx] += g * (wy_bot * wx_right)
    return y


# ---------------------------------------------------------------- 校验
def _check_attrs(align_corners, half_pixel, scales):
    if align_corners and half_pixel:
        raise ValueError(
            "align_corners=True together with half_pixel_centers=True is rejected by "
            "ResizeBilinearV2Grad (tiling GetAttrInfo returns GRAPH_FAILED); this "
            "combination has no valid golden and belongs to the exception tests."
        )
    if scales is not None:
        try:
            length = len(scales)
        except TypeError as exc:
            raise ValueError(
                f"scales must be a 2-element sequence, got {scales!r}."
            ) from exc
        if length != _SCALES_NUM:
            raise ValueError(
                f"scales must hold exactly {_SCALES_NUM} elements, got {length}."
            )


# 设备侧合法 dtype 组合(tiling CheckDtypeValid): grads=float32 时
# y ∈ {float32, float16, bfloat16}; grads 非 float32 时必须 grads == y。
_DEVICE_DTYPE_PAIRS = (
    ("float32", "float32"),
    ("float32", "float16"),
    ("float32", "bfloat16"),
    ("float16", "float16"),
    ("bfloat16", "bfloat16"),
)
# 上表逐项按 DTYPE_PROMOTE_MAP 升精度后的像(float16/bfloat16 -> float32、
# float32 -> float64): float16/bfloat16 都映射到 float32, 故 5 条设备组合塌缩为
# 3 条。
_PROMOTED_DTYPE_PAIRS = (
    ("float64", "float64"),  # <- (float32, float32)
    ("float64", "float32"),  # <- (float32, float16) / (float32, bfloat16)
    ("float32", "float32"),  # <- (float16, float16) / (bfloat16, bfloat16)
)
# 不下发 golden_mode 的旧版 TTK 只能按"设备组合 ∪ 其升精度像"的并集放行: 两种
# 解释下 ALL_COPY(判两者 dtype 是否相等)的结论一致 —— 唯一的异 dtype 合法组合
# float32+float16/bfloat16 升精度为 float64+float32 后仍不相等, 同 dtype 组合升
# 精度后仍相等 —— 故并集不会改变本 golden 的分支选择, 只是放宽了校验。
_UNION_DTYPE_PAIRS = _DEVICE_DTYPE_PAIRS + tuple(
    pair for pair in _PROMOTED_DTYPE_PAIRS if pair not in _DEVICE_DTYPE_PAIRS
)


def _allowed_dtype_pairs(golden_mode):
    """Legal (grads, y) dtype pairs for the mode TTK says it is calling with.

    --golden-mode Promote promotes the input arrays (and the input_dtypes /
    output_dtypes kwargs) by DTYPE_PROMOTE_MAP before calling the golden
    (ttk/core_modules/kernel/golden.py _promote_dtype), so under Promote the
    arrays carry the promoted image of the device combination, not the device
    combination itself. The mode is injected as the golden_mode kwarg by
    _collect_dynamic_golden_kwargs, so each mode can be checked against
    exactly one table. Only a TTK that predates that kwarg forces the union.
    """
    if golden_mode is None:
        return _UNION_DTYPE_PAIRS
    if str(golden_mode) == "Promote":
        return _PROMOTED_DTYPE_PAIRS
    return _DEVICE_DTYPE_PAIRS


def _check_dtypes(grads, original_image, golden_mode=None):
    pair = (str(grads.dtype), str(original_image.dtype))
    allowed = _allowed_dtype_pairs(golden_mode)
    if pair in allowed:
        return
    mode_msg = (
        "a TTK revision that does not pass golden_mode"
        if golden_mode is None
        else f"golden_mode={golden_mode}"
    )
    if pair[0] == pair[1]:
        raise ValueError(
            f"The dtypes of grads and y are {pair[0]} under {mode_msg}, which is not "
            "a legal combination for ResizeBilinearV2Grad (tiling CheckDtypeValid "
            f"accepts {list(_DEVICE_DTYPE_PAIRS)}; under Promote the golden receives "
            f"their promoted image instead). Accepted here: {list(allowed)}."
        )
    if pair[0] not in ("float32", "float64"):
        raise ValueError(
            "The dtypes of grads and y must be the same when the dtype of grads is "
            f"not float32, got {pair[0]} and {pair[1]} (tiling CheckDtypeValid; "
            f"float64 grads only arise under golden_mode=Promote, here {mode_msg})."
        )
    raise ValueError(
        "The dtype of y must be float32, float16 or bfloat16 when grads is float32 "
        f"(float64 grads with float32 y under golden_mode=Promote), got {pair[0]} "
        f"and {pair[1]} under {mode_msg}. Accepted combinations: {list(allowed)}."
    )


def _check_shapes(grads, original_image):
    if grads.ndim != 4 or original_image.ndim != 4:
        raise ValueError(
            "The shapes of grads and original_image must be 4D, got "
            f"{grads.shape} and {original_image.shape}."
        )


def _check_axes(grads_nchw, image_nchw):
    gn, gc, gh, gw = grads_nchw.shape
    yn, yc, yh, yw = image_nchw.shape
    if gn != yn or gc != yc:
        raise ValueError(
            "The N/C dimensions of grads and original_image must be the same, got "
            f"N={gn}/{yn}, C={gc}/{yc}."
        )
    if min(gh, gw, yh, yw) <= 0:
        raise ValueError(
            "The H/W dimensions of grads and original_image must be greater than 0, "
            f"got grads HW=({gh},{gw}) and original_image HW=({yh},{yw})."
        )


# ---------------------------------------------------------------- 后端路由
def _bilinear_grad(grads, hi, wi, align_corners, half_pixel, scales, ref_backend):
    """Dispatch one legal (ac, hp, scales) combination to a deterministic backend."""
    ho, wo = int(grads.shape[2]), int(grads.shape[3])
    drops_scale = _torch_drops_scale(hi, wi, ho, wo, align_corners, scales)

    if ref_backend == "numpy":
        return _numpy_grad(grads, hi, wi, align_corners, half_pixel, scales)
    if ref_backend == "tf":
        return _tf_grad(grads, hi, wi, align_corners, half_pixel, scales)
    if ref_backend == "torch":
        if (not align_corners) and (not half_pixel):
            raise ValueError(
                "torch align_corners=False always implies half_pixel; the torch "
                "backend cannot express (align_corners=False, half_pixel_centers="
                "False). Use ref_backend='numpy' (or 'auto')."
            )
        if drops_scale:
            raise ValueError(
                "torch short-circuits a spatial axis whose input and output "
                "lengths are equal and ignores the scale on that axis, while the "
                f"kernel still scatters: grads HW=({ho},{wo}), y HW=({hi},{wi}), "
                f"scales={scales!r}. The torch backend cannot express this "
                "combination. Use ref_backend='numpy' (or 'auto')."
            )
        return _torch_grad(grads, hi, wi, align_corners, scales)
    if ref_backend != "auto":
        raise ValueError(
            f"ref_backend={ref_backend!r} is not supported; expected one of "
            "'auto', 'numpy', 'torch', 'tf'."
        )
    # auto: 确定性分支选择(调用框架之前), 全部 float64
    if align_corners:  # 分支 A (ac=1, hp=0), scales 被忽略
        return _torch_grad(grads, hi, wi, True, None)
    if half_pixel and not drops_scale:  # 分支 B (ac=0, hp=1), scales 透传 torch
        return _torch_grad(grads, hi, wi, False, scales)
    # 分支 C: torch 无法表达 -> NumPy float64 转写
    #   * (ac=0, hp=0) torch 的 align_corners=False 恒为 half_pixel;
    #   * 等长轴携带显式 scale 时 torch 会短路该轴、丢掉 scale(见
    #     _torch_drops_scale), 此时 hp=1 也必须交给 NumPy。
    return _numpy_grad(grads, hi, wi, False, half_pixel, scales)


class ResizeBilinearV2GradTestSpec:
    """TestSpec golden for ResizeBilinearV2Grad."""

    @staticmethod
    def golden(
        grads,
        original_image,
        *,
        align_corners=False,
        half_pixel_centers=False,
        scales=None,
        ref_backend="auto",
        **kwargs,
    ):
        grads = np.ascontiguousarray(grads)
        original_image = np.ascontiguousarray(original_image)
        align_corners = bool(align_corners)
        half_pixel_centers = bool(half_pixel_centers)

        _check_shapes(grads, original_image)
        _check_attrs(align_corners, half_pixel_centers, scales)
        _check_dtypes(grads, original_image, kwargs.get("golden_mode"))

        layout = _resolve_layout(kwargs)
        grads_nchw = _to_nchw(grads, layout)
        image_nchw = _to_nchw(original_image, layout)
        _check_axes(grads_nchw, image_nchw)

        hi, wi = int(image_nchw.shape[2]), int(image_nchw.shape[3])
        ho, wo = int(grads_nchw.shape[2]), int(grads_nchw.shape[3])

        # ALL_COPY: kernel IsMatchAllCopy 判 gradsDtype==yDtype + H/W 长度相等
        # (resize_bilinear_v2_grad_tiling_arch35.cpp:310-312), 不检查 ac/hp/scales。
        # dtype 不等时 kernel 走 SIMT 真实散射, 这里必须跟着真实计算。
        if grads.dtype == original_image.dtype and hi == ho and wi == wo:
            y = grads.astype(original_image.dtype, copy=True)
            return [y]

        y_nchw = _bilinear_grad(
            grads_nchw,
            hi,
            wi,
            align_corners,
            half_pixel_centers,
            scales,
            str(ref_backend).strip().lower(),
        )
        y = _from_nchw(y_nchw, layout)
        return [y.astype(original_image.dtype, copy=False)]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
