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

"""Golden for upsample_bicubic2d_aa — torch ATen IR implementation.

Two entries:
  * ``upsample_bicubic2d_aa``      —— kernel 路径 (Ascend IR attr 语义, 架构相关)
  * ``aclnnUpsampleBicubic2dAA``   —— ACLNN 路径 (attr 恒为 OUT/IN, 见下)

The golden calls ``torch.ops.aten._upsample_bicubic2d_aa`` directly (per PR
!1808 review: the Ascend IR op is benchmarked against the ATen IR op).

Equivalence (Ascend IR UpsampleBicubic2dAA <-> aten::_upsample_bicubic2d_aa):
both use the Keys cubic filter with A=-0.5 (PIL-compatible antialias;
NPU kernel WeightCalculate vs PyTorch aa_filter "using -0.5 for bicubic,
antialiasing=true"), the normalised filter window
support=2*scale (scale=in/out, downsample) / invscale=1/scale, and the
separable W-then-H passes. Verified point-wise in float64 against an
independent reference (rel <= 2.1e-12 across up/down/identity/align_corners/
non-integer shapes).

Parameter mapping:
* x (tensor input)              -> self
* output_size (attr ListInt)    -> output_size [oh, ow]
* align_corners (attr, bool)    -> align_corners (only changes the scale)
* scales_h / scales_w (attr)    -> scales_h / scales_w. NOTE the ATen args
  carry the OUT/IN ratio (value > 0 -> src/dst scale = 1/value), while the
  Ascend IR attr semantics are architecture-dependent (aclnn
  UpsampleBicubic2dAA IsRegBase branch):
    - Ascend950 (RegBase, tiling_arch35.cpp): IR attr is OUT/IN  -> pass
      through to ATen unchanged;
    - other archs (910b/910_93, upsample_bicubic2d_aa_tiling.cpp): IR attr
      is IN/OUT -> pass 1/attr to ATen when attr > 0.
  The SoC is taken from kwargs when available; defaults to RegBase.

Two ATen limitations, reported in the PR (not worked around silently):
1. aten::_upsample_bicubic2d_aa does NOT dispatch Half/BFloat16 on CPU
   (RuntimeError: "compute_index_ranges_weights" not implemented). Reduced
   floats are lifted to fp32 before the call — matching the NPU kernel,
   which computes AA weights/accumulation in float (upsample_bicubic2d_aa_
   kernel.h local float tensors; fp16/bf16 weights CAST_RINT-quantised) —
   and the result is quantised back.
2. aten::_upsample_bicubic2d_aa collapses the H direction when the OUTPUT
   WIDTH == 1 (ATen CPU bug, reproduced on 2.7.1 with float64 inputs:
   every H row equals the first; independent reference chains — torch
   single-axis + copy construction, pointwise unrolled second
   implementation — agree the correct output differs, rel=1.0). For
   output_size[1] == 1 ONLY, the golden falls back to the in-file
   independent fp64 implementation below (verified by three independent
   reference chains, rel <= 7.8e-6 cross-agreement); the NPU kernel
   behaves correctly on this shape (NPU cases pass against the fallback).
   See the PR discussion for the maintainer decision on this scenario.
"""

import numpy as np
import torch

__spec__ = {
    "upsample_bicubic2d_aa": "UpsampleBicubic2dAATestSpec",
    "aclnnUpsampleBicubic2dAA": "AclnnUpsampleBicubic2dAATestSpec",
    "torch.ops.aten._upsample_bicubic2d_aa": "E2eUpsampleBicubic2dAATestSpec",
}

_CUBIC_A = -0.5  # PIL/Keys antialias coefficient — do NOT use -0.75 here


def _to_torch(x):
    """numpy -> torch tensor; ml_dtypes bfloat16 via bit-exact uint16 view."""
    if str(x.dtype) == "bfloat16":
        return torch.from_numpy(np.ascontiguousarray(x).view(np.uint16)).view(
            torch.bfloat16
        )
    return torch.from_numpy(np.ascontiguousarray(x))


def _to_numpy(t, orig_dtype):
    """torch -> numpy with dtype restore (bfloat16 via uint16 view)."""
    if orig_dtype == "bfloat16":
        from ml_dtypes import bfloat16 as np_bf16

        return t.contiguous().view(torch.uint16).cpu().numpy().view(np_bf16)
    return t.cpu().numpy()


def _is_regbase_soc(short_soc_version=None, full_soc_version=None):
    """Ascend950 (RegBase) family detection.

    The IR attr scales_h/scales_w semantics differ by architecture (see the
    aclnnUpsampleBicubic2dAA IsRegBase branch): RegBase archs carry the
    OUT/IN ratio in the attr, non-RegBase archs carry IN/OUT. Defaults to
    RegBase when no SoC info is available (primary target of this asset).
    """
    for soc in (short_soc_version, full_soc_version):
        if soc:
            s = str(soc)
            if "950" in s or "960" in s:
                return True
            if "910" in s or "310" in s or "920" in s:
                return False
    return True  # default: RegBase (Ascend950) semantics


def _normalize_output_size(output_size):
    """Accept list/tuple/ndarray/int; return (oh, ow)."""
    if output_size is None:
        raise ValueError("output_size attr is required for upsample_bicubic2d_aa")
    if isinstance(output_size, (int, np.integer)):
        return int(output_size), int(output_size)
    if isinstance(output_size, np.ndarray):
        output_size = output_size.flatten().tolist()
    vals = [int(v) for v in output_size]
    if len(vals) == 1:
        return vals[0], vals[0]
    if len(vals) != 2:
        raise ValueError(f"output_size must hold 2 elements, got {output_size!r}")
    return vals[0], vals[1]


# --------------------------- 空 Tensor 口径 (HPTG #155 FR-GOLDEN-ZERO-SIZE)
_AA_ZERO_REJECT = (
    "源空间轴为空但要求非空输出 (in_h == 0 或 in_w == 0, 而 N/C/out_h/out_w 均"
    "非 0): AA 的归一化权重窗口没有任何源像素可取, 插值无定义。golden 按接口合同"
    "显式标注为**不可比较**(返回 None → TTK 记 SUPPRESSED), 既不留作 ATen 的 "
    "'Input and output sizes should be greater than 0' 静默异常, 也不把仅"
    "完成输入准备的行当作精度通过。"
)


def _aa_zero_verdict(n, c, in_h, in_w, out_h, out_w):
    """空 Tensor 分类: "empty" 合法空 / "reject" 无源可插值 / None 常规。

    依据 docs/aclnnUpsampleBicubic2dAA.md —— 该文档对入参 x(:126)与出参(:176)
    均明示"支持空Tensor", 故空输入属**合法域**, 不能一概拒绝:
      * 输出本身为空 (N / C / out_h / out_w 任一为 0) —— 没有需要计算的输出
        元素, 参考值就是同形空张量;
      * 仅源空间轴为空 (in_h / in_w 为 0) 而输出非空 —— 无源像素可插值, 拒绝。

    注: 零尺寸在 Host 侧的拒绝当前失效并整数除零 (#155 FR-CODE-EMPTY-SIGFPE,
    P1, 不在本 PR 范围), 故此处只约束参考侧口径, 不声称设备已支持。
    """
    if 0 in (int(n), int(c), int(out_h), int(out_w)):
        return "empty"
    if int(in_h) == 0 or int(in_w) == 0:
        return "reject"
    return None


# ---------------------------------------------------------------------------
# Independent fp64 fallback — used ONLY for the output_width == 1 shape,
# where aten::_upsample_bicubic2d_aa has the H-collapse bug (see module
# docstring item 2). Verified against three independent reference chains.
# ---------------------------------------------------------------------------


def _w_keys(x, a=_CUBIC_A):
    """Cubic convolution kernel, vectorised (fp64)."""
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    out = np.zeros_like(ax)
    m1 = ax <= 1.0
    m2 = (ax > 1.0) & (ax < 2.0)
    out[m1] = (a + 2.0) * ax[m1] ** 3 - (a + 3.0) * ax[m1] ** 2 + 1.0
    out[m2] = a * ax[m2] ** 3 - 5.0 * a * ax[m2] ** 2 + 8.0 * a * ax[m2] - 4.0 * a
    return out


def _aa_scale(in_size, out_size, align_corners, s, is_regbase=True):
    """AA scale (in/out) from the IR attr value (architecture-aware)."""
    if align_corners:
        return (in_size - 1.0) / (out_size - 1.0) if out_size > 1 else 0.0
    if s is not None and s > 0:
        # RegBase(950): attr = out/in -> internal in/out scale = 1/s
        # others(910b): attr = in/out -> internal in/out scale = s
        return 1.0 / float(s) if is_regbase else float(s)
    if in_size == out_size:
        return 1.0
    return in_size / out_size


def _aa_axis_tables(in_size, out_size, align_corners, s, is_regbase=True):
    """Return list of (idx ndarray, normalised weights ndarray) per dst."""
    scale = _aa_scale(in_size, out_size, align_corners, s, is_regbase)
    support = 2.0 * scale if scale >= 1.0 else 2.0
    invscale = 1.0 / scale if scale > 1.0 else 1.0
    max_interp = int(np.ceil(support)) * 2 + 1
    tables = []
    for dst in range(out_size):
        center = (dst + 0.5) * scale
        x_min = max(0, int(np.floor(center + 0.5 - support)))
        x_max = min(in_size, int(np.floor(center + 0.5 + support)))
        count = max(0, min(x_max - x_min, max_interp))
        if count <= 0:
            tables.append((np.zeros(0, dtype=np.int64), np.zeros(0)))
            continue
        idx = np.arange(x_min, x_min + count)
        w = _w_keys((idx + 0.5 - center) * invscale)
        total = w.sum()
        if total != 0.0:
            w = w / total
        tables.append((idx, w))
    return tables


def _upsample_bicubic2d_aa_nchw(
    x, out_h, out_w, align_corners, scales_h, scales_w, is_regbase=True
):
    """Independent AA forward on NCHW fp input -> fp64 [N,C,oh,ow].

    The two-axis weighted multiply-accumulate is performed with **framework
    tensor ops** (``torch.matmul`` in fp64), not with elementwise NumPy
    accumulation loops (Harness GOLDEN.05). The sparse per-dst
    ``(idx, weights)`` tables are densified into two weight matrices and the
    separable AA resize becomes two matmuls::

        mid = x   @ Ww^T     # (N,C,ih,iw) @ (iw,ow) -> (N,C,ih,ow)
        y   = Wh  @ mid      # (oh,ih) @ (N,C,ih,ow) -> (N,C,oh,ow)

    Densification is semantics-preserving: a dst row of ``Ww``/``Wh`` is zero
    everywhere outside its own ``idx`` window, so the matmul sums exactly the
    same terms as the per-window loop. Only the summation *order* is left to
    the framework; the fp64 deviation from the previous loop form is quantified
    in ``scratch/verify_aa_equiv.py`` (max rel ~1e-16, i.e. rounding only).

    Weight *table construction* (``_w_keys`` / ``_aa_axis_tables``) stays NumPy
    on purpose: it is index/coefficient bookkeeping, not operator math.
    """
    n, c, ih, iw = x.shape
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float64))

    def _dense(tables, in_size, out_size):
        """(idx, w) 稀疏表 -> 稠密 (out_size, in_size) fp64 框架张量。"""
        m = torch.zeros((out_size, in_size), dtype=torch.float64)
        for dst, (idx, w) in enumerate(tables):
            if idx.size:
                m[dst, torch.from_numpy(np.ascontiguousarray(idx))] = torch.from_numpy(
                    np.ascontiguousarray(w, dtype=np.float64)
                )
        return m

    ww = _dense(
        _aa_axis_tables(iw, out_w, align_corners, scales_w, is_regbase), iw, out_w
    )
    wh = _dense(
        _aa_axis_tables(ih, out_h, align_corners, scales_h, is_regbase), ih, out_h
    )

    # Pass 1 (W-direction): mid[n,c,ih,ow] = sum_k Ww[ow,k] * x[n,c,ih,k]
    mid = torch.matmul(xt, ww.transpose(0, 1))
    # Pass 2 (H-direction): y[n,c,oh,ow] = sum_k Wh[oh,k] * mid[n,c,k,ow]
    y = torch.matmul(wh, mid)
    assert tuple(y.shape) == (n, c, out_h, out_w)
    return y.numpy()


class UpsampleBicubic2dAATestSpec:
    """TestSpec for upsample_bicubic2d_aa (kernel path, aten IR golden)."""

    def golden(
        x, *, output_size, align_corners=False, scales_h=0.0, scales_w=0.0, **kwargs
    ):
        """Golden for upsample_bicubic2d_aa via aten._upsample_bicubic2d_aa.

        Parameters follow upsample_bicubic2d_aa_def.cpp without outputs.
        x is NCHW (format ND). output_size is the required [outH, outW] attr.
        scales_h/scales_w follow the architecture-dependent IR semantics
        (see module docstring); the SoC is taken from kwargs when available.

        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name, golden_mode
        """
        x = np.asarray(x)
        if x.ndim != 4:
            raise ValueError(f"x must be 4-D NCHW, got shape {x.shape}")
        out_h, out_w = _normalize_output_size(output_size)

        n_, c_, in_h_, in_w_ = (int(v) for v in x.shape)
        verdict = _aa_zero_verdict(n_, c_, in_h_, in_w_, out_h, out_w)
        if verdict == "reject":
            return [None]  # 见 _AA_ZERO_REJECT
        if verdict == "empty":
            return [np.zeros((n_, c_, out_h, out_w), dtype=x.dtype)]

        orig_dtype = str(x.dtype)
        is_regbase = _is_regbase_soc(
            kwargs.get("short_soc_version"), kwargs.get("full_soc_version")
        )
        s_h_attr, s_w_attr = float(scales_h), float(scales_w)

        if out_w == 1:
            # ATen CPU defect: H direction collapses for output width == 1
            # (module docstring item 2). Independent fp64 fallback, verified
            # by three independent reference chains; NPU is correct here.
            x_f = x.astype(np.float32) if orig_dtype in ("float16", "bfloat16") else x
            y = _upsample_bicubic2d_aa_nchw(
                x_f, out_h, out_w, bool(align_corners), s_h_attr, s_w_attr, is_regbase
            )
            if orig_dtype == "float16":
                y = y.astype(np.float16)
            elif orig_dtype == "bfloat16":
                y = torch.from_numpy(y).to(torch.bfloat16)
                y = y.view(torch.uint16).numpy().view(np.dtype("bfloat16"))
            else:
                y = y.astype(x.dtype, copy=False)
            return [y]

        x_t = _to_torch(x)
        if x_t.dtype in (torch.float16, torch.bfloat16):
            x_t = x_t.to(torch.float32)

        # ATen scales carry the OUT/IN ratio:
        #   RegBase(950) IR attr is out/in -> pass through;
        #   others(910b) IR attr is in/out  -> invert when > 0.
        def _aten_scale(s_attr):
            if s_attr <= 0:
                return None
            return s_attr if is_regbase else 1.0 / s_attr

        y_t = torch.ops.aten._upsample_bicubic2d_aa(
            x_t,
            [out_h, out_w],
            bool(align_corners),
            _aten_scale(s_h_attr),
            _aten_scale(s_w_attr),
        )

        if orig_dtype == "float16":
            y_t = y_t.to(torch.float16)
        elif orig_dtype == "bfloat16":
            y_t = y_t.to(torch.bfloat16)
        return [_to_numpy(y_t, orig_dtype)]

    class ThirdPartyImpl:
        """``torch.ops.aten._upsample_bicubic2d_aa`` reference.

        **Explicit positive scales are now in the legal domain.** The previous
        rejection came from the ``F.interpolate`` wrapper, where ``size`` and
        ``scale_factor`` are mutually exclusive and the default
        ``recompute_scale_factor=True`` re-derives the ratio from out/in. The
        low-level ATen op takes ``output_size`` **and** ``scales_h/scales_w``
        together::

            aten::_upsample_bicubic2d_aa(Tensor self, SymInt[2] output_size,
                bool align_corners, float? scales_h=None, float? scales_w=None)

        and the scales genuinely participate in the coordinate mapping
        (measured on torch 2.7.1 CPU, (2,3,8,16) fp64 -> (5,7),
        align_corners=False: explicit ``scales=2.0`` differs from the
        size-derived result by 5.31 absolute).

        Cross-checked against the in-file fp64 AA implementation over
        align_corners x {False, True} x scales {default, (2,2), (0.5,0.5),
        (2,-), (1.5,3)} x output {(5,7),(4,8),(3,3),(16,32),(1,5)} = 50
        combinations: **worst rel = 6.3e-16**. ``align_corners=True`` ignores
        the scales in ATen, kernel and aclnn alike, so no special case.

        Scale convention follows the in-file golden's ``_aten_scale``: ATen
        wants the OUT/IN ratio, which is exactly the RegBase(950) IR attr, so
        it is passed through. The SoC is **not** carried on this dispatch path
        (only operator attributes reach third_party), so RegBase semantics are
        assumed — the same default ``_is_regbase_soc(None, None)`` takes, and
        the target of this delivery (arch35 / Ascend950) is RegBase.

        **Still rejected, with evidence (not avoidance)**: ``output_size[1] ==
        1``. ATen's CPU AA kernel collapses the H direction for output width 1
        (module docstring item 2). Measured on torch 2.7.1, random (2,3,8,16)
        fp64 -> (5,1), align_corners=False:
          * ATen returns the SAME value in all 5 H rows
            (-0.001254, -0.001254, -0.001254, -0.001254, -0.001254) and
            deviates from the in-file fp64 reference by rel = 1.61;
          * **Pillow** (independent third chain, per-channel ``Image.resize``
            with ``BICUBIC``, which is antialiased by construction) agrees with
            the in-file fp64 reference to rel = **7.6e-08** (fp32 image path)
            — so the in-file reference is right and ATen is wrong here;
          * healthy control on the same tensor: (1,5) rel = 3.2e-08 (PIL) /
            6.4e-16 (ATen), (5,7) rel = 3.6e-08 (PIL) / 1.2e-15 (ATen).
        A third_party reference reproducing the same defect carries no
        cross-check value, so the shape is refused rather than silently passed;
        the in-file golden covers it via the fp64 fallback. Pillow is not
        registered as a provider because ttk only aliases torch / tf / numpy
        (ttk/remote/client.py:_TP_ALIASES).
        """

        def __init__(
            self,
            *,
            output_size=None,
            align_corners=False,
            scales_h=0.0,
            scales_w=0.0,
            device=None,
            **kwargs,
        ):
            # Stage 1: parameter preparation / availability (no data touched)
            if output_size is not None and _normalize_output_size(output_size)[1] == 1:
                raise NotImplementedError(
                    "aten::_upsample_bicubic2d_aa collapses the H direction for "
                    "output_width == 1 (measured torch 2.7.1: random (2,3,8,16) fp64 "
                    "-> (5,1), all 5 H rows identical, rel = 1.61 vs the in-file fp64 "
                    "reference, while Pillow BICUBIC agrees with that reference to "
                    "7.6e-08); it is therefore not an independent cross-check standard "
                    "for this shape — the in-file golden covers it via its fp64 fallback"
                )
            self.output_size = output_size
            self.align_corners = bool(align_corners)
            # ATen wants OUT/IN; RegBase(950) IR attr already is OUT/IN.
            # SoC is not carried on this dispatch path -> RegBase assumed
            # (same default as _is_regbase_soc(None, None)).
            self.scales_h = float(scales_h) if float(scales_h) > 0 else None
            self.scales_w = float(scales_w) if float(scales_w) > 0 else None
            # device 必须**显式声明为形参**才会被注入: ttk 的
            # bind_params 只对 _FRAMEWORK_RESERVED={"device"} 这个**形参名**
            # 注入设备, 它不在 inputs/attrs 里, 因此 `kwargs.get("device")`
            # 恒为空 —— 实测会静默退回 cpu, 配置的设备根本不生效。
            self.device = device or "cpu"

        def __call__(self, x, **kwargs):
            # Stage 2/3: device migration + compute
            oh, ow = _normalize_output_size(self.output_size)
            t = x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x))
            t = t.to(self.device)
            if t.dtype in (torch.float16, torch.bfloat16):
                t = t.to(torch.float32)
            out = torch.ops.aten._upsample_bicubic2d_aa(
                t.contiguous(),
                [oh, ow],
                self.align_corners,
                self.scales_h,
                self.scales_w,
            )
            if torch.is_tensor(x) and x.dtype in (torch.float16, torch.bfloat16):
                out = out.to(x.dtype)
            return [out]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        # fp16/bf16 follow the TTK official dtype table (2^-10 / 2^-7),
        # verified on NPU (Ascend950PR) with multiple seeds. fp32 2e-4 covers
        # the NPU weight CAST_RINT quantisation + fp32 accumulation order.
        "float32": {"standard": "stat_rel_err", "threshold": 2e-4},
        "float16": {"standard": "stat_rel_err", "threshold": 0.0009765625},
        "bfloat16": {"standard": "stat_rel_err", "threshold": 0.0078125},
    }


# --------------------------------------------------------------- ACLNN 入口
_ACLNN_FORMATS = ("ND", "NCHW")


def _aclnn_format(kwargs):
    """首输入 format。两条通路键名**不同**, 都读, 缺省 ND:

      * ``tensor_formats`` —— 本地 golden 调用注入(本函数当前的唯一调用方是
        aclnn golden, 走的就是这个键);
      * ``input_formats`` —— 远端三方执行注入(``executor._compose_kwargs``,
        对任何通路的类形态实现都注入), 是 ``third_party`` provider 侧能看到的
        **唯一** format 来源。

    两者都读, 以免日后 provider 复用本函数时静默退回 ND, 把 NHWC 用例按 NCHW
    算完当成通过(该退化已在本仓库 bicubic/bilinear 的 provider 上实测复现)。
    """
    for key in ("tensor_formats", "input_formats"):
        fmts = kwargs.get(key)
        if not fmts:
            continue
        if isinstance(fmts, str):
            return fmts.strip().upper()
        for v in fmts:
            if v:
                return str(v).strip().upper()
    return "ND"


def _opt_scale(v):
    """aclnn OUT/IN scale → ATen ``Optional[float]``; ``<=0`` 表示未指定。

    直通不反转。依据(``aclnn_upsample_bicubic2d_aa.cpp``
    ``aclnnUpsampleBicubic2dAAGetWorkspaceSize``): ACLNN 边界上
    ``scalesH/scalesW`` 恒为 OUT/IN, 架构差异在该函数内部的 ``IsRegBase()``
    分支里消化 —— RegBase(950) 直通 ``realScalesH = scalesH``(:188-189),
    非 RegBase(910b/910_93) 才取 ``1.0/scalesH`` 喂 kernel(:199-200)。
    ATen ``compute_scales_value`` 同样按 ``scale > 0 → 内部 in/out = 1/scale``
    解读, 与 ACLNN 边界语义一致, 故直通即正确。
    **不要在本入口调用 ``_is_regbase_soc()``** —— 那是 kernel(Ascend IR attr)
    路径的架构修正; 在 aclnn 入口再反转一次, 会在 910b 上把 scale 算成倒数。
    实测(torch 2.7.1): 4x4→8x8 传 ``scales=(2.0,2.0)``(OUT/IN) 与
    ``scales=None`` 逐位相同(rel=0.0); 传 ``(0.5,0.5)``(IN/OUT) rel=1.77。
    16x16→8x8 / 8x8→(16,4) 非对称 / 8x8→5x5 的 5/8 非整数三例同结论。
    """
    if v is None:
        return None
    s = float(v.reshape(-1)[0]) if torch.is_tensor(v) else float(v)
    return s if s > 0.0 else None


# ----------------------------------------------- third_party 共享数学核 (AA)
#
# ACLNN / E2E 两个入口的 third_party 共用下面这个核, 只是形参名不同
# (aclnn 是 camel ``outputSize``/``alignCorners``/``scalesH``/``scalesW``,
# E2E 是 ATen snake ``output_size``/``align_corners``/``scales_h``/
# ``scales_w``) —— 需求允许共享数学核, 但**声明必须各按自己的签名**, 否则
# ttk 的 bind_params 会以 UnknownParamError 拒绝(参数名不在 inputs ∪ attrs)。
#
# ``device`` 必须**显式声明为形参**才会被注入: bind_params 只对
# _FRAMEWORK_RESERVED={"device"} 这个形参名注入, 它不在 inputs/attrs 里,
# 所以 ``kwargs.get("device")`` 恒为空(实测会静默退回 cpu)。
#
# **本族只有 torch 一个 provider**: AA 的独立标杆候选是 Pillow(BICUBIC 按
# 构造即抗锯齿), 但 ttk 只给 torch / tf / numpy 注册别名
# (``ttk/remote/client.py::_TP_ALIASES``), Pillow 无法登记为 provider;
# TF 侧亦无 antialias 的 raw_ops(``tf.image.resize(antialias=True)`` 是
# Python 层组合, 且与 Keys a=-0.5 的 AA 核不同源)。故 AA 的 third_party 只
# 覆盖**参数适配与绑定路径**, 独立数值取证由仓外 Pillow 脚本单独记录
# (见 kernel 入口 ThirdPartyImpl docstring 的 7.6e-08 实测)。


def _tp_aa_aten(x, oh, ow, align_corners, s_h, s_w, device):
    """ATen 低阶 AA bicubic。与文件内 golden 的**非 out_w==1 分支**同源。

    交叉校验价值: 覆盖**参数适配与绑定路径**(camel/snake 形参名、scale 哨兵、
    outputSize 归一、dtype 抬升), 数学与 golden 同源, 不构成独立数值标杆。

    dtype: ``aten::_upsample_bicubic2d_aa`` CPU **不 dispatch fp16/bf16**
    (实测 ``RuntimeError: "compute_index_ranges_weights" not implemented for
    'Half'/'BFloat16'``), 故 fp16/bf16 先抬 fp32 再算, 结果回原 dtype。

    scales 直通不反转: ATen ``compute_scales_value`` 按 ``scale > 0 →
    内部 in/out = 1/scale`` 解读, 即边界语义恒为 OUT/IN, 与 ACLNN 一致。
    """
    t = x if torch.is_tensor(x) else torch.from_numpy(np.ascontiguousarray(x))
    t = t.to(device or "cpu")
    orig = t.dtype
    if orig in (torch.float16, torch.bfloat16):
        t = t.to(torch.float32)
    y = torch.ops.aten._upsample_bicubic2d_aa(
        t.contiguous(), [int(oh), int(ow)], bool(align_corners), s_h, s_w
    )
    return y.contiguous().to(orig)


_AA_TP_W1_REJECT = (
    "output_width == 1: aten::_upsample_bicubic2d_aa 的 CPU 实现在 "
    "``out_w == 1 && out_h != in_h`` 时 H 方向塌缩(torch 2.7.1 实测: 随机 "
    "(2,3,8,16) fp64 -> (5,1), 5 个 H 行全为同一值, 与文件内 fp64 参考 "
    "rel=1.61, 而 Pillow BICUBIC 与该参考吻合 7.6e-08, 且上板 compare 日志显示"
    "设备输出各不相同 ⇒ 设备正确、ATen 错); 复现同一缺陷的标杆无交叉校验价值, "
    "该象限由文件内 golden 的 fp64 独立实现覆盖(用例保留, 不删不排除)"
)


class AclnnUpsampleBicubic2dAATestSpec:
    """aclnnUpsampleBicubic2dAA(x, outputSize, alignCorners, scalesH, scalesW, out).

    双分支(按 ``outputSize[1]`` 确定性选择, 调框架之前就定好, 不捕异常回退)::

        out_w == 1 : 文件内 fp64 独立实现 ``_upsample_bicubic2d_aa_nchw``
        otherwise  : 抬精度后调 ``aten::_upsample_bicubic2d_aa``

    **为什么 out_w==1 必须回退**: ATen CPU 在 ``out_w == 1 && out_h != in_h``
    时 H 方向塌缩(torch 2.7.1 实测)。证据链:

    1. fp64 输入 (2,3,8,16)→(5,1): ATen 输出 5 个 H 行全为同一个值
       (0.20734318...), 而两条独立 CPU 参考链 —— (a) 轴交换法(H/W 转置后调
       ATen, 让坏轴 W==1 变成健康的 out_h==1 轴, AA 可分离故与转置交换)、
       (b) 本文件 fp64 实现 —— 互相吻合 rel=3.0e-14, 共同与 ATen 分歧
       rel=1.98。该矩阵(5 组 shape × out_h ∈ {1,3,5,in_h} × ac ∈ {F,T},
       共 40 例)两条参考链最差 rel=9.0e-14。
    2. 缺陷与 ``alignCorners`` / 显式 scales 正交: ac=True 塌缩且 rel=1.85;
       显式 ``scales=(0.625,0.0625)`` 塌缩且 rel=1.98。
    3. 对偶场景 ``out_h==1 && out_w>1`` 正确(ATen vs 本文件 fp64 rel=1.1e-14)
       ⇒ 缺陷只在 W 轴。``out_w==1 && out_h==in_h`` 亦正确(rel=8.5e-14),
       但判据取 ``out_w == 1`` 这一侧的充分条件, 不按 ``out_h`` 细分。
    4. 上板 compare 日志(Ascend950PR, 无 plugin 裸跑自动 torch 映射):
       (2,3,8,16)→(5,1) fp32 在 RealIndex 26/27/28/29 四个相邻位置 golden
       同为 ``-0.004651321563869715``, NPU 输出各不相同(-0.2528 / -0.2347 /
       0.0366 / 0.0312) ⇒ **设备正确、ATen 错**, 该用例 FAIL 的根因在标杆侧。

    * scales: ``<=0 → None``; **正值直通不反转**, 依据见 ``_opt_scale``
      docstring(ACLNN 边界恒 OUT/IN, 架构差异在 ``GetWorkspaceSize`` 内消化)。
      走 fp64 回退时固定传 ``is_regbase=True``, 即"attr 为 OUT/IN"语义 —— 这是
      ACLNN 边界的无条件事实, 与实际 SoC 无关, 故不查 ``_is_regbase_soc()``。
      ``alignCorners=True`` 时统一传 None: 实测 ATen 在 ac=True 下对
      ``(2.0,2.0) / (0.5,0.5) / (3.0,0.25) / (1e-3,123.0)`` 四组 scales 的输出
      与 ``scales=None`` 逐位相同(rel=0.0); 本文件 fp64 实现同样忽略
      (``_aa_scale`` 的 ac 分支先返回)。
    * dtype: ``aten::_upsample_bicubic2d_aa`` CPU **不 dispatch fp16/bf16**
      (实测 ``RuntimeError: "compute_index_ranges_weights" not implemented
      for 'Half'/'BFloat16'``), 且 ``--golden-mode`` 默认 ``Enable`` 不抬精度
      ⇒ 抬升必须在入口内部完成: fp16/bf16 → fp32(对齐 NPU kernel 的 float
      权重/累加), fp32 → fp64(取最高精度标杆), 结果回原 dtype。
      fp64 回退分支内部恒以 fp64 累加(``_upsample_bicubic2d_aa_nchw``
      对入参 ``astype(np.float64)``), 即低精度入参实际得到的是 ≥fp32 的精度,
      偏严不偏松, 与 kernel 入口的既有做法一致。

      **为什么 fp32 入参抬 fp64 是安全的(与 kernel 入口的 fp32 口径不同)** ——
      注意这里**不能**用"AA 权重全正、无抵消"来论证, 该说法实测为假: Keys
      核含负瓣, AA 各轴权重的负权占比均值约 0.46, 下采样
      (32,28)→(16,14) 的逐点抵消比 ``sum|w x| / |sum w x|`` 可达 9.4e5。
      真正的依据是**口径差不是主导误差项**(TTK ``stat_rel_err`` 口径:
      ``rel = |a-g|/(|g|+1e-7)``, ``mere < th`` 且 ``mare < 10*th``):
        - 把"另一种同样合法的 fp32 实现"(本文件 AA 数学, 反序累加)分别与
          fp32 口径标杆 A、fp64 口径标杆 B 比, 口径差 ``gap(A,B)`` 与实现差
          ``gap(P,A)/gap(P,B)`` 的比值在 5 组形状里 4 组 < 1
          (0.145~0.429), 即抵消点处的误差由 ``eps_fp32 × 抵消比`` 决定,
          任何 fp32 消费方都已携带该量, 不由口径引入;
        - 改回 fp32 口径并不更安全: down2x 上 fp32 口径对该 proxy 的 mare
          预算占用 18.2x, fp64 口径 12.6x(fp64 反而更优, 5 组里 4 组如此);
        - 实测上板余量充裕: 本 csv 17 个 fp32 用例最差 ``mere`` 仅占预算
          2.26%、最差 ``mare`` 占 16.3%(Ascend950PR, 连续 5 次逐例判定逐位
          一致, 26/26 PASS)。
      遗留风险(如实记录): 在均匀随机数据上, 单看口径差本身,
      (32,28)→(16,14) 一例的 ``mare`` 可达 1.08e-2 = mare 预算的 5.4 倍 ——
      即该形状的安全性是"ttk 实际取数未落到病态抵消点"的经验结论, 不是
      结构性保证。若后续该族出现间歇 mare 越界, 首选处置是按 ops-precision-
      standard 对高抵消比位置改用绝对误差兜底, 而不是切换口径(已验证无效)。
    * 布局: aclnn 侧 ``CheckShape`` 只放行 ``FORMAT_ND`` / ``FORMAT_NCHW``
      (``aclnn_upsample_bicubic2d_aa.cpp``:101-107)且 ``OP_CHECK_WRONG_DIMENSION``
      固定 rank 4, 故无 NHWC 分支; 见到其他 format 直接报错而不静默当 NCHW。
    * tolerance 沿用 kernel 侧既有值(同一数学语义、同一后端精度)。
    """

    @staticmethod
    def golden(x, outputSize, alignCorners, scalesH, scalesW, out=None, **kwargs):
        """ACLNN golden; ``out`` 形参仅占位(ttk 传入输出 tensor), 内容不使用。

        **kwargs(aclnn 路径可用): short_soc_version, testcase_name,
        tensor_dtypes, tensor_formats, scalar_dtypes, use_torch
        —— 注意**没有** full_soc_version(kernel 路径才有), 本入口不依赖它。
        """
        t = x.detach().cpu() if torch.is_tensor(x) else torch.as_tensor(x)
        if t.dim() != 4:
            raise ValueError(
                f"aclnnUpsampleBicubic2dAA expects a 4-D NCHW input, got {tuple(t.shape)}"
            )
        fmt = _aclnn_format(kwargs)
        if fmt not in _ACLNN_FORMATS:
            raise ValueError(
                f"aclnnUpsampleBicubic2dAA only supports ND/NCHW storage format "
                f"(aclnn CheckShape), got {fmt!r}"
            )

        out_h, out_w = _normalize_output_size(outputSize)

        n_, c_, in_h_, in_w_ = (int(v) for v in t.shape)
        verdict = _aa_zero_verdict(n_, c_, in_h_, in_w_, out_h, out_w)
        if verdict == "reject":
            return [None]  # 见 _AA_ZERO_REJECT
        if verdict == "empty":
            return [torch.zeros((n_, c_, out_h, out_w), dtype=t.dtype)]

        orig_dtype = t.dtype
        reduced = orig_dtype in (torch.float16, torch.bfloat16)
        # ATen CPU 不支持 fp16/bf16 → fp32; fp32 → fp64 取最高精度标杆
        t = t.contiguous().to(torch.float32 if reduced else torch.float64)

        ac = bool(alignCorners)
        s_h = None if ac else _opt_scale(scalesH)
        s_w = None if ac else _opt_scale(scalesW)

        if out_w == 1:
            # ATen CPU H-collapse defect (class docstring item 1~4).
            # is_regbase=True 固定"attr 为 OUT/IN", 即 ACLNN 边界语义。
            y = _upsample_bicubic2d_aa_nchw(
                t.numpy(),
                out_h,
                out_w,
                ac,
                0.0 if s_h is None else s_h,
                0.0 if s_w is None else s_w,
                True,
            )
            y_t = torch.from_numpy(y)
        else:
            y_t = torch.ops.aten._upsample_bicubic2d_aa(t, [out_h, out_w], ac, s_h, s_w)

        return [y_t.contiguous().to(orig_dtype)]

    class ThirdPartyImpl:
        """ATen 适配器, 形参按 **aclnn C 头**命名(camel ``scalesH``/``scalesW``)。

        数学核与文件内 golden 的非 ``out_w==1`` 分支同源(见 ``_tp_aa_aten``),
        覆盖参数适配/绑定路径。**本族无独立框架标杆**(Pillow 不在 ttk 的
        provider 别名表内), 原因见共享核上方的注释块。

        ``outputSize[1] == 1`` 有据拒绝(ATen CPU H 方向塌缩), 见
        ``_AA_TP_W1_REJECT``。
        """

        def __init__(self, *, device=None, **kwargs):
            self.device = device or "cpu"

        def __call__(self, x, outputSize, alignCorners, scalesH, scalesW, **kwargs):
            oh, ow = _normalize_output_size(outputSize)
            if ow == 1:
                raise NotImplementedError(_AA_TP_W1_REJECT)
            ac = bool(alignCorners)
            return [
                _tp_aa_aten(
                    x,
                    oh,
                    ow,
                    ac,
                    None if ac else _opt_scale(scalesH),
                    None if ac else _opt_scale(scalesW),
                    self.device,
                )
            ]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        # 与 kernel 入口同值: 同一数学语义、同一后端精度, 无理由放宽。
        "float32": {"standard": "stat_rel_err", "threshold": 2e-4},
        "float16": {"standard": "stat_rel_err", "threshold": 0.0009765625},
        "bfloat16": {"standard": "stat_rel_err", "threshold": 0.0078125},
    }


# ------------------------------------------------------- Torch E2E (框架) 入口
class E2eUpsampleBicubic2dAATestSpec:
    """``torch.ops.aten._upsample_bicubic2d_aa`` 的框架 E2E 入口 (NCHW)。

    被测对象是**框架调用链**: ttk 把输入搬到 ``npu:0`` 后调用
    ``torch.ops.aten._upsample_bicubic2d_aa``, 经 torch_npu 适配层落到
    ``aclnnUpsampleBicubic2dAA`` → 设备 kernel ``UpsampleBicubic2dAA``。
    本 golden 只在 **CPU** 上算参考值, 与被测路径完全分离。

    **重载**: 绑定 ``.default``
    ``aten::_upsample_bicubic2d_aa(Tensor self, SymInt[2] output_size, bool
    align_corners, float? scales_h=None, float? scales_w=None) -> Tensor``。
    注意 API 名**带前导下划线**(``_upsample_bicubic2d_aa``), 这是 ATen 的私有
    API 命名, csv ``api_name`` 必须原样写 ``torch.ops.aten._upsample_bicubic2d_aa``。

    * **不注册 ``.vec``**: composite + PrivateUse1 无 kernel, 会被分解。
    * ⚠ **CPU fallback 必须逐例核查**: 本族存在"芯片不支持 ⇒ 整体回落 CPU"的
      风险, 且**输入/输出在 NPU 上不能证明计算发生在 NPU**。验收必须以
      profiling 的设备侧 kernel 落点为准; 若实测为 CPU fallback, 记为**未覆盖
      目标实现**并保留为缺口, **不得**计为 E2E 通过(需求明文要求)。
      另: 本算子在 ops-cv 侧**无 ``REG_OP`` 定义**(缺口 **G-E3**), 故图模式路径
      天然不可达, 与 eager 分开记账。
    * **antialias 语义保持**: 不退化为普通 bicubic, 不改 ``_CUBIC_A``
      (AA 用 PIL/Keys 的 -0.5, **不是** -0.75)。
    * **``out_w == 1`` 用例全部保留, 不删不排除**(验收明文要求), golden 走
      文件内 fp64 独立实现: ATen CPU 在 ``out_w == 1 && out_h != in_h`` 时 H
      方向塌缩(torch 2.7.1 实测, 证据链见本文件 aclnn 入口 docstring 第 1~4 条,
      含上板 compare 日志证明**设备正确、ATen 错**)。
      这是**标杆侧的已知缺陷**, 不是被测实现的问题 —— 因此只换标杆, 不删用例。
    * **scales**: ``<=0 → None``, 正值**直通不反转**。ATen 的
      ``compute_scales_value`` 同样按 ``scale > 0 → 1/scale`` 解读, 即 ``scales``
      边界语义恒为 OUT/IN, 与 ACLNN 边界一致; 故 fp64 回退分支固定传
      ``is_regbase=True``("attr 为 OUT/IN"), 与实际 SoC 无关。
      ``align_corners=True`` 时 ATen 与本文件 fp64 实现一致忽略 scales
      (aclnn 入口已实证四组 scales 逐位相同), 此处直通不特判。
    * **dtype**: ``aten::_upsample_bicubic2d_aa`` CPU **不 dispatch fp16/bf16**
      (实测 ``RuntimeError: "compute_index_ranges_weights" not implemented for
      'Half'/'BFloat16'``) ⇒ 抬升必须在入口内部完成: fp16/bf16 → fp32
      (对齐 NPU kernel 的 float 权重/累加), fp32 → fp64(取最高精度标杆),
      结果回原 dtype。与本文件 aclnn 入口同口径(该口径的余量实测见 aclnn
      docstring, 含遗留风险的如实记录)。
    * **布局**: aclnn 侧 ``CheckShape`` 只放行 ``ND``/``NCHW`` 且固定 rank 4,
      无 NHWC 分支; e2e csv 沿用源用例的 ``ND``/``NCHW`` 标注(两者的 shape 本身
      就是 NCHW 维序, H,W 在 dim 2,3), 不引入 channels_last 逻辑维。
    * tolerance 沿用 kernel/aclnn 入口同值, 无放宽理由。
    """

    @staticmethod
    def golden(
        x,
        output_size,
        align_corners,
        scales_h=None,
        scales_w=None,
        out=None,
        **kwargs,
    ):
        t = x.detach().cpu() if torch.is_tensor(x) else torch.as_tensor(x)
        if t.dim() != 4:
            raise ValueError(
                f"aten._upsample_bicubic2d_aa expects a 4-D NCHW input, "
                f"got {tuple(t.shape)}"
            )

        out_h, out_w = _normalize_output_size(output_size)

        n_, c_, in_h_, in_w_ = (int(v) for v in t.shape)
        verdict = _aa_zero_verdict(n_, c_, in_h_, in_w_, out_h, out_w)
        if verdict == "reject":
            return [None]  # 见 _AA_ZERO_REJECT
        if verdict == "empty":
            return [torch.zeros((n_, c_, out_h, out_w), dtype=t.dtype)]

        orig_dtype = t.dtype
        reduced = orig_dtype in (torch.float16, torch.bfloat16)
        # ATen CPU 不支持 fp16/bf16 → fp32; fp32 → fp64 取最高精度标杆
        t = t.contiguous().to(torch.float32 if reduced else torch.float64)

        ac = bool(align_corners)
        s_h = _opt_scale(scales_h)
        s_w = _opt_scale(scales_w)

        if out_w == 1:
            # ATen CPU H-collapse defect; 见本类与 aclnn 入口 docstring。
            # is_regbase=True 固定"attr 为 OUT/IN", 即 ATen scales 的边界语义。
            y = _upsample_bicubic2d_aa_nchw(
                t.numpy(),
                out_h,
                out_w,
                ac,
                0.0 if s_h is None else s_h,
                0.0 if s_w is None else s_w,
                True,
            )
            y_t = torch.from_numpy(y)
        else:
            y_t = torch.ops.aten._upsample_bicubic2d_aa(t, [out_h, out_w], ac, s_h, s_w)

        return [y_t.contiguous().to(orig_dtype)]

    class ThirdPartyImpl:
        """ATen 适配器, 形参按 **ATen schema** 命名(snake ``scales_h``/``scales_w``)。

        数学核与文件内 golden 的非 ``out_w==1`` 分支同源(见 ``_tp_aa_aten``),
        覆盖参数适配/绑定路径。**本族无独立框架标杆**(Pillow 不在 ttk 的
        provider 别名表内), 原因见共享核上方的注释块。

        scales **直通不特判**(与本入口 golden 同口径): ATen 在 ac=True 下忽略
        scales, 被测侧同样忽略, 特判反而会掩盖不一致。

        ``output_size[1] == 1`` 有据拒绝(ATen CPU H 方向塌缩), 见
        ``_AA_TP_W1_REJECT``; 对应用例本身**保留**, 由文件内 fp64 golden 覆盖。

        **实例槽改名为 ``impl``**: ATen schema 的首参叫 ``self``
        (``aten._upsample_bicubic2d_aa.default`` 的 arguments 为 ``['self',
        'output_size', 'align_corners', 'scales_h', 'scales_w']``), 而 ttk 的
        ``bind_params`` 按**名**绑定, 绑定方法的签名里实例参已被 Python 去掉,
        于是池里的 ``self`` 必须对上一个真的叫 ``self`` 的形参。写成 ``x`` 会
        ``UnknownParamError: parameter 'x' of __call__ has no value``, 整条三方
        标杆静默不跑(本机经真实 ``invoke_third_party`` 实测)。
        本仓库 ``resize_nearest_neighbor_v2`` / ``upsample_nearest3d`` 同此写法。

        注意 ``__init__`` 的首参**必须仍叫 ``self``**: ``bind_params`` 只在
        "形参名是 self 且 func 是未绑定 ``__init__``"时才跳过实例占位
        (``xpu._self_is_instance_slot``), 改名会让它按名去池里找, 找不到即
        ``UnknownParamError``, 而该异常对 ``__init__`` 是静默兜底成空参 ——
        ``device`` 与 ``input_formats`` 会一起丢失。
        """

        def __init__(self, *, device=None, **kwargs):
            self.device = device or "cpu"

        def __call__(
            impl,
            self,
            output_size,
            align_corners,
            scales_h=None,
            scales_w=None,
            **kwargs,
        ):
            oh, ow = _normalize_output_size(output_size)
            if ow == 1:
                raise NotImplementedError(_AA_TP_W1_REJECT)
            return [
                _tp_aa_aten(
                    self,
                    oh,
                    ow,
                    bool(align_corners),
                    _opt_scale(scales_h),
                    _opt_scale(scales_w),
                    impl.device,
                )
            ]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        # 与 kernel/aclnn 入口同值: 同一数学语义、同一后端精度, 无理由放宽。
        "float32": {"standard": "stat_rel_err", "threshold": 2e-4},
        "float16": {"standard": "stat_rel_err", "threshold": 0.0009765625},
        "bfloat16": {"standard": "stat_rel_err", "threshold": 0.0078125},
    }
