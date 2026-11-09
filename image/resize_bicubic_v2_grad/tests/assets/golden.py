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

"""Golden for resize_bicubic_v2_grad — torch ATen IR implementation.

Two entries:
  * ``resize_bicubic_v2_grad``          —— kernel path (Ascend IR)
  * ``aclnnUpsampleBicubic2dBackward``  —— ACLNN path (see
    ``AclnnUpsampleBicubic2dBackwardTestSpec`` at the bottom of this file)

The golden calls ``torch.ops.aten.upsample_bicubic2d_backward`` directly (per
PR !1808 review: the Ascend IR op is benchmarked against the ATen IR op).

Equivalence (Ascend IR ResizeBicubicV2Grad <-> aten::upsample_bicubic2d_
backward), verified point-wise in float64 against an independent
high-precision reference AND torch autograd (rel <= 2.9e-14 across half_pixel
/ align_corners / lenDst==1 / non-identity explicit-scales / default-scale
identity shapes; adjoint identity <grad, x> == <g, forward(x)> at 1e-16).
Same-size explicit scales are an ATen backward exception: backward copies
the gradient although CPU forward still resamples. The adapters preserve that
ATen contract rather than claiming an adjoint equivalence for this case.

Parameter mapping:
* grads (tensor input, forward-output shape)   -> grad_output
* original_image (tensor input, forward-input  -> only its shape is used:
  shape)                                          input_size (int[4]); the
                                                  Ascend infershape defines
                                                  y.shape == original_image
                                                  .shape and y.dtype ==
                                                  grads.dtype, matching the
                                                  ATen backward output.
* align_corners (attr, bool)                   -> align_corners
* scales (attr ListFloat, OUT/IN ratio)        -> scales_h / scales_w with
  the same OUT/IN semantics as the forward op (ATen compute_scales_value:
  value > 0 -> src/dst scale = 1/value). Non-positive -> None.
* align_corners=True and lenDst==1: the Ascend grad tiling sets scale=0
  while ATen area_pixel_compute_scale also returns 0 — identical; the
  forward-side in/out variant degenerates the same (dst=0 -> src=0).
* Layout: ATen is NCHW; NHWC inputs are transposed around the aten call.
* Golden dtype fp16/bf16: lifted to fp32, then quantised back to grads dtype.
  Inputs already promoted by the harness retain their supplied precision.
* Third-party dtype: preserved, including fp16/bf16. PyTorch CUDA computes
  coefficients in opmath precision but casts each contribution to scalar_t
  before atomic addition (ATen/native/cuda/UpSample.cuh,
  upsample_increment_value_bounded). Promoting the input would replace the
  native low-precision benchmark with a different fp32-accumulation operation.
  Likewise, the NPU scatter uses T_DATA atomics, whereas its gather uses a
  float accumulator; the two paths do not share one accumulation precision.
* NaN/Inf: retain ATen's four-tap scatter, including zero-weight products.
  Skipping those products can change NaN propagation. Finite extreme inputs
  can also overflow intermediate device accumulators; neither discrepancy
  justifies replacing the reference with the device result.
"""

import numpy as np
import torch

__spec__ = {
    "resize_bicubic_v2_grad": "ResizeBicubicV2GradTestSpec",
    "aclnnUpsampleBicubic2dBackward": "AclnnUpsampleBicubic2dBackwardTestSpec",
    "torch.ops.aten.upsample_bicubic2d_backward": (
        "E2eUpsampleBicubic2dBackwardTestSpec"
    ),
}

_CHANNELS_LAST_FORMATS = ("NHWC",)


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


def _is_nhwc(input_formats):
    fmts = input_formats or []
    if isinstance(fmts, str):
        x_format = fmts
    else:
        try:
            x_format = fmts[0]
        except (TypeError, IndexError, KeyError):
            return False
    return str(x_format).strip().upper() == "NHWC"


def _normalize_scales(scales):
    """Accept list/tuple/ndarray/None; return per-axis floats (default 0.0)."""
    if scales is None:
        return (0.0, 0.0)
    if isinstance(scales, np.ndarray):
        scales = scales.flatten().tolist()
    if isinstance(scales, (int, float)):
        return (float(scales), float(scales))
    vals = [float(v) for v in scales]
    if len(vals) == 1:
        return (vals[0], vals[0])
    return (vals[0], vals[1])


class ResizeBicubicV2GradTestSpec:
    """TestSpec for resize_bicubic_v2_grad (kernel path, aten IR golden)."""

    def golden(
        grads, original_image, *, align_corners=False, scales=(0.0, 0.0), **kwargs
    ):
        """Golden for resize_bicubic_v2_grad via aten.upsample_bicubic2d_backward.

        Parameters follow resize_bicubic_v2_grad_def.cpp without outputs.
        y has the shape of original_image and the dtype of grads.

        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name, golden_mode
        """
        grads = np.asarray(grads)
        original_image = np.asarray(original_image)
        if grads.ndim != 4 or original_image.ndim != 4:
            raise ValueError(
                f"grads/original_image must be 4-D, got {grads.shape} / {original_image.shape}"
            )

        orig_dtype = str(grads.dtype)
        g_t = _to_torch(grads)
        o_t = _to_torch(original_image)
        if g_t.dtype in (torch.float16, torch.bfloat16):
            g_t = g_t.to(torch.float32)
            o_t = o_t.to(torch.float32)

        nhwc = _is_nhwc(kwargs.get("input_formats"))
        if nhwc:
            g_t = g_t.permute(0, 3, 1, 2).contiguous()
            o_t = o_t.permute(0, 3, 1, 2).contiguous()
        input_size = list(o_t.shape)

        scales_pair = _normalize_scales(scales)
        s_h = scales_pair[0] if scales_pair[0] > 0 else None
        s_w = scales_pair[1] if scales_pair[1] > 0 else None

        y_t = torch.ops.aten.upsample_bicubic2d_backward(
            g_t, list(g_t.shape[2:]), input_size, bool(align_corners), s_h, s_w
        )

        if nhwc:
            y_t = y_t.permute(0, 2, 3, 1).contiguous()
        if orig_dtype == "float16":
            y_t = y_t.to(torch.float16)
        elif orig_dtype == "bfloat16":
            y_t = y_t.to(torch.bfloat16)
        return [_to_numpy(y_t, orig_dtype)]

    class ThirdPartyImpl:
        """Low-level ATen bicubic backward adapter.

        ATen accepts output size and explicit scales together, unlike the
        ``F.interpolate`` wrapper. It copies same-size gradients even with
        positive scales; this is the backward API's behavior, not the adjoint
        of the resampling performed by CPU forward for those scales. This adapter
        shares ATen with the golden and is not an independent numerical oracle.

        Preserve the input dtype: CUDA's native half/bfloat16 atomics are part
        of the benchmark. Only the CPU golden may promote its inputs.

        Layout: kernel cross_check **does** carry per-input format, so the
        NHWC normalisation below is live on that path. Chain verified in the
        installed ttk: ``ttk/core_modules/kernel/worker.py::_do_xpu_profiling``
        passes ``input_formats=context.input_formats`` to
        ``ttk/remote/client.py::dispatch_xpu``, which forwards it to
        ``ttk/remote/dispatcher.py::_build_input_schema`` (embedded as the
        ``format`` field of each X-Input-Schema entry), and
        ``ttk/remote/server/executor.py::_compose_kwargs`` re-injects it as the
        ``input_formats`` kwarg **for class-form implementations** (this class).
        ``input_formats`` is in fact the only format key a third-party provider
        ever sees — ``tensor_formats`` is injected only into local golden calls.
        """

        def __init__(
            self, *, align_corners=False, scales=(0.0, 0.0), device=None, **kwargs
        ):
            self.align_corners = bool(align_corners)
            scales_pair = _normalize_scales(scales)
            # ac=True -> ATen ignores scales, same as kernel/aclnn
            if self.align_corners:
                self.scales = (None, None)
            else:
                self.scales = tuple(s if s > 0 else None for s in scales_pair[:2])
            self.channels_last = _is_nhwc(kwargs.get("input_formats"))
            # device 必须**显式声明为形参**才会被注入: ttk 的
            # bind_params 只对 _FRAMEWORK_RESERVED={"device"} 这个**形参名**
            # 注入设备, 它不在 inputs/attrs 里, 因此 `kwargs.get("device")`
            # 恒为空 —— 实测会静默退回 cpu, 配置的设备根本不生效。
            self.device = device or "cpu"

        def __call__(self, grads, original_image, **kwargs):
            g = grads if torch.is_tensor(grads) else _to_torch(np.asarray(grads))
            input_size = list(original_image.shape)
            out_dtype = g.dtype
            g = g.to(self.device)
            if self.channels_last:
                if g.dim() != 4 or len(input_size) != 4:
                    raise ValueError(
                        f"NHWC expects 4-D inputs, got {tuple(g.shape)} / {tuple(input_size)}"
                    )
                g = torch.movedim(g, -1, 1).contiguous()
                input_size = [input_size[i] for i in (0, 3, 1, 2)]
            # Use native ATen precision; original_image supplies only shape.
            grad_in = torch.ops.aten.upsample_bicubic2d_backward(
                g.contiguous(),
                [int(g.shape[2]), int(g.shape[3])],
                [int(v) for v in input_size],
                self.align_corners,
                self.scales[0],
                self.scales[1],
            )
            if self.channels_last:
                grad_in = torch.movedim(grad_in, 1, -1).contiguous()
            return [grad_in.to(out_dtype)]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        # fp16/bf16 follow the TTK official dtype table (2^-10 / 2^-7),
        # verified on NPU (Ascend950PR) with multiple seeds.
        "float32": {"standard": "stat_rel_err", "threshold": 1e-4},
        "float16": {"standard": "stat_rel_err", "threshold": 0.0009765625},
        "bfloat16": {"standard": "stat_rel_err", "threshold": 0.0078125},
    }


# ---------------------------------------------------------------- ACLNN entry
def _opt_scale(v):
    """aclnn OUT/IN scale -> ATen Optional[float]; <=0 means "unspecified".

    Pass-through, NOT inverted: the ACLNN boundary always carries OUT/IN and
    ATen ``compute_scales_value`` reads it the same way (``scale > 0 ->
    src/dst = 1/scale``); architecture differences are absorbed inside
    ``GetWorkspaceSize``. Semantics verified in fp64 (a precision-independent
    equivalence check, unrelated to this entry's fp32 compute dtype;
    torch 2.7.1, inputSize H/W=8 -> outputSize 4): ``scales=0.5`` is bitwise
    identical to the default (maxdiff 0.0) while ``scales=2.0`` differs by
    2.91 — confirming OUT/IN.
    """
    if v is None:
        return None
    s = float(v.reshape(-1)[0]) if torch.is_tensor(v) else float(v)
    return s if s > 0.0 else None


def _aclnn_format(kwargs):
    """ACLNN / E2E 通路的首输入 format。两条通路的键名**不同**, 必须都读:

      * ``input_formats`` —— 远端三方执行注入(``executor._compose_kwargs``, 对
        **任何**通路的类形态实现都注入), 即 ``third_party`` provider 能看到的
        唯一 format 来源;
      * ``tensor_formats`` —— 本地 golden 调用注入, provider 侧**拿不到**。

    只读 ``tensor_formats`` 会让 provider 侧恒按缺省 ND 解读, NHWC 用例被静默按
    NCHW 算完当成通过。
    """
    for key in ("input_formats", "tensor_formats"):
        fmts = kwargs.get(key)
        if not fmts:
            continue
        v = fmts if isinstance(fmts, str) else (fmts[0] if len(fmts) > 0 else None)
        if v:
            return str(v).strip().upper()
    return "ND"


# -------------------------------------- third_party shared math core (2D bwd)
#
# The ACLNN and E2E entries share the two cores below; only the parameter
# NAMES differ (aclnn is camel ``gradOut``/``outputSize``/``inputSize``/
# ``alignCorners``/``scalesH``/``scalesW``, E2E is ATen snake
# ``grad_output``/``output_size``/``input_size``/``align_corners``/
# ``scales_h``/``scales_w``). Sharing the math is allowed, but **each
# declaration must follow its own signature**, otherwise ttk's bind_params
# rejects it with UnknownParamError (a parameter name not in inputs ∪ attrs).
#
# ``device`` must be declared as an EXPLICIT parameter to be injected:
# bind_params only injects for the parameter name in
# _FRAMEWORK_RESERVED={"device"}, which is not part of inputs/attrs, so
# ``kwargs.get("device")`` is always empty (measured: silently falls back
# to cpu).


def _tp_bicubic_grad_aten(
    grads, input_size, align_corners, s_h, s_w, device, channels_last
):
    """Low-level ATen bicubic backward. Same source as the in-file golden.

    Cross-check value: it covers the **parameter adaptation and binding path**
    (camel/snake names, scale sentinels, NHWC normalisation, inputSize
    reordering, dtype handling). The math is shared with the golden, so it is
    NOT an independent numerical benchmark. No TF provider is registered: the
    op proto (``op_graph/resize_bicubic_v2_grad_proto.h``) declares
    "Compatible with PyTorch upsample_bicubic2d_backward operator." only.
    Promoting an ATen golden to fp64 does not make it independent of ATen.

    Unlike the CPU golden, the third-party call preserves the input dtype.
    In particular, CUDA fp16/bf16 atomics must not be replaced with fp32
    accumulation by promoting the gradient before calling ATen.
    """
    g = grads if torch.is_tensor(grads) else _to_torch(np.asarray(grads))
    g = g.to(device or "cpu")
    orig = g.dtype
    isz = [int(v) for v in input_size]
    if channels_last:
        if g.dim() != 4:
            raise ValueError(f"NHWC expects a 4-D gradOut, got {tuple(g.shape)}")
        g = torch.movedim(g, -1, 1)
        # {N,H,W,C} -> {N,C,H,W}
        isz = [isz[0], isz[3], isz[1], isz[2]]
    y = torch.ops.aten.upsample_bicubic2d_backward(
        g.contiguous(),
        [int(g.shape[2]), int(g.shape[3])],
        isz,
        bool(align_corners),
        s_h,
        s_w,
    )
    if channels_last:
        y = torch.movedim(y, 1, -1)
    return y.contiguous().to(orig)


class AclnnUpsampleBicubic2dBackwardTestSpec:
    """aclnnUpsampleBicubic2dBackward(gradOut, outputSize, inputSize,
    alignCorners, scalesH, scalesW, gradInput).

    Single branch ``aten::upsample_bicubic2d_backward`` — the aclnn coordinate
    semantics are fully determined by ``alignCorners`` and ATen covers both
    values natively. No in-file fallback branch, and no copy of any kernel
    identity shortcut: ATen is measured bit-exact for in == out (fp64 7x7
    grad maxdiff 0.0).

    * The trailing output parameter is named ``gradInput`` (not ``out``), per
      the C header ``aclnnUpsampleBicubic2dBackwardGetWorkspaceSize(...,
      aclTensor* gradInput, ...)``. Its content is ignored.
    * scales: ``<=0 -> None``; ``alignCorners=True`` passes None because ATen
      ignores scales in that mode (measured bitwise identical for
      scales (2.0, 2.0)), matching kernel / aclnn.
    * dtype: **fp16/bf16 -> fp32; fp32 stays fp32** (no fp64 lift), result
      cast back to the ``gradOut`` dtype. ``--golden-mode`` defaults to no
      promotion, so the lift happens inside this entry.

      This matches the kernel entry's convention in this same file on
      purpose, and is the one place where this op family deliberately
      departs from the "lift to fp64" rule used by the bilinear / linear
      aclnn entries. Rationale: bicubic carries an EXPLICIT fp32
      ``threshold`` of 1e-4 (see ``tolerance`` above), calibrated against
      the kernel golden's fp32-in/fp32-compute convention — golden and
      device round the same way. bilinear / linear may lift to fp64 because
      their kernel goldens already compute in fp64 AND their tolerance uses
      the default ``stat_rel_err`` with no explicit threshold; bicubic is
      the exception and stays aligned with its own kernel golden.

      Why lifting the reference would break it: the backward scatter has
      near-total cancellation positions when down-sampling. Measured on
      this op's (2,7,5,16) -> [3,9] weight matrix (adjoint verified to
      1.3e-15), the worst position reaches a 285x cancellation ratio
      (``sum|terms| = 6.36`` against ``|result| = 2.23e-2``), and the fp32
      deviation stays inside the deterministic rounding bound
      ``n*eps*sum|terms|`` at 35/35 positions. Against an fp64 reference
      that tiny denominator inflates ``mare`` (bound = 10*threshold) with no
      device error present — CPU alone, "fp32 compute vs fp64 compute"
      exceeds the 1e-3 mare bound on 4 of 12 seeds for that shape. Keeping
      the reference at fp32 keeps ``mare`` controlled while ``mere`` stays
      ~1e-6, so semantic regressions remain detectable.
    * Layout: ``gradOut`` is normalised NHWC -> NCHW with ``movedim`` and
      moved back afterwards. ``inputSize`` comes from the attribute in the
      tensor's own storage order, so under NHWC it must be REORDERED from
      {N, H, W, C} to {N, C, H, W} before reaching ATen — feeding it
      unreordered raises ``Expected grad_output to have the same shape as
      output; output.size(1) = 8 but got grad_output.size(1) = 16``
      (reproduced on torch 2.7.1 with gradOut (1,4,4,16) and inputSize
      [1,8,8,16]). ``outputSize`` is always the logical [H, W].
    """

    @staticmethod
    def golden(
        gradOut,
        outputSize,
        inputSize,
        alignCorners,
        scalesH,
        scalesW,
        gradInput=None,
        **kwargs,
    ):
        g = (
            gradOut.detach().cpu()
            if torch.is_tensor(gradOut)
            else torch.as_tensor(gradOut)
        )
        orig_dtype = g.dtype
        channels_last = _aclnn_format(kwargs) in _CHANNELS_LAST_FORMATS

        isize = [int(v) for v in np.asarray(inputSize).reshape(-1)]
        assert len(isize) == 4, f"inputSize must carry 4 dims, got {isize}"
        if channels_last:
            if g.dim() != 4:
                raise ValueError(f"NHWC expects a 4-D gradOut, got {tuple(g.shape)}")
            g = torch.movedim(g, -1, 1)
            # {N,H,W,C} -> {N,C,H,W}
            isize = [isize[0], isize[3], isize[1], isize[2]]
        # fp16/bf16 -> fp32; fp32 (and fp64 under golden-mode Promote) kept
        # as-is. Same convention as the kernel entry above; see the class
        # docstring for why this family does NOT lift fp32 to fp64.
        if g.dtype in (torch.float16, torch.bfloat16):
            g = g.to(torch.float32)
        g = g.contiguous()

        osize = [int(v) for v in np.asarray(outputSize).reshape(-1)]
        assert len(osize) >= 2, "outputSize must carry [H_out, W_out]"
        ac = bool(alignCorners)
        s_h = None if ac else _opt_scale(scalesH)
        s_w = None if ac else _opt_scale(scalesW)

        y = torch.ops.aten.upsample_bicubic2d_backward(
            g, osize[:2], isize, ac, s_h, s_w
        )

        if channels_last:
            y = torch.movedim(y, 1, -1)
        return [y.contiguous().to(orig_dtype)]

    class ThirdPartyImpl:
        """ATen adapter, parameters named after the **aclnn C header** (camel).

        Math core shared with the in-file golden (see
        ``_tp_bicubic_grad_aten``); covers the parameter adaptation / binding
        path. Per the op proto this entry is aligned to PyTorch only, so
        there is no TF sibling or independent numerical oracle here.
        """

        def __init__(self, *, device=None, **kwargs):
            self.device = device or "cpu"
            self.channels_last = _aclnn_format(kwargs) in _CHANNELS_LAST_FORMATS

        def __call__(
            self,
            gradOut,
            outputSize,
            inputSize,
            alignCorners,
            scalesH,
            scalesW,
            **kwargs,
        ):
            isz = [int(v) for v in np.asarray(inputSize).reshape(-1)]
            ac = bool(alignCorners)
            return [
                _tp_bicubic_grad_aten(
                    gradOut,
                    isz,
                    ac,
                    None if ac else _opt_scale(scalesH),
                    None if ac else _opt_scale(scalesW),
                    self.device,
                    self.channels_last,
                )
            ]

    third_party = {"torch": ThirdPartyImpl}

    # Same math semantics and same backend precision as the kernel entry —
    # no reason to relax; values carried over verbatim.
    tolerance = {
        "float32": {"standard": "stat_rel_err", "threshold": 1e-4},
        "float16": {"standard": "stat_rel_err", "threshold": 0.0009765625},
        "bfloat16": {"standard": "stat_rel_err", "threshold": 0.0078125},
    }


# ------------------------------------------------------- Torch E2E (框架) 入口
class E2eUpsampleBicubic2dBackwardTestSpec:
    """Framework E2E entry for ``torch.ops.aten.upsample_bicubic2d_backward``.

    What is under test is the **framework call chain**: ttk moves the upstream
    gradient to ``npu:0`` and calls
    ``torch.ops.aten.upsample_bicubic2d_backward``, which goes through the
    torch_npu adapter to ``aclnnUpsampleBicubic2dBackward`` and lands on the
    device kernel ``ResizeBicubicV2Grad``. This golden computes the reference
    on **CPU** only, fully separated from the path under test.

    **Overload**: bound to ``.default``
    ``aten::upsample_bicubic2d_backward(Tensor grad_output, SymInt[2]
    output_size, SymInt[4] input_size, bool align_corners, float?
    scales_h=None, float? scales_w=None) -> Tensor``.

    * **``.grad_input`` is never bound** (that overload is the CPU-fallback
      risk path) and **``.vec`` is not registered** (composite, decomposed).
      ttk's ``match_overload`` picks ``.default`` as long as the csv params
      carry no ``grad_input`` key — so the csv must not contain one.
    * **``input_size`` is the 4-element ``{N, C, H_in, W_in}``**: the e2e csv
      always uses NCHW logical dims, so **no** ``{N,H,W,C}→{N,C,H,W}``
      reordering is needed here (that is only required for the aclnn NHWC
      cases). The second entry of ``tensor_view_shapes`` must agree with
      ``input_size``, otherwise the path under test raises a shape error
      rather than silently computing something wrong.
    * **Upstream gradients are real non-uniform random values**
      (``(-3.0, 5.0)``), not constants — constant upstream gradients let
      scatter errors mask each other.
    * **dtype — deliberately NOT lifted to fp64**: ``fp16``/``bf16`` → ``fp32``
      and ``fp32`` stays ``fp32``, matching the kernel and aclnn entries in
      this file. The explicit fp32 ``threshold`` of 1e-4 was calibrated
      against that fp32-in/fp32-compute convention; lifting the reference to
      fp64 inflates ``mare`` at bicubic's cancellation positions with no
      device error present. See the aclnn entry's docstring for the
      measurement chain.
    * **Layout**: NHWC logical-dim cases are **not** converted into e2e (ATen
      always reads H,W from dims 2,3; transplanting silently changes
      semantics — gap **G-E8**); channels_last is covered by a standalone
      script, recorded separately.
    * **No bitwise-equality assertion**: Ascend950PR backward is
      non-deterministic by default (gap **G-E5**), so comparison is
      ``stat_rel_err`` tolerance only.
    * tolerance carried over verbatim from the kernel/aclnn entries.
    """

    @staticmethod
    def golden(
        grad_output,
        output_size,
        input_size,
        align_corners,
        scales_h=None,
        scales_w=None,
        out=None,
        **kwargs,
    ):
        g = (
            grad_output.detach().cpu()
            if torch.is_tensor(grad_output)
            else torch.as_tensor(grad_output)
        )
        orig_dtype = g.dtype
        assert g.dim() == 4, (
            f"aten.upsample_bicubic2d_backward expects a 4-D NCHW grad_output, "
            f"got {tuple(g.shape)}"
        )
        # fp16/bf16 -> fp32; fp32 kept as-is (see docstring: do NOT lift fp32).
        if g.dtype in (torch.float16, torch.bfloat16):
            g = g.to(torch.float32)
        g = g.contiguous()

        osize = [int(v) for v in np.asarray(output_size).reshape(-1)]
        isize = [int(v) for v in np.asarray(input_size).reshape(-1)]
        assert len(osize) >= 2, "output_size must carry [H_out, W_out]"
        assert len(isize) == 4, "input_size must carry 4 dims [N, C, H_in, W_in]"

        y = torch.ops.aten.upsample_bicubic2d_backward(
            g,
            osize[:2],
            isize,
            bool(align_corners),
            _opt_scale(scales_h),
            _opt_scale(scales_w),
        )
        return [y.contiguous().to(orig_dtype)]

    class ThirdPartyImpl:
        """ATen adapter, parameters named after the **ATen schema** (snake).

        Math core shared with the in-file golden (see
        ``_tp_bicubic_grad_aten``); covers the parameter adaptation / binding
        path. Per the op proto this entry is aligned to PyTorch only, so
        there is no TF sibling or independent numerical oracle here.

        The e2e csv always uses NCHW logical dims (see the class docstring,
        gap **G-E8**), so ``channels_last`` is fixed to False here.
        """

        def __init__(self, *, device=None, **kwargs):
            self.device = device or "cpu"

        def __call__(
            self,
            grad_output,
            output_size,
            input_size,
            align_corners,
            scales_h=None,
            scales_w=None,
            **kwargs,
        ):
            isz = [int(v) for v in np.asarray(input_size).reshape(-1)]
            ac = bool(align_corners)
            return [
                _tp_bicubic_grad_aten(
                    grad_output,
                    isz,
                    ac,
                    None if ac else _opt_scale(scales_h),
                    None if ac else _opt_scale(scales_w),
                    self.device,
                    False,
                )
            ]

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        "float32": {"standard": "stat_rel_err", "threshold": 1e-4},
        "float16": {"standard": "stat_rel_err", "threshold": 0.0009765625},
        "bfloat16": {"standard": "stat_rel_err", "threshold": 0.0078125},
    }
