#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Golden reference (spec-class form) for img_warp_resize (bilinear interpolation sampling, arch35).

Operator (REG_OP IMGWarpResize; op name img_warp_resize):
    For each output element (n, c, h, w), the fractional parts of the
    floating-point coordinates (x_lerp, y_lerp, both in [0, 1)) weight the
    pre-packed four corner pixels of img and combine them with a two-step
    linear interpolation (horizontal lerp, then vertical lerp).

    img:        [N, 4, C, H, W], float16/float32,
                dim1 = [top_left, top_right, bottom_left, bottom_right]
    warp_index: [N, 2, H, W], float32, dim1 = [x_float, y_float]
    warp_img:   [N, C, H, W], dtype follows img

Numerical contract (the operator's math, computed as a true value):
    - 16-bit float inputs (float16/bfloat16) are upcast to float32
      BEFORE any arithmetic (kernel __half2float); inputs arriving at a
      higher precision keep that precision through the chain;
    - the chain is plain float arithmetic on the current input values —
      a true-value reference of the operator's math, NOT a simulation of
      device-side roundings (those deviations are absorbed by the
      comparison thresholds);
    - coordinates keep arrival-precision floor semantics (negative and
      non-finite coordinates never pass through an int32 narrowing);
    - the result is returned rounded (round-to-nearest-even) to the
      CURRENT img dtype — the operator's dtype rule applied to the
      inputs as they arrive;
    - the interpolation weight broadcast along the C axis is a true data
      replication of the [N, H, W] weights.

Output dtype rule:
    warp_img follows the CURRENT img dtype (the operator's InferShape
    dtype rule applied to whatever dtype the inputs carry). The reference
    needs no knowledge of WHY the inputs carry their current dtype: a
    caller delivering inputs upcast one precision level simply gets the
    math computed and returned at that level.

Tolerance (cross_check, level L1 for both float16 and float32):
    a computing-type float operator compares three-way — device output
    vs this numpy reference vs the independent torch ThirdPartyImpl —
    whenever a third-party endpoint is available. "level" is an
    ecosystem-level annotation. Environments without a third-party
    endpoint may degrade to a two-way comparison externally; any such
    degradation happens outside this file, which stays
    environment-independent. The reference is a true-value model, so
    the device output is expected to agree with it within the
    comparison thresholds. Torch (ThirdPartyImpl) is the cross-check
    leg, not the reference oracle.

Registration:
    "img_warp_resize" — op name used by kernel-path suites;
    "IMGWarpResize"   — GE op type name (graph-mode suites name the op
                        by its GE type).

Import discipline: torch is imported lazily inside ThirdPartyImpl method
bodies only — this file is loaded inside the aclnn process where aclrt is
already initialized, and a module-level competitor-framework import is
unsafe there.
"""

import numpy

__spec__ = {
    "img_warp_resize": "ImgWarpResizeSpec",
    "IMGWarpResize": "ImgWarpResizeSpec",
}

INT32_MAX = numpy.iinfo(numpy.int32).max


def _validate(img, warp_index):
    """Check the operator's input contract (shape structure and N/H/W match).

    Raises ValueError on violation. Used by the third-party implementation
    only; the golden itself stays permissive so that negative cases are
    rejected by the operator under test, not by the reference.
    """
    if img.ndim != 5 or img.shape[1] != 4:
        raise ValueError(f"img must have shape (N, 4, C, H, W), got {img.shape}")
    if warp_index.ndim != 4 or warp_index.shape[1] != 2:
        raise ValueError(
            f"warp_index must have shape (N, 2, H, W), got {warp_index.shape}"
        )
    if (img.shape[0], img.shape[3], img.shape[4]) != (
        warp_index.shape[0],
        warp_index.shape[2],
        warp_index.shape[3],
    ):
        raise ValueError("img and warp_index must have equal N/H/W dimensions")
    if any(
        size == 0 for size in (img.shape[0], img.shape[2], img.shape[3], img.shape[4])
    ):
        raise ValueError("N/C/H/W dimensions must be non-zero")
    if img.shape[0] * img.shape[2] * img.shape[3] * img.shape[4] > INT32_MAX:
        raise ValueError("output element count must not exceed INT32_MAX")


class ImgWarpResizeSpec:
    """img_warp_resize test spec (kernel pathway, numpy inputs)."""

    def golden(img, warp_index, **kwargs):
        """Kernel golden for img_warp_resize.
        Parameters follow the operator definition (img_warp_resize_def.cpp)
        input order: img, warp_index (numpy.ndarray).
        kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
          input_formats, output_formats, input_ori_formats, output_ori_formats,
          input_dtypes, output_dtypes.

        img: numpy.ndarray, shape [N, 4, C, H, W], dtype float16/float32
             dim1=4=[top_left, top_right, bottom_left, bottom_right]
        warp_index: numpy.ndarray, shape [N, 2, H, W], dtype float32
             dim1=2=[x_float, y_float]
        returns: numpy.ndarray, shape [N, C, H, W], dtype follows img
        """
        out_dtype = img.dtype
        work = numpy.float32 if "float16" in img.dtype.name else img.dtype
        np_batch = img.shape[0]
        np_c = img.shape[2]
        np_h = img.shape[3]
        np_w = img.shape[4]
        np_area = np_h * np_w

        img_flat = img.reshape([np_batch, 4, np_c, np_area]).astype(work)
        warp_flat = warp_index.reshape([np_batch, 2, np_area])
        chain = numpy.promote_types(work, warp_flat.dtype)
        output = numpy.zeros([np_batch, np_c, np_area], dtype=chain)

        for batch_idx in range(np_batch):
            lerp_x_data = warp_flat[batch_idx, 0, :]  # x_float (w direction)
            lerp_y_data = warp_flat[batch_idx, 1, :]  # y_float (h direction)

            # Floor keeps the arrival dtype semantics (negative and
            # non-finite coordinates never pass through an int32 narrowing).
            lerp_x = lerp_x_data - numpy.floor(lerp_x_data)
            lerp_y = lerp_y_data - numpy.floor(lerp_y_data)

            for batch_c in range(np_c):
                top_left = img_flat[batch_idx, 0, batch_c, :]
                top_right = img_flat[batch_idx, 1, batch_c, :]
                bottom_left = img_flat[batch_idx, 2, batch_c, :]
                bottom_right = img_flat[batch_idx, 3, batch_c, :]

                # Plain interpolation at the chain precision.
                top_delta = top_right - top_left
                top = top_left + top_delta * lerp_x
                bottom_delta = bottom_right - bottom_left
                bottom = bottom_left + bottom_delta * lerp_x
                vertical_delta = bottom - top
                out = top + vertical_delta * lerp_y
                output[batch_idx, batch_c, :] = out

        # Round (RNE) to the current img dtype.
        output = output.reshape([np_batch, np_c, np_h, np_w]).astype(out_dtype)
        return output

    class ThirdPartyImpl:
        """Independent torch re-implementation used for cross-checking.

        Receives torch tensors (the caller converts the numpy inputs).
        Mirrors the operator's numerical contract: 16-bit float inputs
        are upcast to float32, the chain runs at the current input
        precision, and the result is rounded (round-to-nearest-even)
        back to the CURRENT img dtype.
        """

        def __init__(self, img, warp_index, **kwargs):
            _validate(img, warp_index)
            self.output_dtype = img.dtype
            self.img = img
            self.warp_index = warp_index

        def __call__(self, **kwargs):
            import torch

            name = str(self.img.dtype).split(".")[-1]
            work = torch.float32 if "float16" in name else self.img.dtype
            img_w = torch.as_tensor(self.img).to(work)
            warp_w = torch.as_tensor(self.warp_index)
            x_lerp, y_lerp = warp_w[:, 0:1], warp_w[:, 1:2]
            x_lerp, y_lerp = x_lerp - torch.floor(x_lerp), y_lerp - torch.floor(y_lerp)
            top_left, top_right = img_w[:, 0], img_w[:, 1]
            bottom_left, bottom_right = img_w[:, 2], img_w[:, 3]
            top = top_left + (top_right - top_left) * x_lerp
            bottom = bottom_left + (bottom_right - bottom_left) * x_lerp
            output = top + (bottom - top) * y_lerp
            return output.to(self.output_dtype)

    third_party = {"torch": ThirdPartyImpl}

    # Three-way cross-check tolerance (both supported float dtypes):
    # device output vs this numpy reference vs the independent torch
    # ThirdPartyImpl. The reference is a true-value model, so the device
    # output is expected to agree within the comparison thresholds
    # (device-side roundings are absorbed). "level" is an
    # ecosystem-level annotation; "threshold" belongs to a two-way
    # degraded caliber and is intentionally absent here (any degradation
    # is done by the caller, never in this file).
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }
