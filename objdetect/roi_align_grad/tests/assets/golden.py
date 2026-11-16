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

"""roi_align_grad 在 kernel/geir/aclnn 测试路径下的 golden 编写。

Kernel/GEIR 的 golden 收到 numpy.ndarray，统一转换为 torch tensor 后，用 torch 小算子
（arange + index_add_ + clamp + where）向量化实现四种对齐模式的反向，采用 float32
坐标链以对齐 kernel；不依赖 torchvision。
ACLNN/third_party 的 golden 直接收到 torch.Tensor，复用同一实现。

四种对齐模式（与 device kernel 一致）：
  mode 0 (aligned=False): 无偏移；roi_w/roi_h = max(., 1.0)
  mode 1 (TF): x2/y2 += spatial_scale；roi_w/roi_h = max(., 1.0)
  mode 2 (PyTorch aligned=True): x1/y1/x2/y2 -= 0.5；不 clamp roi_w/roi_h
  mode 3 (MMDetection v0.6 aligned=True): 同 mode 2 的 -0.5 偏移；不 clamp roi_w/roi_h
aclnn 用 bool aligned：aligned=True -> mode 2，aligned=False -> mode 0。
"""

import math

import torch

# TTK compatibility layer: __golden__ dict expected by ttk's scanner
__golden__ = {
    "kernel": {
        "roi_align_grad": "_kernel_golden_wrapper",
        "ROIAlignGrad": "_kernel_golden_wrapper",
    },
    "aclnn": {"aclnnRoiAlignV2Backward": "_aclnn_golden_wrapper"},
}


def _kernel_golden_wrapper(
    ydiff,
    rois,
    rois_n=None,
    *,
    xdiff_shape,
    pooled_width,
    pooled_height,
    spatial_scale,
    sample_num=2,
    roi_end_mode=1,
    **kwargs,
):
    return RoiAlignGradKernelSpec.golden(
        ydiff,
        rois,
        rois_n,
        xdiff_shape=xdiff_shape,
        pooled_width=pooled_width,
        pooled_height=pooled_height,
        spatial_scale=spatial_scale,
        sample_num=sample_num,
        roi_end_mode=roi_end_mode,
        **kwargs,
    )


def _aclnn_golden_wrapper(
    gradOutput,
    boxes,
    inputShape,
    pooledHeight,
    pooledWidth,
    spatialScale,
    samplingRatio,
    aligned,
    gradInput,
    **kwargs,
):
    result = AclnnRoiAlignV2BackwardSpec.golden(
        gradOutput,
        boxes,
        inputShape,
        pooledHeight,
        pooledWidth,
        spatialScale,
        samplingRatio,
        aligned,
        gradInput,
        **kwargs,
    )
    if isinstance(result, list) and len(result) == 1:
        return result[0]
    return result


__spec__ = {
    "roi_align_grad": "RoiAlignGradKernelSpec",
    "ROIAlignGrad": "RoiAlignGradKernelSpec",
    "aclnnRoiAlignV2Backward": "AclnnRoiAlignV2BackwardSpec",
}


def _f32(value):
    return float(torch.tensor(value, dtype=torch.float32))


def _validate(
    ydiff,
    rois,
    xdiff_shape,
    pooled_width,
    pooled_height,
    spatial_scale,
    sample_num,
    roi_end_mode,
):
    if ydiff.ndim != 4:
        raise ValueError(f"ydiff must be 4D (N, C, pH, pW), got {ydiff.shape}")
    if rois.ndim != 2 or rois.shape[1] < 5:
        raise ValueError(f"rois must be 2D with last dim >= 5, got {rois.shape}")
    if len(xdiff_shape) != 4:
        raise ValueError(f"xdiff_shape must be 4D (B, C, H, W), got {xdiff_shape}")
    if pooled_width <= 0 or pooled_height <= 0:
        raise ValueError(
            f"pooled_width/height must be positive, got {pooled_width}/{pooled_height}"
        )
    if spatial_scale <= 0:
        raise ValueError(f"spatial_scale must be positive, got {spatial_scale}")
    if sample_num < 0:
        raise ValueError(f"sample_num must be non-negative, got {sample_num}")
    if roi_end_mode not in (0, 1, 2, 3):
        raise ValueError(f"roi_end_mode only supports 0/1/2/3, got {roi_end_mode}")


def _clamp_rois(rois_t, B, H, W, spatial_scale):
    rois_t = rois_t.clone()
    upper_x = (W - 0.5) / spatial_scale
    upper_y = (H - 0.5) / spatial_scale

    rois_t[:, 0] = rois_t[:, 0].clamp(0, B - 1)
    rois_t[:, 1] = rois_t[:, 1].clamp(0, upper_x)
    rois_t[:, 2] = rois_t[:, 2].clamp(0, upper_y)
    rois_t[:, 3] = rois_t[:, 3].clamp(min=rois_t[:, 1])
    rois_t[:, 3] = rois_t[:, 3].clamp(max=upper_x)
    rois_t[:, 4] = rois_t[:, 4].clamp(min=rois_t[:, 2])
    rois_t[:, 4] = rois_t[:, 4].clamp(max=upper_y)
    return rois_t


def _roi_align_grad_torchvision(
    ydiff_t,
    rois_t,
    xdiff_shape,
    pooled_width,
    pooled_height,
    spatial_scale,
    sample_num,
    roi_end_mode,
):
    from torchvision.ops import RoIAlign

    B, C, H, W = (int(v) for v in xdiff_shape)
    device = ydiff_t.device
    work_dtype = ydiff_t.dtype
    xdiff = torch.zeros((B, C, H, W), dtype=work_dtype, device=device)

    N = rois_t.shape[0]
    if N == 0 or ydiff_t.numel() == 0:
        return xdiff

    aligned = roi_end_mode == 2
    sampling_ratio = int(sample_num) if sample_num > 0 else -1

    x = torch.zeros((B, C, H, W), dtype=work_dtype, device=device, requires_grad=True)
    y = RoIAlign(
        output_size=(int(pooled_height), int(pooled_width)),
        spatial_scale=float(spatial_scale),
        sampling_ratio=sampling_ratio,
        aligned=aligned,
    )(x, rois_t)

    if y.shape != ydiff_t.shape:
        return xdiff

    y.backward(ydiff_t)
    xdiff_t = x.grad
    if xdiff_t is None:
        return xdiff
    return xdiff_t


def _roi_align_grad_torch_small(
    ydiff_t,
    rois_t,
    xdiff_shape,
    pooled_width,
    pooled_height,
    spatial_scale,
    sample_num,
    roi_end_mode,
):
    B, C, H, W = (int(v) for v in xdiff_shape)
    device = ydiff_t.device
    work_dtype = ydiff_t.dtype
    xdiff = torch.zeros((B, C, H, W), dtype=work_dtype, device=device)

    N = rois_t.shape[0]
    if N == 0 or ydiff_t.numel() == 0:
        return xdiff

    pooled_width = int(pooled_width)
    pooled_height = int(pooled_height)
    spatial_scale = float(spatial_scale)
    sample_num = int(sample_num)

    for n in range(N):
        roi_row = rois_t[n]
        if not bool(torch.isfinite(roi_row).all()):
            raise ValueError(
                f"roi_align_grad: non-finite ROI at index {n}; undefined by the operator contract"
            )
        batch_idx = int(rois_t[n, 0].item())
        if batch_idx < 0 or batch_idx >= B:
            continue

        # Match the kernel: the whole coordinate/grid chain is float32.
        scale = _f32(spatial_scale)
        x1 = _f32(_f32(rois_t[n, 1].item()) * scale)
        y1 = _f32(_f32(rois_t[n, 2].item()) * scale)
        x2 = _f32(_f32(rois_t[n, 3].item()) * scale)
        y2 = _f32(_f32(rois_t[n, 4].item()) * scale)
        if not (
            math.isfinite(x1)
            and math.isfinite(y1)
            and math.isfinite(x2)
            and math.isfinite(y2)
        ):
            raise ValueError(
                f"roi_align_grad: ROI coordinates overflow to non-finite at index {n}"
            )

        if roi_end_mode == 1:
            x2 = _f32(x2 + scale)
            y2 = _f32(y2 + scale)
        elif roi_end_mode in (2, 3):
            x1 = _f32(x1 - 0.5)
            y1 = _f32(y1 - 0.5)
            x2 = _f32(x2 - 0.5)
            y2 = _f32(y2 - 0.5)

        roi_w = _f32(x2 - x1)
        roi_h = _f32(y2 - y1)
        if roi_end_mode in (0, 1):
            roi_w = _f32(max(roi_w, 1.0))
            roi_h = _f32(max(roi_h, 1.0))

        bin_w = _f32(roi_w / _f32(pooled_width))
        bin_h = _f32(roi_h / _f32(pooled_height))

        if sample_num > 0:
            gH = gW = sample_num
        else:
            gH = max(
                int(
                    math.ceil(
                        float(torch.tensor(roi_h, dtype=torch.float32) / pooled_height)
                    )
                ),
                1,
            )
            gW = max(
                int(
                    math.ceil(
                        float(torch.tensor(roi_w, dtype=torch.float32) / pooled_width)
                    )
                ),
                1,
            )

        count = max(gH * gW, 1)

        iy = torch.arange(gH, dtype=work_dtype, device=device).view(gH, 1, 1, 1)
        ix = torch.arange(gW, dtype=work_dtype, device=device).view(1, gW, 1, 1)
        ph = torch.arange(pooled_height, dtype=work_dtype, device=device).view(
            1, 1, pooled_height, 1
        )
        pw = torch.arange(pooled_width, dtype=work_dtype, device=device).view(
            1, 1, 1, pooled_width
        )
        inv_grid_h = _f32(1.0 / gH)
        inv_grid_w = _f32(1.0 / gW)
        y = y1 + ph * bin_h + (iy + 0.5) * bin_h * inv_grid_h
        x = x1 + pw * bin_w + (ix + 0.5) * bin_w * inv_grid_w
        y = y.expand(gH, gW, pooled_height, pooled_width)
        x = x.expand(gH, gW, pooled_height, pooled_width)

        valid = (y >= -1.0) & (y <= H) & (x >= -1.0) & (x <= W)

        y = torch.clamp(y, min=0.0)
        x = torch.clamp(x, min=0.0)
        y_low = y.floor().long()
        x_low = x.floor().long()
        y_high = y_low + 1
        x_high = x_low + 1

        yl_clamp = y_low >= H - 1
        y_low = torch.where(yl_clamp, torch.full_like(y_low, H - 1), y_low)
        y_high = torch.where(yl_clamp, torch.full_like(y_high, H - 1), y_high)
        xl_clamp = x_low >= W - 1
        x_low = torch.where(xl_clamp, torch.full_like(x_low, W - 1), x_low)
        x_high = torch.where(xl_clamp, torch.full_like(x_high, W - 1), x_high)

        # Guard index_add_ against a residual out-of-range index.
        y_low = y_low.clamp(0, H - 1)
        y_high = y_high.clamp(0, H - 1)
        x_low = x_low.clamp(0, W - 1)
        x_high = x_high.clamp(0, W - 1)

        ly = y - y_low.to(work_dtype)
        lx = x - x_low.to(work_dtype)
        hy = 1.0 - ly
        hx = 1.0 - lx
        w1 = hy * hx
        w2 = hy * lx
        w3 = ly * hx
        w4 = ly * lx

        K = gH * gW * pooled_height * pooled_width
        w1f = w1.reshape(-1)
        w2f = w2.reshape(-1)
        w3f = w3.reshape(-1)
        w4f = w4.reshape(-1)
        valid_f = valid.to(work_dtype).reshape(-1)

        gv = ydiff_t[n].contiguous().view(C, 1, 1, pooled_height, pooled_width)
        gv = gv.expand(C, gH, gW, pooled_height, pooled_width).contiguous()
        gv = gv.reshape(C, K)
        inv_count = _f32(1.0 / count)
        g = (gv * inv_count) * valid_f.view(1, K)

        c1 = g * w1f.view(1, K)
        c2 = g * w2f.view(1, K)
        c3 = g * w3f.view(1, K)
        c4 = g * w4f.view(1, K)

        idx_yl = y_low.reshape(-1)
        idx_xl = x_low.reshape(-1)
        idx_yh = y_high.reshape(-1)
        idx_xh = x_high.reshape(-1)
        flat_ll = (idx_yl * W + idx_xl).long()
        flat_lh = (idx_yl * W + idx_xh).long()
        flat_hl = (idx_yh * W + idx_xl).long()
        flat_hh = (idx_yh * W + idx_xh).long()

        xdiff_b = xdiff[batch_idx].view(C, H * W)
        xdiff_b.index_add_(1, flat_ll, c1)
        xdiff_b.index_add_(1, flat_lh, c2)
        xdiff_b.index_add_(1, flat_hl, c3)
        xdiff_b.index_add_(1, flat_hh, c4)

    return xdiff


def _roi_align_grad_dispatch(
    ydiff_t,
    rois_t,
    xdiff_shape,
    pooled_width,
    pooled_height,
    spatial_scale,
    sample_num,
    roi_end_mode,
):
    # Skip what the kernel skips: an out-of-range batch index or a non-finite
    # ROI. torchvision.ops.RoIAlign segfaults on an out-of-range batch index,
    # so keep such ROIs out of both paths and filter ydiff in lockstep.
    if rois_t.dim() == 2 and rois_t.shape[0] == ydiff_t.shape[0]:
        B = int(xdiff_shape[0])
        # The kernel truncates the float batch index toward zero, so use the
        # same integer when deciding whether to skip the ROI.
        bidx = rois_t[:, 0].to(torch.int64)
        keep = torch.isfinite(rois_t).all(dim=1) & (bidx >= 0) & (bidx < B)
        if not bool(keep.all()):
            rois_t = rois_t[keep]
            ydiff_t = ydiff_t[keep]
    return _roi_align_grad_torch_small(
        ydiff_t,
        rois_t,
        xdiff_shape,
        pooled_width,
        pooled_height,
        spatial_scale,
        sample_num,
        roi_end_mode,
    )


class RoiAlignGradKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray。"""

    def golden(
        ydiff,
        rois,
        rois_n=None,
        *,
        xdiff_shape,
        pooled_width,
        pooled_height,
        spatial_scale,
        sample_num=2,
        roi_end_mode=1,
        **kwargs,
    ):
        """
        Golden function for roi_align_grad.
        All the parameters (names and order) follow @roi_align_grad_def.cpp without outputs.
        All the input Tensors are numpy.ndarray.

        Args:
            ydiff: numpy array, shape (N, C, pooled_H, pooled_W), backward gradient input
            rois: numpy array, shape (N, 5), ROI coordinates [batch_idx, x1, y1, x2, y2]
            rois_n: numpy array (optional), shape (B,), int32, number of ROIs per batch (unused)
            xdiff_shape: tuple/list (B, C, H, W), output shape
            pooled_width: int, pooled output width
            pooled_height: int, pooled output height
            spatial_scale: float, spatial scale factor
            sample_num: int, sampling number, 0 means auto-compute
            roi_end_mode: int, alignment mode (0/1/2/3)
            **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                      full_soc_version, short_soc_version, testcase_name

        Returns:
            [xdiff]: list of numpy.ndarray, shape (B, C, H, W)
        """
        import numpy as np

        ydiff_np = np.asarray(ydiff)
        rois_np = np.asarray(rois)
        _validate(
            ydiff_np,
            rois_np,
            xdiff_shape,
            pooled_width,
            pooled_height,
            spatial_scale,
            sample_num,
            roi_end_mode,
        )

        ydiff_t = torch.from_numpy(ydiff_np)
        rois_t = torch.from_numpy(rois_np)
        xdiff_t = _roi_align_grad_dispatch(
            ydiff_t,
            rois_t,
            xdiff_shape,
            pooled_width,
            pooled_height,
            spatial_scale,
            sample_num,
            roi_end_mode,
        )
        return [xdiff_t.cpu().numpy()]

    class ThirdPartyImpl:
        """torch 竞品实现 — 收到 torch.Tensor，复用核心分发逻辑。"""

        def __init__(
            self,
            ydiff,
            rois,
            rois_n=None,
            *,
            xdiff_shape,
            pooled_width,
            pooled_height,
            spatial_scale,
            sample_num=2,
            roi_end_mode=1,
            **kwargs,
        ):
            self.xdiff_shape = tuple(int(v) for v in xdiff_shape)
            self.pooled_width = int(pooled_width)
            self.pooled_height = int(pooled_height)
            self.spatial_scale = float(spatial_scale)
            self.sample_num = int(sample_num)
            self.roi_end_mode = int(roi_end_mode)

            self.ydiff_t = ydiff
            self.rois_t = rois
            self.orig_dtype = ydiff.dtype

        def __call__(self, **kwargs):
            xdiff_t = _roi_align_grad_dispatch(
                self.ydiff_t,
                self.rois_t,
                self.xdiff_shape,
                self.pooled_width,
                self.pooled_height,
                self.spatial_scale,
                self.sample_num,
                self.roi_end_mode,
            )
            return [xdiff_t.to(self.orig_dtype)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
    }


class AclnnRoiAlignV2BackwardSpec:
    """ACLNN 流程 — golden 收到 torch.Tensor（已在设备上）。"""

    def golden(
        gradOutput,
        boxes,
        inputShape,
        pooledHeight,
        pooledWidth,
        spatialScale,
        samplingRatio,
        aligned,
        gradInput,
        **kwargs,
    ):
        """
        Aclnn golden for aclnnRoiAlignV2Backward.
        All the parameters (name & order) follow \
            function `aclnnRoiAlignV2BackwardGetWorkspaceSize` in \
            @aclnn_roi_align_v2_backward.h \
            without `workspaceSize` & `executor`.
        When all dtypes are natively supported by torch, \
            the Tensors in the parameters are all torch.Tensor. \
            Conversely, when not, the Tensors in the parameters are all numpy.ndarray.

        Args:
            kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

        Returns:
            [gradInput]: list of torch.Tensor, shape (B, C, H, W)
        """
        roi_end_mode = 2 if aligned else 0

        xdiff_shape = (
            tuple(inputShape) if isinstance(inputShape, (list, tuple)) else inputShape
        )

        gradOutput_t = (
            gradOutput
            if isinstance(gradOutput, torch.Tensor)
            else torch.from_numpy(gradOutput)
        )
        boxes_t = boxes if isinstance(boxes, torch.Tensor) else torch.from_numpy(boxes)

        gradOutput_t = torch.where(
            torch.isfinite(gradOutput_t), gradOutput_t, torch.zeros_like(gradOutput_t)
        )

        xdiff_t = _roi_align_grad_dispatch(
            gradOutput_t,
            boxes_t,
            xdiff_shape,
            int(pooledWidth),
            int(pooledHeight),
            float(spatialScale),
            int(samplingRatio),
            roi_end_mode,
        )

        return [xdiff_t]

    class ThirdPartyImpl:
        """torch 竞品实现 — 形参名与 aclnn camelCase 接口对齐。

        独立于 Kernel Spec 的 ThirdPartyImpl：后者形参为 snake_case
        （ydiff/rois/xdiff_shape/...），与 aclnn 参数名（gradOutput/boxes/
        inputShape/...）不匹配，直接复用会导致三方按名绑定失败，
        cross_check 因三方无参考而判 GOLDEN_FAILURE。
        形参名与顺序跟随 aclnnRoiAlignV2BackwardGetWorkspaceSize。
        """

        def __init__(
            self,
            gradOutput,
            boxes,
            inputShape,
            pooledHeight,
            pooledWidth,
            spatialScale,
            samplingRatio,
            aligned,
            gradInput=None,
            **kwargs,
        ):
            self.xdiff_shape = tuple(int(v) for v in inputShape)
            self.pooled_width = int(pooledWidth)
            self.pooled_height = int(pooledHeight)
            self.spatial_scale = float(spatialScale)
            self.sample_num = int(samplingRatio)
            self.aligned = bool(aligned)

            self.grad_output_t = (
                gradOutput
                if isinstance(gradOutput, torch.Tensor)
                else torch.from_numpy(gradOutput)
            )
            self.boxes_t = (
                boxes if isinstance(boxes, torch.Tensor) else torch.from_numpy(boxes)
            )
            self.orig_dtype = self.grad_output_t.dtype

        def __call__(self, **kwargs):
            roi_end_mode = 2 if self.aligned else 0
            grad_output_t = torch.where(
                torch.isfinite(self.grad_output_t),
                self.grad_output_t,
                torch.zeros_like(self.grad_output_t),
            )
            xdiff_t = _roi_align_grad_dispatch(
                grad_output_t,
                self.boxes_t,
                self.xdiff_shape,
                self.pooled_width,
                self.pooled_height,
                self.spatial_scale,
                self.sample_num,
                roi_end_mode,
            )
            return [xdiff_t.to(self.orig_dtype)]

    third_party = {"torch": ThirdPartyImpl}
    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
    }
