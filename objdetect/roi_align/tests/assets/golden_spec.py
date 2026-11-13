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

"""同一算子在四种测试路径下的 golden 编写。

Kernel/GEIR 的 golden 收到 numpy.ndarray，需手动转 torch 计算后转回 numpy；
ACLNN/E2E 的 golden 直接收到 torch.Tensor，无需转换。
字符串形式（golden = "torch.abs"）由框架自动处理转换，所有路径通用。
"""

__spec__ = {
    "roi_align": "RoiAlignKernelSpec",
    "aclnnRoiAlign": "AclnnRoiAlignSpec",
    "aclnnRoiAlignV2": "AclnnRoiAlignV2Spec",
}

import torch
import numpy as np
from torchvision.ops import roi_align as tv_roi_align

torch.set_printoptions(precision=6, sci_mode=False)


def _f32_scale(v):
    """Round spatial_scale to the operator's Float (fp32) attribute precision.

    The REG_OP attribute ``spatial_scale`` is Float (fp32), while the CSV value
    is a Python double.  Using the unrounded double in the reference makes ROI
    coordinates near the feature border differ by ~1e-7 from the operator,
    flipping out-of-bounds samples at the exact ROI edge.
    """
    return float(np.float32(v))


def _third_party_roi_align_torch(
    features,
    boxes,
    pooled_height,
    pooled_width,
    spatial_scale,
    sample_num,
    pool_mode,
    roi_end_mode,
):
    """Independent ROIAlign reference used only by the third_party providers.

    ``avg`` uses the third-party ``torchvision.ops.roi_align`` interface; ``max``
    (unsupported by torchvision) reuses the composed torch operators from the
    golden implementation.
    """
    features = torch.as_tensor(features)
    boxes = torch.as_tensor(boxes)
    spatial_scale = _f32_scale(spatial_scale)
    if isinstance(pool_mode, int):
        pool_mode = "avg" if pool_mode == 0 else "max"
    elif not isinstance(pool_mode, str):
        pool_mode = str(pool_mode)

    orig_dtype = features.dtype
    if orig_dtype in (torch.float16, torch.bfloat16):
        compute_dtype = torch.float32
    else:
        compute_dtype = orig_dtype

    features_c = features.to(compute_dtype)
    boxes_c = boxes.to(compute_dtype)
    n, c, height, width = features_c.shape
    k = boxes_c.shape[0]
    if features_c.numel() == 0 or boxes_c.numel() == 0:
        return [
            torch.zeros(
                (k, c, pooled_height, pooled_width),
                dtype=orig_dtype,
                device=features.device,
            )
        ]

    if pool_mode == "avg":
        rois = boxes_c
        if roi_end_mode == 1:
            rois = boxes_c.clone()
            rois[:, 1:5] = rois[:, 1:5] + 1.0
        output = tv_roi_align(
            features_c,
            rois,
            output_size=(pooled_height, pooled_width),
            spatial_scale=spatial_scale,
            sampling_ratio=sample_num,
            aligned=(roi_end_mode >= 2),
        )
    else:
        output = RoiAlignKernelSpec._roi_align_torch_compose(
            features_c,
            boxes_c,
            spatial_scale,
            pooled_height,
            pooled_width,
            sample_num,
            roi_end_mode,
            pool_mode,
        )
    return [output.to(orig_dtype)]


class RoiAlignKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray，third_party 收到 torch.Tensor"""

    @staticmethod
    def _bilinear_interpolate_torch(data, height, width, y, x):
        """
        双线性插值，使用 torch 张量小算子拼接实现。

        拼接的 torch 算子: torch.floor + torch.clamp + advanced indexing + tensor arithmetic

        Args:
            data: torch.Tensor [C, H, W]
            height: int, H 维度
            width: int, W 维度
            ys: torch.Tensor [num_points] y 坐标 (float32)
            xs: torch.Tensor [num_points] x 坐标 (float32)

        Returns:
            torch.Tensor [C, num_points] 插值结果
        """
        C = data.shape[0]

        # @constraint: 越界点返回 0（与算子 kernel 边界行为一致）
        # 非有限坐标（NaN/Inf）同样按越界处理，避免后续 long 转换溢出。
        out_of_bounds = (
            (y < -1.0)
            | (y > height)
            | (x < -1.0)
            | (x > width)
            | ~torch.isfinite(y)
            | ~torch.isfinite(x)
        )

        # 极端 spatial_scale 会把坐标放大到 1e38 量级，floor→long 会溢出成负索引；
        # 先夹到安全索引范围，越界点最后由 out_of_bounds 置 0。
        y = torch.nan_to_num(y, nan=0.0, posinf=float(max(height - 1, 0)), neginf=0.0)
        x = torch.nan_to_num(x, nan=0.0, posinf=float(max(width - 1, 0)), neginf=0.0)
        y = torch.clamp(y, min=0.0, max=float(max(height - 1, 0)))
        x = torch.clamp(x, min=0.0, max=float(max(width - 1, 0)))

        y_low = torch.floor(y).long()
        x_low = torch.floor(x).long()

        y_edge = y_low >= height - 1
        y_high = torch.where(y_edge, torch.full_like(y_low, height - 1), y_low + 1)
        y_low = torch.where(y_edge, torch.full_like(y_low, height - 1), y_low)
        y = torch.where(y_edge, y_low.to(data.dtype), y)

        x_edge = x_low >= width - 1
        x_high = torch.where(x_edge, torch.full_like(x_low, width - 1), x_low + 1)
        x_low = torch.where(x_edge, torch.full_like(x_low, width - 1), x_low)
        x = torch.where(x_edge, x_low.to(data.dtype), x)

        ly = (y - y_low.float()).unsqueeze(0)
        lx = (x - x_low.float()).unsqueeze(0)
        hy = 1.0 - ly
        hx = 1.0 - lx

        data_flat = data.reshape(C, -1)

        idx_ll = y_low * width + x_low
        idx_lh = y_low * width + x_high
        idx_hl = y_high * width + x_low
        idx_hh = y_high * width + x_high

        v1 = data_flat[:, idx_ll]
        v2 = data_flat[:, idx_lh]
        v3 = data_flat[:, idx_hl]
        v4 = data_flat[:, idx_hh]

        v11 = hy * hx
        v22 = hy * lx * v2
        v33 = ly * hx
        v44 = ly * lx
        r1 = torch.addcmul(v22, v11, v1)
        r2 = torch.addcmul(r1, v33, v3)
        result = torch.addcmul(r2, v44, v4)
        result[:, out_of_bounds] = 0.0
        return result

    @staticmethod
    def _roi_align_torch_compose(
        features_t,
        rois_t,
        spatial_scale,
        pooled_height,
        pooled_width,
        sample_num,
        roi_end_mode,
        pool_mode,
    ):
        """
        ROIAlign 使用 torch 小算子拼接实现。
        用于 torchvision 不支持的场景：pool_mode="max"。

        拼接的 torch 算子：
        - torch.arange + tensor arithmetic: 生成采样点坐标
        - torch.meshgrid: 构建采样网格
        - torch.floor + torch.clamp + advanced indexing: 双线性插值
        - torch.amax / torch.mean: 池化

        roi_end_mode 坐标映射：
          - 0: x' = x * spatial_scale（无偏移）
          - 1: x' = (x + 1) * spatial_scale（加1后缩放）
          - >=2: x' = x * spatial_scale - 0.5（减0.5偏移）
        """

        device = features_t.device
        spatial_scale = torch.tensor(spatial_scale, dtype=torch.float32).to(device)

        N, C, H, W = features_t.shape
        K = rois_t.shape[0]

        output = torch.zeros(
            (K, C, pooled_height, pooled_width), dtype=features_t.dtype
        ).to(device)

        # 单块元素上限，控制 C × block_h × grid_h × block_w × grid_w 的峰值内存。
        # 大 pooled 用例（pooled_h/pooled_w 达 1e4~1e5）靠二维分块避免物化整张网格。
        max_block_elems = 4_000_000
        one = torch.tensor(1.0, dtype=features_t.dtype).to(device)
        half = torch.tensor(0.5, dtype=features_t.dtype).to(device)

        with torch.no_grad():
            for n in range(K):
                batch_idx = int(rois_t[n, 0].item())
                x1 = torch.tensor(rois_t[n, 1].item(), dtype=features_t.dtype).to(
                    device
                )
                y1 = torch.tensor(rois_t[n, 2].item(), dtype=features_t.dtype).to(
                    device
                )
                x2 = torch.tensor(rois_t[n, 3].item(), dtype=features_t.dtype).to(
                    device
                )
                y2 = torch.tensor(rois_t[n, 4].item(), dtype=features_t.dtype).to(
                    device
                )

                if roi_end_mode == 0:
                    roi_start_w = x1 * spatial_scale
                    roi_start_h = y1 * spatial_scale
                elif roi_end_mode == 1:
                    roi_start_w = (x1 + one) * spatial_scale
                    roi_start_h = (y1 + one) * spatial_scale
                else:
                    roi_start_w = x1 * spatial_scale - half
                    roi_start_h = y1 * spatial_scale - half
                roi_width = torch.addcmul(-x1 * spatial_scale, x2, spatial_scale)
                roi_height = torch.addcmul(-y1 * spatial_scale, y2, spatial_scale)

                # @constraint: roi_end_mode 0/1 时强制 roi_width/height 最小为 1.0
                if roi_end_mode <= 1:
                    roi_width = torch.clamp(roi_width, min=one)
                    roi_height = torch.clamp(roi_height, min=one)

                bin_size_h = roi_height / torch.tensor(
                    pooled_height, dtype=features_t.dtype
                ).to(device)
                bin_size_w = roi_width / torch.tensor(
                    pooled_width, dtype=features_t.dtype
                ).to(device)

                if sample_num > 0:
                    roi_bin_grid_h = sample_num
                    roi_bin_grid_w = sample_num
                else:
                    roi_bin_grid_h = max(
                        1,
                        int(
                            torch.ceil(
                                roi_height
                                / torch.tensor(
                                    pooled_height, dtype=features_t.dtype
                                ).to(device)
                            ).item()
                        ),
                    )
                    roi_bin_grid_w = max(
                        1,
                        int(
                            torch.ceil(
                                roi_width
                                / torch.tensor(pooled_width, dtype=features_t.dtype).to(
                                    device
                                )
                            ).item()
                        ),
                    )

                iy = (
                    torch.arange(roi_bin_grid_h, dtype=features_t.dtype).to(device)
                    + half
                )
                ix = (
                    torch.arange(roi_bin_grid_w, dtype=features_t.dtype).to(device)
                    + half
                )
                off_h = (iy * bin_size_h) / roi_bin_grid_h
                off_w = (ix * bin_size_w) / roi_bin_grid_w

                feat = features_t[batch_idx]

                # 按 (pooled_h, pooled_w) 二维分块；块内中间量不超过 max_block_elems。
                bh, bw = pooled_height, pooled_width
                elem_per_bh_bw = max(int(C) * roi_bin_grid_h * roi_bin_grid_w, 1)
                while bh * bw * elem_per_bh_bw > max_block_elems and (bh > 1 or bw > 1):
                    if bh >= bw and bh > 1:
                        bh = (bh + 1) // 2
                    else:
                        bw = (bw + 1) // 2

                for ph0 in range(0, pooled_height, bh):
                    ph1 = min(ph0 + bh, pooled_height)
                    phv = torch.arange(ph0, ph1, dtype=features_t.dtype).to(device)
                    yy = roi_start_h + phv[:, None] * bin_size_h + off_h[None, :]
                    for pw0 in range(0, pooled_width, bw):
                        pw1 = min(pw0 + bw, pooled_width)
                        pwv = torch.arange(pw0, pw1, dtype=features_t.dtype).to(device)
                        xx = roi_start_w + pwv[:, None] * bin_size_w + off_w[None, :]

                        grid_y, grid_x = torch.meshgrid(
                            yy.reshape(-1), xx.reshape(-1), indexing="ij"
                        )
                        ys = grid_y.reshape(-1)
                        xs = grid_x.reshape(-1)

                        sampled = RoiAlignKernelSpec._bilinear_interpolate_torch(
                            feat, H, W, ys, xs
                        )
                        sampled = sampled.view(
                            C, ph1 - ph0, roi_bin_grid_h, pw1 - pw0, roi_bin_grid_w
                        )

                        if pool_mode == "max":
                            block = sampled.amax(dim=2).amax(dim=3)
                        else:
                            block = sampled.mean(dim=2).mean(dim=3)
                        output[n, :, ph0:ph1, pw0:pw1] = block

                        del sampled, block, grid_y, grid_x, ys, xs

        return output

    @staticmethod
    def _is_boundary_case(testcase_name):
        """Infer boundary/OOB intent from the case name.

        The CSV has no flag that reaches ``customize_inputs``; the case name is
        the available signal.  Existing boundary cases use ``bound`` /
        ``rangebound`` / ``upboundary`` in their names.
        """
        name = str(testcase_name or "").lower()
        return any(
            k in name for k in ("boundary", "rangebound", "upboundary", "bound_")
        )

    @staticmethod
    def _gen_rois(
        fm_shape,
        spatial_scale,
        rois,
        rois_n,
        *,
        roi_end_mode=1,
        pooled_height=1,
        pooled_width=1,
        boundary_mode=False,
    ):
        """Generate valid ROI coordinates within feature map bounds.

        Args:
            fm_shape: feature map shape (N, C, H, W)
            spatial_scale: float, scaling factor
            rois: numpy.ndarray [K, 5], modified in-place
            rois_n: numpy.ndarray [K] or None

        Returns:
            (rois, rois_n) — rois modified in-place, rois_n new array if not None
        """
        if rois.size == 0:
            t = torch.empty(0, 5)
            n = torch.empty(0)
            if rois_n is not None:
                if rois_n.size == 0:
                    return t.numpy(), n.numpy()
                else:
                    n_array = np.sort(
                        np.random.choice(
                            a=np.arange(rois_n.shape[0]).astype(rois.dtype, copy=False),
                            size=rois_n.shape[0],
                        )
                    ).astype(rois_n.dtype)
                    return t.numpy(), n_array
            else:
                return t.numpy(), None
        fm_h, fm_w = fm_shape[2], fm_shape[3]
        dtype = np.finfo(rois.dtype)
        dtype_max = float(dtype.max)
        scale = _f32_scale(spatial_scale)
        ph = max(int(pooled_height), 1)
        pw = max(int(pooled_width), 1)

        if boundary_mode:
            # Boundary/OOB coverage: keep the original full-range generation that
            # intentionally lets the ROI reach (or cross) the feature border.
            roi_w_max = min(fm_w / scale, dtype_max)
            roi_h_max = min(fm_h / scale, dtype_max)
        else:
            # Normal functional cases: leave at least one bin of margin and
            # account for the roi_end_mode offset, so no sample lands on x==W.
            margin_w = max(fm_w / pw, 1e-3)
            margin_h = max(fm_h / ph, 1e-3)
            end_w = fm_w - margin_w
            end_h = fm_h - margin_h
            if roi_end_mode == 0:
                roi_w_max = end_w / scale
                roi_h_max = end_h / scale
            elif roi_end_mode == 1:
                roi_w_max = end_w / scale - 1.0
                roi_h_max = end_h / scale - 1.0
            else:
                roi_w_max = (end_w + 0.5) / scale
                roi_h_max = (end_h + 0.5) / scale
            roi_w_max = max(min(roi_w_max, dtype_max), 1e-3)
            roi_h_max = max(min(roi_h_max, dtype_max), 1e-3)
        K = rois.shape[0]
        max_value = fm_shape[0]
        if fm_shape[0] == 0:
            max_value = K
        exact_int = 2 ** (dtype.nmant + 1)
        max_value = min(max_value, exact_int)
        rois[:, 0] = np.sort(
            np.random.choice(
                a=np.arange(max_value).astype(rois.dtype, copy=False), size=K
            )
        )
        # Same shape of distribution in both modes; only the cap differs:
        #   boundary_mode -> cap at W/scale (ROI reach/cross the border)
        #   normal        -> cap inside W by >=1 bin (no sample lands on W)
        x1 = np.random.uniform(0, roi_w_max, size=K)
        y1 = np.random.uniform(0, roi_h_max, size=K)
        x2 = np.minimum(
            x1 + np.random.uniform(1e-3, max(roi_w_max, 1e-3), size=K), roi_w_max
        )
        y2 = np.minimum(
            y1 + np.random.uniform(1e-3, max(roi_h_max, 1e-3), size=K), roi_h_max
        )
        rois[:, 1] = x1.astype(rois.dtype)
        rois[:, 2] = y1.astype(rois.dtype)
        rois[:, 3] = x2.astype(rois.dtype)
        rois[:, 4] = y2.astype(rois.dtype)
        if rois_n is not None:
            rois_n = rois[:, 0].astype(rois_n.dtype)
        return rois, rois_n

    @staticmethod
    def golden(
        features,
        rois,
        rois_n=None,
        *,
        spatial_scale,
        pooled_height,
        pooled_width,
        sample_num=2,
        roi_end_mode=1,
        pool_mode=0,
        **kwargs,
    ):
        """
        Golden function for roi_align.
        All the parameters (names and order) follow @roi_align_def.cpp without outputs.
        All the input Tensors are numpy.ndarray.

        Dispatch strategy:
        - pool_mode="avg": torchvision.ops.roi_align (torch 接口)
          - roi_end_mode=0:  aligned=False
          - roi_end_mode=1:  rois[:,1:5] += 1 (torch 小算子) → aligned=False
          - roi_end_mode>=2: aligned=True
        - pool_mode="max": torch 小算子拼接 (arange+meshgrid+floor+clamp+indexing+amax)

        roi_end_mode coordinate mapping:
          - 0: x' = x * spatial_scale (torchvision aligned=False)
          - 1: x' = (x+1) * spatial_scale (rois+1 → torchvision aligned=False)
          - >=2: x' = x * spatial_scale - 0.5 (torchvision aligned=True)

        pool_mode:
          - 0 or "avg": average pooling (torchvision supported → torch 接口)
          - 1 or "max": max pooling (torchvision unsupported → torch 小算子拼接)

        Args:
            features: numpy array [N, C, H, W], float16/float32 (REG_OP INPUT features)
            rois: numpy array [K, 5], float16/float32, columns [batch_index, x1, y1, x2, y2] (REG_OP INPUT rois)
            spatial_scale: float, scaling factor (REG_OP REQUIRED_ATTR spatial_scale)
            pooled_height: int, output H dimension (REG_OP REQUIRED_ATTR pooled_height)
            pooled_width: int, output W dimension (REG_OP REQUIRED_ATTR pooled_width)
            sample_num: int, sampling points per bin, 0 for adaptive (REG_OP ATTR sample_num, default 2)
            roi_end_mode: int, coordinate transform mode (REG_OP ATTR roi_end_mode, default 1)
            pool_mode: int or str, pooling mode 0/"avg" or 1/"max" (REG_OP ATTR pool_mode, default 0/"avg")
            rois_n: numpy array [K,] int32 or None, batch counts per ROI (REG_OP OPTIONAL_INPUT rois_n)
            **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                      full_soc_version, short_soc_version, testcase_name

        Returns:
            Output tensor [K, C, pooled_height, pooled_width]
        """
        if isinstance(pool_mode, int):
            pool_mode = "avg" if pool_mode == 0 else "max"
        elif not isinstance(pool_mode, str):
            pool_mode = str(pool_mode)
        spatial_scale = _f32_scale(spatial_scale)

        # Convert numpy inputs to torch tensors
        features_t = torch.as_tensor(features).contiguous()
        rois_t = torch.as_tensor(rois).contiguous()
        orig_dtype = features_t.dtype
        if features_t.numel() == 0 or rois_t.numel() == 0:
            k = rois_t.shape[0] if rois_t.dim() > 0 else 0
            c = features_t.shape[1] if features_t.dim() > 1 else 0
            out = torch.zeros((k, c, pooled_height, pooled_width), dtype=orig_dtype)
            return [out.numpy()]

        # @constraint: fp16/bf16 提升至 fp32 计算（与 kernel ReadAsFloat/WriteAsType 行为一致）
        if orig_dtype == torch.float16:
            features_comp = features_t.to(torch.float32)
            rois_comp = rois_t.to(torch.float32)
        else:
            if features_t.is_cpu:
                features_comp = features_t.to(torch.float64)
                rois_comp = rois_t.to(torch.float64)
            else:
                features_comp = features_t
                rois_comp = rois_t

        if pool_mode == "avg":
            # avg 分支：使用 torchvision.ops.roi_align 接口
            # roi_end_mode 坐标映射：
            #   0: aligned=False
            #   1: rois[:,1:5] += 1 → aligned=False
            #   >=2: aligned=True
            tv_aligned = roi_end_mode >= 2
            if roi_end_mode == 1:
                rois_comp = rois_comp.clone()
                rois_comp[:, 1:5] = rois_comp[:, 1:5] + 1.0

            output_t = tv_roi_align(
                features_comp,
                rois_comp,
                output_size=(pooled_height, pooled_width),
                spatial_scale=spatial_scale,
                sampling_ratio=sample_num,
                aligned=tv_aligned,
            )
        else:
            # max 分支：torch 小算子拼接（torchvision 不支持 max pooling）
            output_t = RoiAlignKernelSpec._roi_align_torch_compose(
                features_comp,
                rois_comp,
                spatial_scale,
                pooled_height,
                pooled_width,
                sample_num,
                roi_end_mode,
                pool_mode,
            )

        output_t = output_t.to(orig_dtype)
        return [output_t.cpu().numpy()]

    @staticmethod
    def customize_inputs(
        features,
        rois,
        rois_n=None,
        *,
        spatial_scale,
        pooled_height,
        pooled_width,
        sample_num=2,
        roi_end_mode=1,
        pool_mode=0,
        **kwargs,
    ):
        """customize_inputs for roi_align (Kernel/GEIR path).

        All the parameters (names and order) follow @roi_align_def.cpp without outputs.
        All the input Tensors are numpy.ndarray.

        Generates valid ROI coordinates within feature map bounds, ensuring
        batch indices and spatial coordinates are consistent with the feature map.

        Args:
            features: numpy.ndarray [N, C, H, W]
            rois: numpy.ndarray [K, 5]
            rois_n: numpy.ndarray [K] or None
            spatial_scale: float
            pooled_height: int
            pooled_width: int
            sample_num: int
            roi_end_mode: int
            pool_mode: int or str
            **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                      full_soc_version, short_soc_version, testcase_name

        Returns:
            tuple (features, rois, rois_n) — modified input arrays
        """
        feature_shape = features.shape
        boundary_mode = RoiAlignKernelSpec._is_boundary_case(
            kwargs.get("testcase_name")
        )
        if rois_n is not None:
            rois, rois_n = RoiAlignKernelSpec._gen_rois(
                feature_shape,
                spatial_scale,
                rois,
                rois_n,
                roi_end_mode=roi_end_mode,
                pooled_height=pooled_height,
                pooled_width=pooled_width,
                boundary_mode=boundary_mode,
            )
            return (features, rois, rois_n)
        else:
            rois, _ = RoiAlignKernelSpec._gen_rois(
                feature_shape,
                spatial_scale,
                rois,
                None,
                roi_end_mode=roi_end_mode,
                pooled_height=pooled_height,
                pooled_width=pooled_width,
                boundary_mode=boundary_mode,
            )
            return (features, rois, None)

    class KernelThirdPartyImpl:
        def __call__(
            self,
            features,
            rois,
            rois_n=None,
            *,
            spatial_scale,
            pooled_height,
            pooled_width,
            sample_num=2,
            roi_end_mode=1,
            pool_mode=0,
            **kwargs,
        ):
            return _third_party_roi_align_torch(
                features,
                rois,
                pooled_height,
                pooled_width,
                spatial_scale,
                sample_num,
                pool_mode,
                roi_end_mode,
            )

    third_party = {"torch": KernelThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }


class AclnnRoiAlignSpec:
    """ACLNN / E2E 流程 — golden 直接收到 torch.Tensor"""

    @staticmethod
    def golden(*args, **kwargs):
        """
        Aclnn golden for aclnnRoiAlign.
        All the parameters (name & order) follow \
            function `aclnnRoiAlignGetWorkspaceSize` in @aclnn_roi_align.cpp \
            without `workspaceSize` & `executor`.

        aclnn interface differs from REG_OP:
        - rois is [K, 4] (x1, y1, x2, y2 only, no batch_index column)
        - batchIndices is a separate [K] int32 tensor
        - batchIndices is cast to rois dtype, reshaped to [K, 1], concatenated with rois → [K, 5]
        - mode is a string ("avg" / "max")
        - roi_end_mode is not exposed, fixed to 0 (aclnn v1 launcher passes 0; the
          REG_OP/kernel default value is 1)
        - samplingRatio=0 means adaptive (same as sample_num=0 in REG_OP)

        Args:
            self: features tensor [N, C, H, W], float16/float32
            rois: rois tensor [K, 4], columns [x1, y1, x2, y2]
            batchIndices: batch index tensor [K], int32
            mode: pooling mode string, "avg" or "max"
            outputHeight: pooled output height
            outputWidth: pooled output width
            samplingRatio: sampling points per bin, 0 for adaptive
            spatialScale: spatial scaling factor
            out: output tensor [K, C, outputHeight, outputWidth] (for shape reference)
            kwargs: tensor_{dtypes, formats}, scalar_dtypes, short_soc_version, testcase_name

        Returns:
            Output tensor [K, C, outputHeight, outputWidth]
        """
        features = args[0]
        rois = args[1]
        batchIndices = args[2]
        mode = args[3]
        pooled_height = args[4]
        pooled_width = args[5]
        sample_num = args[6]
        spatial_scale = _f32_scale(args[7])
        if features.numel() == 0 or rois.numel() == 0:
            return [
                features.new_zeros(
                    (rois.shape[0], features.shape[1], pooled_height, pooled_width)
                )
            ]

        # @constraint: batchIndices cast to rois dtype, reshape [K,1], concat with rois → [K, 5]
        batchIndices_cast = batchIndices.to(rois.dtype).reshape(-1, 1)
        rois_concat = torch.cat(
            [batchIndices_cast, rois], dim=1
        )  # [K, 5] = [batch_idx, x1, y1, x2, y2]
        rois_t = rois_concat
        # @constraint: aclnn does not expose roi_end_mode, use 0
        roi_end_mode = 0

        # mode is already a string ("avg" / "max")
        pool_mode = mode
        if isinstance(pool_mode, int):
            pool_mode = "avg" if pool_mode == 0 else "max"
        elif not isinstance(pool_mode, str):
            pool_mode = str(pool_mode)

        orig_dtype = features.dtype
        features_t = features

        # @constraint: fp16/bf16 提升至 fp32 计算（与 kernel ReadAsFloat/WriteAsType 行为一致）
        if orig_dtype == torch.float16:
            features_comp = features_t.to(torch.float32)
            rois_comp = rois_t.to(torch.float32)
        else:
            if features_t.is_cpu:
                features_comp = features_t.to(torch.float64)
                rois_comp = rois_t.to(torch.float64)
            else:
                features_comp = features_t
                rois_comp = rois_t

        if pool_mode == "avg":
            # avg 分支：使用 torchvision.ops.roi_align 接口
            # aclnn 不暴露 roi_end_mode，固定为 0 → aligned=False
            sampling_ratio = sample_num if sample_num > 0 else -1
            output_t = tv_roi_align(
                features_comp,
                rois_comp,
                output_size=(pooled_height, pooled_width),
                spatial_scale=spatial_scale,
                sampling_ratio=sampling_ratio,
                aligned=False,
            )
        else:
            # max 分支：torch 小算子拼接
            output_t = RoiAlignKernelSpec._roi_align_torch_compose(
                features_comp,
                rois_comp,
                spatial_scale,
                pooled_height,
                pooled_width,
                sample_num,
                roi_end_mode,
                pool_mode,
            )

        output_t = output_t.to(orig_dtype)
        return [output_t]

    @staticmethod
    def customize_inputs(*args, **kwargs):
        """customize_inputs for aclnnRoiAlign (ACLNN/E2E path).

        All the parameters (names and order) follow aclnnRoiAlignGetWorkspaceSize
        without workspaceSize & executor.
        All the input Tensors are torch.Tensor.

        Modifies rois and batchIndices in-place to generate valid ROI coordinates
        within feature map bounds.

        Args:
            self: features tensor [N, C, H, W]
            rois: rois tensor [K, 4]
            batchIndices: batch index tensor [K]
            mode: pooling mode string
            outputHeight: int
            outputWidth: int
            samplingRatio: int
            spatialScale: float
            out: output tensor
            **kwargs: tensor_{dtypes,formats}, scalar_dtypes, short_soc_version, testcase_name
        """
        self = args[0]
        rois = args[1]
        batchIndices = args[2]
        spatialScale = args[7]
        rois_dtype = rois.dtype
        batch_indices_dtype = batchIndices.dtype

        feature_shape = tuple(self.shape)
        num_rois = rois.shape[0]
        temp_rois = np.zeros((num_rois, 5), dtype=rois.numpy().dtype)
        batch_indices_np = batchIndices.numpy()
        temp_rois, batch_indices_np = RoiAlignKernelSpec._gen_rois(
            feature_shape,
            spatialScale,
            temp_rois,
            batch_indices_np,
            roi_end_mode=0,
            pooled_height=args[4],
            pooled_width=args[5],
            boundary_mode=RoiAlignKernelSpec._is_boundary_case(
                kwargs.get("testcase_name")
            ),
        )
        rois_np = temp_rois[:, 1:5].copy()
        rois[:] = torch.from_numpy(rois_np).to(dtype=rois_dtype)
        batchIndices[:] = torch.from_numpy(batch_indices_np).to(
            dtype=batch_indices_dtype
        )

    class AclnnThirdPartyImpl:
        def __call__(self, *args, **kwargs):
            features = torch.as_tensor(args[0])
            rois = torch.as_tensor(args[1])
            batch_indices = torch.as_tensor(args[2]).to(rois.dtype).reshape(-1, 1)
            boxes = torch.cat([batch_indices, rois], dim=1)
            return _third_party_roi_align_torch(
                features,
                boxes,
                args[4],
                args[5],
                args[7],
                args[6],
                args[3],
                0,
            )

    third_party = {"torch": AclnnThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }


class AclnnRoiAlignV2Spec:
    """ACLNN V2 流程 — golden 直接收到 torch.Tensor"""

    @staticmethod
    def golden(*args, **kwargs):
        """
        Aclnn golden for aclnnRoiAlignV2.
        All the parameters (name & order) follow \
            function `aclnnRoiAlignV2GetWorkspaceSize` in @aclnn_roi_align_v2.cpp \
            without `workspaceSize` & `executor`.

        aclnnRoiAlignV2 differs from aclnnRoiAlign:
        - boxes is [K, 5] (already contains batch_index column)
        - aligned is a bool; it maps to roi_end_mode 2 (aligned) / 0 (not aligned)
        - pool_mode is fixed to "avg"
        - samplingRatio <= 0 means adaptive

        Args:
            self: features tensor [N, C, H, W], float16/float32
            boxes: rois tensor [K, 5], columns [batch_index, x1, y1, x2, y2]
            pooledHeight: pooled output height
            pooledWidth: pooled output width
            spatialScale: spatial scaling factor
            samplingRatio: sampling points per bin, <=0 for adaptive
            aligned: whether to use aligned coordinate transform
            out: output tensor [K, C, pooledHeight, pooledWidth] (for shape reference)
            **kwargs: tensor_{dtypes,formats}, scalar_dtypes, short_soc_version, testcase_name

        Returns:
            Output tensor [K, C, pooledHeight, pooledWidth]
        """
        features = args[0]
        boxes = args[1]
        pooled_height = args[2]
        pooled_width = args[3]
        spatial_scale = _f32_scale(args[4])
        sample_num = args[5]
        aligned = args[6]

        if features.numel() == 0 or boxes.numel() == 0:
            return [
                features.new_zeros(
                    (boxes.shape[0], features.shape[1], pooled_height, pooled_width)
                )
            ]

        # @constraint: boxes 与 features 保持同一 dtype（torchvision.ops.roi_align 要求一致）
        if boxes.dtype != features.dtype:
            boxes = boxes.to(features.dtype)

        # @constraint: aligned=True → roi_end_mode=2；否则 roi_end_mode=0（与 V2 API 一致）
        roi_end_mode = 2 if bool(aligned) else 0

        outputs = RoiAlignKernelSpec.golden(
            features,
            boxes,
            spatial_scale=spatial_scale,
            pooled_height=pooled_height,
            pooled_width=pooled_width,
            sample_num=sample_num,
            roi_end_mode=roi_end_mode,
            pool_mode="avg",
        )
        return [torch.as_tensor(output) for output in outputs]

    class AclnnThirdPartyImpl:
        def __call__(self, *args, **kwargs):
            return _third_party_roi_align_torch(
                torch.as_tensor(args[0]),
                torch.as_tensor(args[1]),
                args[2],
                args[3],
                args[4],
                args[5],
                "avg",
                2 if bool(args[6]) else 0,
            )

    third_party = {"torch": AclnnThirdPartyImpl}
    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }
