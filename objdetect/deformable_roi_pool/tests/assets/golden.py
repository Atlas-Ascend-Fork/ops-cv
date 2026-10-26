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

"""deformable_roi_pool 的 Torch CPU golden 与 MMCV GPU 三方标杆。"""

import torch


__spec__ = {
    "DeformableRoiPool": "DeformableRoiPoolKernelSpec",
    "deformable_roi_pool": "DeformableRoiPoolKernelSpec",
}


def _finite_or_zero(value):
    """使用 Torch 小算子将非有限标量替换为同 dtype/device 的零。"""
    return torch.where(torch.isfinite(value), value, torch.zeros_like(value))


def _adaptive_grid_size(bin_size):
    """将自适应采样网格转换为与 Kernel 一致的安全 int32 范围。"""
    if not bool(torch.isfinite(bin_size).item()):
        return 0
    value = float(bin_size.item())
    if value <= 0.0 or value > 46340.0:
        return 0
    return int(torch.ceil(bin_size).item())


def _deformable_roi_pool_cpu_compute(
    x, rois, offset, spatial_scale, output_size, sampling_ratio, gamma
):
    """仅用 Torch 基础算子复现原 NumPy golden 的采样语义。

    输入与采样坐标按 fp32 计算；bin 内采样点归约按 fp64 累加再转回 fp32，
    以避免大值合法 fp32 输入时 fp32 累加器溢出为 inf（数学均值仍可表示）。
    """
    output_dtype = x.dtype
    x_fp32 = x.to(device="cpu", dtype=torch.float32)
    rois_fp32 = rois.to(device="cpu", dtype=torch.float32)
    offset_fp32 = (
        None if offset is None else offset.to(device="cpu", dtype=torch.float32)
    )

    num_rois = int(rois_fp32.shape[0])
    batch_size, channels, height, width = map(int, x_fp32.shape)
    pooled_height = int(output_size[0])
    pooled_width = int(output_size[1])
    output_shape = (num_rois, channels, pooled_height, pooled_width)

    # 对齐原 golden：空 ROI 或 N=0 时直接返回与 x 同 dtype 的零 Tensor。
    if num_rois == 0 or batch_size == 0:
        return torch.zeros(output_shape, dtype=output_dtype, device="cpu")

    result = torch.zeros(output_shape, dtype=torch.float32, device="cpu")

    for roi_index in range(num_rois):
        raw_batch_index = rois_fp32[roi_index, 0]
        if bool(torch.isfinite(raw_batch_index).item()):
            batch_index = int(raw_batch_index.item())
        else:
            batch_index = 0
        batch_index = max(0, min(batch_index, batch_size - 1))

        # 对齐原 golden V3 防御：ROI 的四个坐标遇到 NaN/Inf 时置零。
        roi_x1 = _finite_or_zero(rois_fp32[roi_index, 1])
        roi_y1 = _finite_or_zero(rois_fp32[roi_index, 2])
        roi_x2 = _finite_or_zero(rois_fp32[roi_index, 3])
        roi_y2 = _finite_or_zero(rois_fp32[roi_index, 4])

        roi_start_w = roi_x1 * spatial_scale - 0.5
        roi_start_h = roi_y1 * spatial_scale - 0.5
        roi_end_w = roi_x2 * spatial_scale - 0.5
        roi_end_h = roi_y2 * spatial_scale - 0.5
        roi_width = roi_end_w - roi_start_w
        roi_height = roi_end_h - roi_start_h

        bin_size_h = roi_height / pooled_height
        bin_size_w = roi_width / pooled_width

        if sampling_ratio > 0:
            roi_bin_grid_h = int(sampling_ratio)
            roi_bin_grid_w = int(sampling_ratio)
        else:
            roi_bin_grid_h = _adaptive_grid_size(bin_size_h)
            roi_bin_grid_w = _adaptive_grid_size(bin_size_w)

        grid_h = (
            bin_size_h / roi_bin_grid_h
            if roi_bin_grid_h > 0
            else bin_size_h.new_zeros(())
        )
        grid_w = (
            bin_size_w / roi_bin_grid_w
            if roi_bin_grid_w > 0
            else bin_size_w.new_zeros(())
        )
        count = max(roi_bin_grid_h * roi_bin_grid_w, 1)
        feature = x_fp32[batch_index].to(dtype=torch.float64)

        for pooled_h in range(pooled_height):
            for pooled_w in range(pooled_width):
                bin_start_h = roi_start_h + pooled_h * bin_size_h
                bin_start_w = roi_start_w + pooled_w * bin_size_w

                if offset_fp32 is not None:
                    # 对齐原 golden V4 防御：offset 的 NaN/Inf 置零后再采样。
                    offset_w = _finite_or_zero(
                        offset_fp32[roi_index, 0, pooled_h, pooled_w]
                    )
                    offset_h = _finite_or_zero(
                        offset_fp32[roi_index, 1, pooled_h, pooled_w]
                    )
                    bin_start_w = bin_start_w + offset_w * gamma * roi_width
                    bin_start_h = bin_start_h + offset_h * gamma * roi_height

                value = torch.zeros(channels, dtype=torch.float64, device="cpu")
                for sample_h in range(roi_bin_grid_h):
                    raw_y = bin_start_h + (sample_h + 0.5) * grid_h
                    if (
                        not bool(torch.isfinite(raw_y).item())
                        or bool((raw_y < -1.0).item())
                        or bool((raw_y > height).item())
                    ):
                        continue

                    for sample_w in range(roi_bin_grid_w):
                        raw_x = bin_start_w + (sample_w + 0.5) * grid_w
                        if (
                            not bool(torch.isfinite(raw_x).item())
                            or bool((raw_x < -1.0).item())
                            or bool((raw_x > width).item())
                        ):
                            continue

                        y_clip = torch.clamp(raw_y, min=0.0)
                        x_clip = torch.clamp(raw_x, min=0.0)
                        y_low = min(int(torch.floor(y_clip).item()), height - 1)
                        x_low = min(int(torch.floor(x_clip).item()), width - 1)
                        y_high = min(y_low + 1, height - 1)
                        x_high = min(x_low + 1, width - 1)
                        y_clip = torch.clamp(y_clip, max=float(height - 1))
                        x_clip = torch.clamp(x_clip, max=float(width - 1))

                        ly = y_clip - y_low
                        lx = x_clip - x_low
                        hy = 1.0 - ly
                        hx = 1.0 - lx
                        weight1 = hy * hx
                        weight2 = hy * lx
                        weight3 = ly * hx
                        weight4 = ly * lx

                        value += (
                            weight1 * feature[:, y_low, x_low]
                            + weight2 * feature[:, y_low, x_high]
                            + weight3 * feature[:, y_high, x_low]
                            + weight4 * feature[:, y_high, x_high]
                        )

                result[roi_index, :, pooled_h, pooled_w] = (value / count).to(
                    dtype=torch.float32
                )

    return result.to(dtype=output_dtype)


class DeformableRoiPoolKernelSpec:
    """deformable_roi_pool 的 Kernel/GEIR TestSpec。"""

    @staticmethod
    def golden(
        x,
        rois,
        offset=None,
        *,
        spatial_scale=1.0,
        output_size=None,
        sampling_ratio=0,
        gamma=0.1,
        **kwargs,
    ):
        """CPU 真值：numpy 边界转换，计算过程仅使用 Torch 基础小算子。"""
        del kwargs
        if output_size is None:
            raise ValueError("output_size is required for deformable_roi_pool")

        x_tensor = torch.from_numpy(x)
        rois_tensor = torch.from_numpy(rois)
        offset_tensor = None if offset is None else torch.from_numpy(offset)
        result = _deformable_roi_pool_cpu_compute(
            x_tensor,
            rois_tensor,
            offset_tensor,
            spatial_scale,
            output_size,
            sampling_ratio,
            gamma,
        )
        return [result.numpy()]

    class ThirdPartyImpl:
        """GPU 竞品标杆：使用 MMCV 的同名 deform_roi_pool 实现。"""

        def __init__(self, **kwargs):
            if kwargs.get("output_size") is None:
                raise ValueError("output_size is required for deformable_roi_pool")

            x = kwargs["x"]
            rois = kwargs["rois"]
            offset = kwargs.get("offset")
            requested_device = kwargs.get("device")

            if torch.is_tensor(x):
                x_tensor = x
                device = x.device
            else:
                if requested_device is None:
                    requested_device = "cuda" if torch.cuda.is_available() else "cpu"
                device = torch.device(requested_device)
                x_tensor = torch.as_tensor(x, device=device)

            # 以 x 所在设备为准，Torch 输入保持原 device；numpy 输入可由 device 指定。
            self.device = x_tensor.device
            self.output_dtype = x_tensor.dtype
            self.x = x_tensor.to(dtype=torch.float32)
            self.rois = torch.as_tensor(rois, device=self.device).to(torch.float32)
            self.offset = (
                None
                if offset is None
                else torch.as_tensor(offset, device=self.device).to(torch.float32)
            )
            self.output_size = (
                int(kwargs["output_size"][0]),
                int(kwargs["output_size"][1]),
            )
            self.spatial_scale = float(kwargs.get("spatial_scale", 1.0))
            self.sampling_ratio = int(kwargs.get("sampling_ratio", 0))
            self.gamma = float(kwargs.get("gamma", 0.1))

        def __call__(self, x=None, rois=None, offset=None, **kwargs):
            """直接执行 MMCV 竞品；设备不受支持时明确报错且不回退。"""
            del x, rois, offset, kwargs
            if self.device.type == "cpu":
                raise RuntimeError(
                    "MMCV deform_roi_pool third-party golden requires a supported "
                    "accelerator device; CPU fallback is intentionally disabled"
                )

            # 对齐原 golden 的空 ROI/N 防御，同时保持输出 dtype/device。
            if self.rois.shape[0] == 0 or self.x.shape[0] == 0:
                return [
                    torch.zeros(
                        (
                            int(self.rois.shape[0]),
                            int(self.x.shape[1]),
                            self.output_size[0],
                            self.output_size[1],
                        ),
                        dtype=self.output_dtype,
                        device=self.device,
                    )
                ]

            # 延迟导入，避免 CPU golden 进程被 MMCV GPU 扩展的装载条件影响。
            from mmcv.ops import deform_roi_pool as mmcv_deform_roi_pool

            result = mmcv_deform_roi_pool(
                self.x,
                self.rois,
                self.offset,
                self.output_size,
                self.spatial_scale,
                self.sampling_ratio,
                self.gamma,
            )
            return [result.to(dtype=self.output_dtype)]

    third_party = {"mmcv": ThirdPartyImpl}

    # 比较标准说明：cross_check 需要第三方 MMCV（CUDA）参考并要求可用的 XPU/GPU endpoint。
    # 目标验证环境为 Ascend NPU、无 mmcv 且无 XPU provider，cross_check 只能得到 GOLDEN_FAILURE，
    # 无法建立任何数值结论。因此声明 TTK 官方标准 mix_tolerance（仅需本交付 CPU golden）完成数值比较；
    # third_party 继续交付并在具备 GPU/XPU 环境时可切回 cross_check。
    tolerance = {
        "float16": {"standard": "mix_tolerance"},
        "float32": {"standard": "mix_tolerance"},
    }
