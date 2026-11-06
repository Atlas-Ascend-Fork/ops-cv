#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""rotated_feature_align_grad 的 Torch CPU golden 与 MMCV GPU 三方标杆。"""

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

__spec__ = {
    "rotated_feature_align_grad": "RotatedFeatureAlignGradKernelSpec",
}

__golden__ = {
    "kernel": {
        "rotated_feature_align_grad": "golden_rotated_feature_align_grad",
    },
}


def _mmcv_sample_points(bboxes, spatial_scale, points):
    """MMCV rotated_feature_align 采样点：bboxes (N,H,W,5)=(y,x,w,h,a)，
    y/x/w/h 乘 spatial_scale、角度不乘；points=5 时按 MMCV 角点顺序补 4 个旋转角点。"""
    roi_y = bboxes[..., 0] * spatial_scale
    roi_x = bboxes[..., 1] * spatial_scale
    px_list = [roi_x]
    py_list = [roi_y]
    if points > 1:
        roi_w = bboxes[..., 2] * spatial_scale
        roi_h = bboxes[..., 3] * spatial_scale
        roi_a = bboxes[..., 4]
        w_2 = roi_w * 0.5
        h_2 = roi_h * 0.5
        cosa = torch.cos(roi_a)
        sina = torch.sin(roi_a)
        wx = cosa * w_2
        wy = sina * w_2
        hx = -sina * h_2
        hy = cosa * h_2
        px_list = px_list + [
            roi_x + wx + hx,
            roi_x - wx + hx,
            roi_x - wx - hx,
            roi_x + wx - hx,
        ]
        py_list = py_list + [
            roi_y + wy + hy,
            roi_y - wy + hy,
            roi_y - wy - hy,
            roi_y + wy - hy,
        ]
    return torch.stack(py_list, dim=-1), torch.stack(px_list, dim=-1)


def _mmcv_bilinear_terms(bboxes, H, W, spatial_scale, points):
    """MMCV bilinear_interpolate_gradient 语义（common_cuda_helper.hpp）：
    越界（fy<-1 或 fy>H 或 fx<-1 或 fx>W）权重全零；fy/fx 下限钳 0 后取整；
    y_low>=H-1 时 y_low=y_high=H-1（x 同理）。非有限坐标（NaN/±Inf，来自
    inf/nan bboxes 或 cos/sin 溢出）同样按越界处理——与 kernel 侧
    ComputeBilinearWeight 的非有限拦截保持一致。返回每个采样点的四角坐标与权重。"""
    py, px = _mmcv_sample_points(bboxes, spatial_scale, points)
    oob = (py < -1.0) | (py > float(H)) | (px < -1.0) | (px > float(W))
    oob = oob | ~torch.isfinite(py) | ~torch.isfinite(px)
    yc = torch.clamp(py, min=0.0)
    xc = torch.clamp(px, min=0.0)
    y_low = yc.floor().long()
    x_low = xc.floor().long()
    y_edge = y_low >= (H - 1)
    x_edge = x_low >= (W - 1)
    y_low_c = torch.where(y_edge, torch.full_like(y_low, H - 1), y_low)
    x_low_c = torch.where(x_edge, torch.full_like(x_low, W - 1), x_low)
    y_high = torch.where(y_edge, y_low_c, y_low_c + 1)
    x_high = torch.where(x_edge, x_low_c, x_low_c + 1)
    ly = torch.where(y_edge, yc - y_low_c.to(yc.dtype), yc - y_low.to(yc.dtype))
    lx = torch.where(x_edge, xc - x_low_c.to(xc.dtype), xc - x_low.to(xc.dtype))
    hy = 1.0 - ly
    hx = 1.0 - lx
    w1 = hy * hx
    w2 = hy * lx
    w3 = ly * hx
    w4 = ly * lx
    zero = torch.zeros_like(w1)
    w1 = torch.where(oob, zero, w1)
    w2 = torch.where(oob, zero, w2)
    w3 = torch.where(oob, zero, w3)
    w4 = torch.where(oob, zero, w4)
    return y_low_c, x_low_c, y_high, x_high, w1, w2, w3, w4


def _rotated_feature_align_grad_cpu_compute(dy, bboxes, spatial_scale, points):
    """仅用 Torch 基础算子向量化复现 MMCV backward 语义。

    dx[n,y,x,c] = dy[n,y,x,c] + sum_(h,w,p) dy[n,h,w,c] * w_p((n,h,w) -> (y,x))，
    其中 w_p 为采样点四角命中 (y,x) 的权重之和。采样点与四角权重仅依赖
    (n,h,w,p)，与输出位置无关，故预计算一次后按目标像素 scatter 累加，避免
    逐输出位置重算的 O(N·(H·W)²) 复杂度。累加全程 float32 单层进行。
    """
    output_dtype = dy.dtype
    dy_fp32 = dy.to(device="cpu", dtype=torch.float32).contiguous()
    bboxes_fp32 = bboxes.to(device="cpu", dtype=torch.float32).contiguous()

    N, H, W, C = map(int, dy_fp32.shape)
    # 对齐 kernel：空输入直接返回零。
    if dy_fp32.numel() == 0 or bboxes_fp32.numel() == 0:
        return torch.zeros((N, H, W, C), dtype=output_dtype, device="cpu")

    y_low, x_low, y_high, x_high, w1, w2, w3, w4 = _mmcv_bilinear_terms(
        bboxes_fp32, H, W, spatial_scale, points
    )

    # 自梯度：dx 初值为 dy；再 scatter 累加各采样点的四角贡献。
    result = dy_fp32.clone()
    hw = H * W
    src_hw = torch.arange(hw, dtype=torch.long).repeat_interleave(points)
    corners = (
        (y_low, x_low, w1),
        (y_low, x_high, w2),
        (y_high, x_low, w3),
        (y_high, x_high, w4),
    )
    for n in range(N):
        dy_n = dy_fp32[n].reshape(hw, C)
        out_n = result[n].reshape(hw, C)
        for cy_t, cx_t, w_t in corners:
            cy = cy_t[n].reshape(-1)
            cx = cx_t[n].reshape(-1)
            cw = w_t[n].reshape(-1)
            valid = (cw != 0.0) & (cy >= 0) & (cy < H) & (cx >= 0) & (cx < W)
            if not bool(valid.any()):
                continue
            dst = cy[valid] * W + cx[valid]
            contrib = dy_n[src_hw[valid]] * cw[valid].unsqueeze(1)
            out_n.index_add_(0, dst, contrib)
    return result.to(dtype=output_dtype)


class RotatedFeatureAlignGradKernelSpec:
    """rotated_feature_align_grad 的 Kernel/GEIR TestSpec。"""

    @staticmethod
    def golden(dy, bboxes, *, spatial_scale, points=1, **kwargs):
        """CPU 真值：numpy 输入转换，计算过程仅使用 Torch 基础小算子。"""
        del kwargs
        dy_tensor = torch.from_numpy(dy)
        bboxes_tensor = torch.from_numpy(bboxes)
        result = _rotated_feature_align_grad_cpu_compute(
            dy_tensor, bboxes_tensor, spatial_scale, points
        )
        return [result.numpy()]

    class ThirdPartyImpl:
        """GPU 竞品标杆：使用 MMCV 的 rotated_feature_align 反向传播。"""

        def __init__(self, *, spatial_scale, points=1, **kwargs):
            dy = kwargs.get("dy")
            bboxes = kwargs.get("bboxes")
            requested_device = kwargs.get("device")

            if torch.is_tensor(dy):
                dy_tensor = dy
                device = dy_tensor.device
            else:
                if requested_device is None:
                    requested_device = "cuda" if torch.cuda.is_available() else "cpu"
                device = torch.device(requested_device)
                dy_tensor = None if dy is None else torch.as_tensor(dy, device=device)

            # 以 dy 所在设备为准，Torch 输入保持原 device；numpy 输入可由 device 指定。
            # 输入类型/精度转换统一前置到构造阶段：dy/bboxes 固化为 device 上的 float32。
            self.device = device
            self.output_dtype = (
                dy_tensor.dtype if dy_tensor is not None else torch.float32
            )
            self.dy = (
                None
                if dy_tensor is None
                else dy_tensor.to(device=self.device, dtype=torch.float32)
            )
            self.bboxes = (
                None
                if bboxes is None
                else torch.as_tensor(bboxes).to(device=self.device, dtype=torch.float32)
            )
            self.spatial_scale = float(spatial_scale)
            self.points = int(points)

        def __call__(self, dy=None, bboxes=None, **kwargs):
            """直接执行 MMCV 竞品的反向传播；设备不受支持时明确报错且不回退。"""
            del dy, bboxes, kwargs

            if self.device.type == "cpu":
                raise RuntimeError(
                    "MMCV rotated_feature_align third-party golden requires a supported "
                    "accelerator device; CPU fallback is intentionally disabled"
                )

            N, H, W, C = map(int, self.dy.shape)
            # 对齐 kernel：空输入直接返回零。
            if self.dy.numel() == 0 or self.bboxes.numel() == 0:
                return [
                    torch.zeros(
                        (N, H, W, C), dtype=self.output_dtype, device=self.device
                    )
                ]

            # MMCV 使用 NCHW 布局；bboxes (N,H,W,5)=(y,x,w,h,a) 布局一致。
            # 前向对特征是线性的，零特征足以取得反向梯度。
            features = torch.zeros(
                (N, C, H, W),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True,
            )
            output = mmcv_rotated_feature_align_call(
                features, self.bboxes, self.spatial_scale, self.points
            )
            output.backward(self.dy.permute(0, 3, 1, 2).contiguous())
            grad = features.grad.permute(0, 2, 3, 1).contiguous()
            return [grad.to(dtype=self.output_dtype)]

    third_party = {"torch": ThirdPartyImpl}

    # 交付范围内只有 float32，且验收环境无 GPU（第三方 MMCV 显式禁用 CPU 回退），
    # cross_check 无法建立数值结论。声明 TTK 官方标准 mix_tolerance，仅依赖本交付
    # CPU golden 即可完成浮点数值比较；third_party 继续交付，具备可用 XPU 时可切回 cross_check。
    tolerance = {
        "float32": {"standard": "mix_tolerance"},
    }


def mmcv_rotated_feature_align_call(features, bboxes, spatial_scale, points):
    """延迟导入 MMCV，避免 CPU golden 进程被 MMCV GPU 扩展的装载条件影响。"""
    from mmcv.ops import rotated_feature_align as mmcv_rotated_feature_align

    return mmcv_rotated_feature_align(features, bboxes, spatial_scale, points)


def golden_rotated_feature_align_grad(dy, bboxes, *, spatial_scale, points=1, **kwargs):
    """TTK __golden__ 兼容入口 — receives numpy, returns list of numpy。"""
    return RotatedFeatureAlignGradKernelSpec.golden(
        dy, bboxes, spatial_scale=spatial_scale, points=points, **kwargs
    )
