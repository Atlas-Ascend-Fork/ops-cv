#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

"""Rasterizer Golden 与 Torch 三方小算子拼接实现。

仅注册两条交付通路：
- Kernel：``rasterizer``
- ACLNN：``aclnnRasterizer``

PyTorch 没有与 CANN Rasterizer 完全同语义的单一原生接口，因此 third_party
按 CANN 公式使用 Torch 基础算子拼接。Golden 与 third_party 共用纯 Torch
计算核心，但两者的输入准备严格分离：

- Golden：输入在计算开始前一次性准备；FP16/BF16 浮点输入统一升到 FP32。
- ThirdPartyImpl：不升精度、不迁移设备、不转 NumPy，计时范围内只执行算子链。

CANN 语义：
1. x_s=(x/w*0.5+0.5)*(width-1)+0.5
   y_s=(0.5+0.5*y/w)*(height-1)+0.5
   z_s=z/w*0.49999+0.5
2. findices 为 1-based 面索引，0 表示背景。
3. 深度乘 262144 后向 0 截断，严格小于当前深度时更新。
4. barycentric 为透视矫正后的重心坐标，背景为 (0, 0, 0)。
5. d/dOptional、occlusionTruncation、useDepthPrior 当前均不参与计算。
"""

import numpy as np
import torch


# 兼容旧 Golden 插件加载路径。
__golden__ = {
    "kernel": {"rasterizer": "rasterizer_golden"},
    "aclnn": {"aclnnRasterizer": "aclnn_rasterizer_golden"},
}


# Rasterizer 正式输出为 INT32 + FLOAT32。FP16/BF16 条目用于防御性支持：
# 如果后续扩展输入/输出 dtype，Golden 仍按“先统一升 FP32 再计算”的规则执行。
_RASTERIZER_TOLERANCE = {
    "float16": {"standard": "cross_check", "level": "L0"},
    "bfloat16": {"standard": "cross_check", "level": "L0"},
    "float32": {"standard": "cross_check", "level": "L0"},
    "int32": {"standard": "binary_equal"},
}

_PROMOTE_TO_FP32 = (torch.float16, torch.bfloat16)
_MAGN = 2 << 17


def _as_cpu_golden_tensor(value):
    """把 Golden 输入转换为 CPU Torch Tensor，不执行算子计算。"""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()

    array = np.asarray(value)
    try:
        return torch.from_numpy(array)
    except TypeError:
        # NumPy 扩展 BF16 dtype 可能无法直接被 torch.from_numpy 接收。
        # 这里属于计算前输入准备，并直接得到 FP32 Tensor。
        if array.dtype.name in ("bfloat16", "bf16"):
            return torch.from_numpy(array.astype(np.float32))
        raise


def _prepare_golden_inputs(v, f, d_optional=None):
    """一次性完成 Golden 的全部输入准备和 FP16/BF16 升精度。

    所有 dtype/device 转换都在调用 ``_rasterizer_torch_compute`` 前完成；
    计算核心内部不会把 FP16/BF16 零散升到 FP32。
    """
    raw_tensors = (
        _as_cpu_golden_tensor(v),
        _as_cpu_golden_tensor(f),
        _as_cpu_golden_tensor(d_optional),
    )
    prepared_tensors = tuple(
        tensor.float()
        if tensor is not None and tensor.dtype in _PROMOTE_TO_FP32
        else tensor
        for tensor in raw_tensors
    )
    return prepared_tensors


def _tri_area(ax, ay, bx, by, cx, cy):
    """三角形有向面积的 2 倍，方向为 A -> B -> C。"""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _screen_transform(v, width, height):
    """把齐次顶点坐标投影到屏幕坐标。"""
    x, y, z, w = torch.unbind(v, dim=1)
    xs = (x / w * 0.5 + 0.5) * float(width - 1) + 0.5
    ys = (0.5 + 0.5 * y / w) * float(height - 1) + 0.5
    zs = z / w * 0.49999 + 0.5
    return xs, ys, zs


def _persp_x(v, width):
    """第二阶段使用的屏幕 x 坐标。"""
    return v[:, 0] / v[:, 3] * (0.5 * float(width - 1)) + 0.5 * float(width)


def _persp_y(v, height):
    """第二阶段使用的屏幕 y 坐标。"""
    return v[:, 1] / v[:, 3] * (0.5 * float(height - 1)) + 0.5 * float(height)


def _triangle_bbox(xs, ys, i0, i1, i2, width, height):
    """计算并裁剪三角形包围盒。标量提取只用于确定切片边界。"""
    xmin_t = torch.clamp(
        torch.minimum(torch.minimum(xs[i0], xs[i1]), xs[i2]), 0.0, float(width - 1)
    )
    xmax_t = torch.clamp(
        torch.maximum(torch.maximum(xs[i0], xs[i1]), xs[i2]), 0.0, float(width - 1)
    )
    ymin_t = torch.clamp(
        torch.minimum(torch.minimum(ys[i0], ys[i1]), ys[i2]), 0.0, float(height - 1)
    )
    ymax_t = torch.clamp(
        torch.maximum(torch.maximum(ys[i0], ys[i1]), ys[i2]), 0.0, float(height - 1)
    )
    return int(xmin_t), int(xmax_t), int(ymin_t), int(ymax_t)


def _rasterizer_torch_compute(v, f, width, height):
    """纯 Torch Rasterizer 计算核心，返回设备侧 Tensor。

    约束：调用者必须提前准备好 dtype/device。本函数不执行 FP16/BF16
    升精度，不调用 ``.to()``、``.cpu()`` 或 ``.numpy()``。
    """
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got {width}x{height}")
    if not v.dtype.is_floating_point:
        raise TypeError(f"v must be a floating-point tensor, got {v.dtype}")
    if v.device != f.device:
        raise ValueError(
            f"v and f must be on the same device, got {v.device} and {f.device}"
        )

    device = v.device
    calc_dtype = v.dtype
    num_faces = f.shape[0]

    xs, ys, zs = _screen_transform(v, width, height)

    # Kernel 的 depth workspace 为浮点 T；Cast(CAST_TRUNC) 后仍存储整数值浮点数。
    best_depth_q = torch.full(
        (height, width),
        torch.finfo(calc_dtype).max,
        dtype=calc_dtype,
        device=device,
    )
    best_face = torch.zeros((height, width), dtype=torch.int32, device=device)

    for face_no in range(num_faces):
        i0, i1, i2 = f[face_no, 0], f[face_no, 1], f[face_no, 2]
        x0, y0, z0 = xs[i0], ys[i0], zs[i0]
        x1, y1, z1 = xs[i1], ys[i1], zs[i1]
        x2, y2, z2 = xs[i2], ys[i2], zs[i2]

        area = _tri_area(x0, y0, x1, y1, x2, y2)
        if bool(area == 0.0):
            continue

        xmin, xmax, ymin, ymax = _triangle_bbox(xs, ys, i0, i1, i2, width, height)
        if xmin > xmax or ymin > ymax:
            continue

        px = torch.arange(xmin, xmax + 1, dtype=calc_dtype, device=device) + 0.5
        py = torch.arange(ymin, ymax + 1, dtype=calc_dtype, device=device) + 0.5
        pixel_x, pixel_y = torch.meshgrid(px, py, indexing="xy")

        beta = _tri_area(x0, y0, pixel_x, pixel_y, x2, y2) / area
        gamma = _tri_area(x0, y0, x1, y1, pixel_x, pixel_y) / area
        alpha = 1.0 - beta - gamma

        depth = alpha * z0 + beta * z1 + gamma * z2
        depth_q = torch.trunc(depth * _MAGN)

        region_depth = best_depth_q[ymin : ymax + 1, xmin : xmax + 1]
        region_face = best_face[ymin : ymax + 1, xmin : xmax + 1]
        inside = (
            (alpha >= 0.0)
            & (alpha <= 1.0)
            & (beta >= 0.0)
            & (beta <= 1.0)
            & (gamma >= 0.0)
            & (gamma <= 1.0)
        )
        update = inside & (depth_q < region_depth)
        region_depth.copy_(torch.where(update, depth_q, region_depth))
        region_face.copy_(
            torch.where(update, torch.full_like(region_face, face_no + 1), region_face)
        )

    barycentric = torch.zeros((height, width, 3), dtype=calc_dtype, device=device)
    valid = best_face > 0
    if bool(torch.any(valid)):
        face_indices = best_face[valid] - 1
        face_vertices = f[face_indices]
        flat_vertex_indices = face_vertices.reshape(-1)
        gathered_vertices = v[flat_vertex_indices]

        screen_vertices = torch.stack(
            [_persp_x(gathered_vertices, width), _persp_y(gathered_vertices, height)],
            dim=-1,
        ).reshape(face_indices.shape[0], 3, 2)
        vertex_w = gathered_vertices[:, 3].reshape(face_indices.shape[0], 3)

        pixel_x = (torch.arange(width, dtype=calc_dtype, device=device) + 0.5).expand(
            height, width
        )[valid]
        pixel_y = (
            (torch.arange(height, dtype=calc_dtype, device=device) + 0.5)
            .unsqueeze(1)
            .expand(height, width)[valid]
        )
        pixels = torch.stack([pixel_x, pixel_y], dim=-1)

        p0 = screen_vertices[:, 0]
        p1 = screen_vertices[:, 1]
        p2 = screen_vertices[:, 2]
        area = _tri_area(p0[:, 0], p0[:, 1], p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1])
        beta = (
            _tri_area(
                p0[:, 0], p0[:, 1], pixels[:, 0], pixels[:, 1], p2[:, 0], p2[:, 1]
            )
            / area
        )
        gamma = (
            _tri_area(
                p0[:, 0], p0[:, 1], p1[:, 0], p1[:, 1], pixels[:, 0], pixels[:, 1]
            )
            / area
        )
        alpha = 1.0 - beta - gamma

        corrected_alpha = alpha / vertex_w[:, 0]
        corrected_beta = beta / vertex_w[:, 1]
        corrected_gamma = gamma / vertex_w[:, 2]
        corrected_sum = corrected_alpha + corrected_beta + corrected_gamma
        safe_sum = torch.where(
            corrected_sum == 0.0, torch.ones_like(corrected_sum), corrected_sum
        )
        corrected = torch.stack(
            [
                corrected_alpha / safe_sum,
                corrected_beta / safe_sum,
                corrected_gamma / safe_sum,
            ],
            dim=-1,
        )
        corrected = torch.where(
            (torch.abs(area) > 0.0).unsqueeze(-1),
            corrected,
            torch.full_like(corrected, -1.0),
        )
        barycentric[valid] = corrected

    return [best_face, barycentric]


def rasterizer_golden(
    v,
    f,
    d=None,
    width=None,
    height=None,
    occlusion_truncation=0.0,
    use_depth_prior=0,
    **kwargs,
):
    """Kernel Golden；输入/输出遵循 NumPy 路径契约。"""
    del occlusion_truncation, use_depth_prior, kwargs
    if width is None or height is None:
        raise ValueError("rasterizer golden requires width and height")

    # 包含未生效的 d：若它是 FP16/BF16，也必须在计算前完成统一升精度。
    v_tensor, f_tensor, _d_tensor = _prepare_golden_inputs(v, f, d)
    outputs = _rasterizer_torch_compute(v_tensor, f_tensor, width, height)
    return [output.detach().cpu().numpy() for output in outputs]


def aclnn_rasterizer_golden(
    v,
    f,
    dOptional=None,
    width=None,
    height=None,
    occlusionTruncation=0.0,
    useDepthPrior=0,
    findicesOut=None,
    barycentricOut=None,
    **kwargs,
):
    """ACLNN Golden；参数顺序与 aclnnRasterizerGetWorkspaceSize 一致。"""
    del occlusionTruncation, useDepthPrior, findicesOut, barycentricOut, kwargs
    if width is None or height is None:
        raise ValueError("aclnnRasterizer golden requires width and height")

    v_tensor, f_tensor, _d_tensor = _prepare_golden_inputs(v, f, dOptional)
    return _rasterizer_torch_compute(v_tensor, f_tensor, width, height)


class ThirdPartyImpl:
    """Torch 小算子拼接竞品实现。

    ``__init__`` 在性能计时外保存属性；``__call__`` 不升精度、不做设备迁移、
    不转 NumPy，直接返回当前竞品设备上的 Torch Tensor。
    """

    def __init__(self, width=None, height=None, **kwargs):
        del kwargs
        if width is None or height is None:
            raise ValueError("ThirdPartyImpl requires width and height")
        self.width = int(width)
        self.height = int(height)

    def __call__(self, v, f, *args, **kwargs):
        del args, kwargs
        return _rasterizer_torch_compute(v, f, self.width, self.height)


class RasterizerKernelTestSpec:
    """Kernel 通路测试规范。"""

    golden = staticmethod(rasterizer_golden)
    third_party = {"torch": ThirdPartyImpl}
    tolerance = _RASTERIZER_TOLERANCE


class RasterizerAclnnTestSpec:
    """ACLNN 通路测试规范。"""

    golden = staticmethod(aclnn_rasterizer_golden)
    third_party = {"torch": ThirdPartyImpl}
    tolerance = _RASTERIZER_TOLERANCE


__spec__ = {
    "rasterizer": "RasterizerKernelTestSpec",
    "aclnnRasterizer": "RasterizerAclnnTestSpec",
}


if __name__ == "__main__":
    # 本地轻量冒烟；正式三方测试由 TTK 在竞品设备上执行。
    vertices = np.array(
        [
            [0.0, 0.0, 10.0, 1.0],
            [2.0, 0.0, 10.0, 1.0],
            [0.0, 2.0, 10.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)

    kernel_outputs = rasterizer_golden(
        vertices,
        faces,
        width=4,
        height=4,
        occlusion_truncation=0.0,
        use_depth_prior=0,
    )
    assert kernel_outputs[0].shape == (4, 4)
    assert kernel_outputs[0].dtype == np.int32
    assert kernel_outputs[1].shape == (4, 4, 3)
    assert kernel_outputs[1].dtype == np.float32

    promoted_v, promoted_f, _ = _prepare_golden_inputs(
        torch.from_numpy(vertices).half(), torch.from_numpy(faces)
    )
    assert promoted_v.dtype == torch.float32
    assert promoted_f.dtype == torch.int32

    third_party_outputs = ThirdPartyImpl(width=4, height=4)(
        torch.from_numpy(vertices), torch.from_numpy(faces)
    )
    assert torch.equal(third_party_outputs[0], torch.from_numpy(kernel_outputs[0]))
    assert torch.allclose(third_party_outputs[1], torch.from_numpy(kernel_outputs[1]))
    print("Rasterizer Golden/ThirdParty smoke tests passed")
