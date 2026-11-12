#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

"""aclnnRasterizer 精度测试执行器。

布局参考 aclnnGridSampler2D / aclnnRoiAlign：

- ``aclnn_rasterizer``：CPU 标杆执行器。``init_by_input_data`` 依据用例 id
  固定随机种子重建语义合法的几何数据（随机网格、规则网格、分层遮挡、
  屏幕内外混合、退化面混合、等深双面 6 类场景轮转）；``__call__`` 使用
  Torch 小算子拼接计算 golden。由于 ATK 双后端共享同一 InputDataset，
  aclnn 侧拿到的输入与 CPU 标杆完全一致。
- ``RasterizerAclnnApi``：pyaclnn 侧执行器，仅指定输入 format 为 ND。
"""

import random

import numpy as np
import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.tasks.dataset.base_dataset import OpsDataset


_MAGN = 2 << 17


def _tri_area(ax, ay, bx, by, cx, cy):
    """三角形有向面积的 2 倍，方向为 A -> B -> C。"""
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _rasterizer_golden_compute(v, f, width, height):
    """纯 Torch Rasterizer golden 计算，与 tests/assets/golden.py 语义一致。"""
    width = int(width)
    height = int(height)

    num_faces = f.shape[0]
    x, y, z, w = torch.unbind(v, dim=1)
    xs = (x / w * 0.5 + 0.5) * float(width - 1) + 0.5
    ys = (0.5 + 0.5 * y / w) * float(height - 1) + 0.5
    zs = z / w * 0.49999 + 0.5

    best_depth_q = torch.full(
        (height, width), torch.finfo(torch.float32).max, dtype=torch.float32
    )
    best_face = torch.zeros((height, width), dtype=torch.int32)

    for face_no in range(num_faces):
        i0, i1, i2 = f[face_no, 0], f[face_no, 1], f[face_no, 2]
        x0, y0, z0 = xs[i0], ys[i0], zs[i0]
        x1, y1, z1 = xs[i1], ys[i1], zs[i1]
        x2, y2, z2 = xs[i2], ys[i2], zs[i2]

        area = _tri_area(x0, y0, x1, y1, x2, y2)
        if bool(area == 0.0):
            continue

        xmin = int(
            torch.clamp(
                torch.minimum(torch.minimum(xs[i0], xs[i1]), xs[i2]),
                0.0,
                float(width - 1),
            )
        )
        xmax = int(
            torch.clamp(
                torch.maximum(torch.maximum(xs[i0], xs[i1]), xs[i2]),
                0.0,
                float(width - 1),
            )
        )
        ymin = int(
            torch.clamp(
                torch.minimum(torch.minimum(ys[i0], ys[i1]), ys[i2]),
                0.0,
                float(height - 1),
            )
        )
        ymax = int(
            torch.clamp(
                torch.maximum(torch.maximum(ys[i0], ys[i1]), ys[i2]),
                0.0,
                float(height - 1),
            )
        )
        if xmin > xmax or ymin > ymax:
            continue

        px = torch.arange(xmin, xmax + 1, dtype=torch.float32) + 0.5
        py = torch.arange(ymin, ymax + 1, dtype=torch.float32) + 0.5
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

    barycentric = torch.zeros((height, width, 3), dtype=torch.float32)
    valid = best_face > 0
    if bool(torch.any(valid)):
        face_indices = best_face[valid] - 1
        face_vertices = f[face_indices]
        flat_vertex_indices = face_vertices.reshape(-1)
        gathered_vertices = v[flat_vertex_indices]

        screen_x = gathered_vertices[:, 0] / gathered_vertices[:, 3] * (
            0.5 * float(width - 1)
        ) + 0.5 * float(width)
        screen_y = gathered_vertices[:, 1] / gathered_vertices[:, 3] * (
            0.5 * float(height - 1)
        ) + 0.5 * float(height)
        screen_vertices = torch.stack([screen_x, screen_y], dim=-1).reshape(
            face_indices.shape[0], 3, 2
        )
        vertex_w = gathered_vertices[:, 3].reshape(face_indices.shape[0], 3)

        pixel_x = (torch.arange(width, dtype=torch.float32) + 0.5).expand(
            height, width
        )[valid]
        pixel_y = (
            (torch.arange(height, dtype=torch.float32) + 0.5)
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


class _CaseSeededRng(random.Random):
    """以用例 id 为种子的独立 RNG，隔离全局 random 状态。"""


# 每面唯一深度表（40 档 > 当前用例最大有效面数 33）。深度选取同时满足：
# - z ∈ (0.2, 0.8) 且 |z - 0.5| >= 0.02（避开中值与端点）；
# - z*262144 与 zs*262144（zs = z*0.49999 + 0.5）的小数部分均落在
#   [0.2, 0.8]，截断量化远离整数边界，两端 1ulp 级舍入差不会翻转；
# - 相邻档距 >= 0.0145：深度胜者由面深度唯一决定，无量化碰撞、无等深
#   tie-break 歧义，跨平台确定性由构造保证。
_DEPTH_VALUES = (
    0.2010021209716797,
    0.2155017852783203,
    0.23000526428222656,
    0.2445087432861328,
    0.25901222229003906,
    0.2735157012939453,
    0.28801918029785156,
    0.3025226593017578,
    0.31702613830566406,
    0.3315296173095703,
    0.34603309631347656,
    0.3605365753173828,
    0.37503623962402344,
    0.38953590393066406,
    0.4040355682373047,
    0.41853904724121094,
    0.4330425262451172,
    0.44754600524902344,
    0.4620494842529297,
    0.47655296325683594,
    0.5200557708740234,
    0.5345592498779297,
    0.5490627288818359,
    0.5635623931884766,
    0.5780620574951172,
    0.5925617218017578,
    0.6070652008056641,
    0.6215686798095703,
    0.6360721588134766,
    0.6505756378173828,
    0.6650791168212891,
    0.6795825958251953,
    0.6940860748291016,
    0.7085895538330078,
    0.7230930328369141,
    0.7375965118408203,
    0.7520961761474609,
    0.7665958404541016,
    0.7810955047607422,
    0.7955989837646484,
)


def _face_depth(face_no):
    """面 face_no 的唯一深度；有效面数不超过表长时无重复。"""
    return _DEPTH_VALUES[face_no % len(_DEPTH_VALUES)]


def _screen_xy(rng, scenario, idx):
    """按场景生成第 idx 个屏幕内顶点的 (x, y)。

    - scenario 0/2：均匀随机点（带中心区偏置增强重叠）。
    - scenario 1：jitter 网格布点。
    - scenario 3：3/4 在屏内、1/4 在屏外。
    - scenario 4：偶数顶点与其后顶点共线（退化面构造）。
    - scenario 5：大小两个嵌套三角形的顶点。
    """
    if scenario == 1:
        cols = max(2, int(np.sqrt(idx + 4)))
        row, col = divmod(idx, cols)
        return (
            -0.9 + 1.8 * (col + 0.5) / cols + rng.uniform(-0.02, 0.02),
            -0.9 + 1.8 * (row + 0.5) / max(2, cols) + rng.uniform(-0.02, 0.02),
        )
    if scenario == 3 and idx % 4 == 3:
        return (rng.uniform(1.2, 2.0), rng.uniform(-1.5, 1.5))
    if scenario == 4 and idx % 2 == 1:
        # 与前一个顶点共线制造退化（零面积）面。
        return (rng.uniform(-0.8, 0.4), rng.uniform(-0.8, 0.8))
    if scenario == 5:
        big = [(-0.8, -0.8), (0.8, -0.8), (0.8, 0.8)]
        small = [(-0.4, -0.4), (0.4, -0.4), (0.4, 0.4)]
        px, py = (big + small)[idx % 6]
        return (px + rng.uniform(-0.01, 0.01), py + rng.uniform(-0.01, 0.01))
    return (rng.uniform(-0.85, 0.85), rng.uniform(-0.85, 0.85))


def _rebuild_inputs_by_scenario(case_id, v, f, width, height):
    """按场景类型重建语义合法的 v/f 张量，规模与 json 占位形状精确一致。

    跨平台稳定性核心设计：
    - 每个面是独立三角形（顶点不跨面共享），三顶点同深度（平面三角形），
      面深度取自全局阶梯且不同面档位不同——重叠像素的深度胜者由面深度
      唯一决定，既无插值量化碰撞也无等深 tie-break 歧义。
    - json 中的 range_values 随机张量仅用于占位（形状/dtype 信息），
      实际几何按 case_id 固定种子生成，双端一致且可复现。
    """
    scenario = case_id % 6
    num_verts = v.shape[0]
    num_faces = f.shape[0]
    rng = _CaseSeededRng(case_id)

    verts = np.zeros((num_verts, 4), dtype=np.float32)
    faces = np.zeros((num_faces, 3), dtype=np.int32)

    # 所有顶点先置为屏幕外合法默认值（w=1）：后续任何引用——含退化面
    # (vid,vid,vid) 与场景 4 的相邻顶点复制——都不会读到 w=0 的未初始
    # 化顶点（w=0 会使屏幕变换 x/w 产生 NaN）。
    for vid in range(num_verts):
        verts[vid, 0] = 1.5 + 0.01 * vid
        verts[vid, 1] = -0.9
        verts[vid, 2] = 0.63
        verts[vid, 3] = 1.0

    # 有效面数受顶点数约束：每个有效面独享三个顶点（不跨面共享）。
    num_valid_faces = min(num_faces, num_verts // 3)
    for face_no in range(num_faces):
        base = face_no * 3
        if face_no < num_valid_faces:
            # 顶点充足：每面独享三个顶点。
            tri = (base, base + 1, base + 2)
        else:
            # 顶点不足（nv < nf*3）：多余面填退化面（三顶点同索引，零
            # 面积），两平台一致跳过，不影响任何像素。
            vid = base % num_verts
            faces[face_no] = (vid, vid, vid)
            continue
        z = _face_depth(face_no)
        for vid in tri:
            x, y = _screen_xy(rng, scenario, vid)
            if scenario == 4 and vid % 2 == 1:
                # 共线顶点：y 对齐前一顶点，制造零面积退化面。
                verts[vid, 0] = verts[vid - 1, 0] * 0.5
                verts[vid, 1] = verts[vid - 1, 1]
            else:
                verts[vid, 0] = x
                verts[vid, 1] = y
            verts[vid, 2] = z
            verts[vid, 3] = 1.0
        faces[face_no] = tri

    new_v = torch.from_numpy(verts)
    return new_v, torch.from_numpy(faces)


@register("aclnn_rasterizer")
class MethodTorchRasterizerApi(BaseApi):
    """CPU 标杆执行器：定向构造几何 + Torch 小算子拼接 golden。"""

    def __init__(self, task_result: TaskResult):
        super(MethodTorchRasterizerApi, self).__init__(task_result)
        OpsDataset.seed_everything()

    def init_by_input_data(self, input_data: InputDataset):
        case_id = self.task_result.case_config.id
        v = input_data.kwargs["v"]
        f = input_data.kwargs["f"]
        width = int(input_data.kwargs["width"])
        height = int(input_data.kwargs["height"])

        new_v, new_f = _rebuild_inputs_by_scenario(case_id, v, f, width, height)

        v.copy_(new_v)
        f.copy_(new_f)

        # 可选输入 d：非空时按 case_id 种子生成 0~1 深度图（不参与计算）。
        d = input_data.kwargs.get("d")
        if d is not None:
            seed = _CaseSeededRng(case_id + 100000).randrange(1 << 31)
            d.copy_(
                torch.from_numpy(
                    np.random.RandomState(seed)
                    .uniform(0.0, 1.0, (height, width))
                    .astype(np.float32)
                )
            )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        v = input_data.kwargs["v"]
        f = input_data.kwargs["f"]
        width = int(input_data.kwargs["width"])
        height = int(input_data.kwargs["height"])
        return _rasterizer_golden_compute(v, f, width, height)


@register("aclnn_rasterizer_aclnn")
class RasterizerAclnnApi(AclnnBaseApi):
    """pyaclnn 侧执行器：仅指定 ND format。"""

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_ND
