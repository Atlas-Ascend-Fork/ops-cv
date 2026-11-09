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
"""NMSWithMask input tensor generator"""

# Standard Packages

# Third-Party Packages

import numpy as np

__input__ = {
    "kernel": {"nms_with_mask": "nms_with_mask_input"},
}


def generate_random_boxes(
    num_boxes: int,
    img_height: int = 1,
    img_width: int = 1,
    min_box_size: float = 0.05,
    max_box_size: float = 0.5,
    input_dtype=np.float32,
):
    """
    参数：
    num_boxes: 要生成的检测框数量
    img_height: 图像高度（用于像素坐标，归一化时设为1即可）
    img_width: 图像宽度（用于像素坐标，归一化时设为1即可）
    min_box_size: 框的最小边长（归一化时为0~1，像素坐标时为像素值）
    max_box_size: 框的最大边长（归一化时为0~1，像素坐标时为像素值）
    返回：
    box_scores: 形状为[num_boxes, 5]的numpy array，格式为[y1, x1, y2, x2, score]
    """

    # 确保最小/最大尺寸合法

    assert min_box_size > 0, "min_box_size必须大于0"
    assert max_box_size > min_box_size, "max_box_size必须大于min_box_size"
    assert img_height > 0 and img_width > 0, "图像尺寸必须大于0"

    # 1. 随机生成框的左上角坐标 (y1, x1)

    # 为了保证框的边长足够，左上角坐标的最大值需预留出最大框的空间

    max_y1 = img_height - max_box_size
    max_x1 = img_width - max_box_size
    y1 = np.random.uniform(low=0, high=max_y1, size=num_boxes)
    x1 = np.random.uniform(low=0, high=max_x1, size=num_boxes)

    # 2. 随机生成框的高度和宽度（在min和max之间）

    box_heights = np.random.uniform(low=min_box_size, high=max_box_size, size=num_boxes)
    box_widths = np.random.uniform(low=min_box_size, high=max_box_size, size=num_boxes)

    # 3. 计算右下角坐标 (y2, x2)，保证y2 > y1，x2 > x1

    y2 = y1 + box_heights
    x2 = x1 + box_widths

    scores = np.random.normal(loc=0.7, scale=0.2, size=num_boxes)
    scores = np.clip(scores, 0.0, 1.0)  # 限制在0~1之间

    # 4. 拼接成检测框张量 [y1, x1, y2, x2, score]

    box_scores = np.column_stack((y1, x1, y2, x2, scores)).astype(input_dtype)
    box_scores = np.array(box_scores)

    # 5. 生成置信度分数（0~1的随机值，可模拟不同置信度分布）

    # 按分數降序排序

    box_scores = box_scores[np.argsort(box_scores[:, 4])[::-1]]
    return box_scores


def nms_with_mask_input(box_scores, **kwargs):
    del kwargs
    n = box_scores.shape[0]
    box_scores_out = generate_random_boxes(n, input_dtype=box_scores.dtype)
    return (box_scores_out,)
