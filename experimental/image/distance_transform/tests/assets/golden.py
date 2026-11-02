#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Kernel golden implementation for DistanceTransform."""

import numpy as np

__golden__ = {"kernel": {"distance_transform": "distance_transform_golden"}}


def _images(x):
    if x.ndim == 2:
        return x.reshape(1, *x.shape)
    if x.ndim == 3:
        return x
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0]
    raise ValueError("x must have shape (H,W), (N,H,W), or (N,1,H,W)")


def _l2_chamfer(image, mask_size):
    """Independent shortest-path reference for OpenCV's L2 chamfer masks."""
    import heapq

    height, width = image.shape
    result = np.full((height, width), float(height + width), dtype=np.float32)
    queue = []
    for row, col in np.argwhere(image == 0):
        result[row, col] = 0.0
        heapq.heappush(queue, (0.0, int(row), int(col)))

    axial = 0.955 if mask_size == 3 else 1.0
    diagonal = 1.3693 if mask_size == 3 else 1.4
    moves = [
        (-1, 0, axial),
        (1, 0, axial),
        (0, -1, axial),
        (0, 1, axial),
        (-1, -1, diagonal),
        (-1, 1, diagonal),
        (1, -1, diagonal),
        (1, 1, diagonal),
    ]
    if mask_size == 5:
        for dr, dc in (
            (-2, -1),
            (-2, 1),
            (2, -1),
            (2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
        ):
            moves.append((dr, dc, 2.1969))

    while queue:
        distance, row, col = heapq.heappop(queue)
        if distance > float(result[row, col]):
            continue
        for dr, dc, cost in moves:
            next_row, next_col = row + dr, col + dc
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            candidate = distance + cost
            if candidate < float(result[next_row, next_col]):
                result[next_row, next_col] = candidate
                heapq.heappush(queue, (candidate, next_row, next_col))
    return result


def distance_transform_golden(x, distance_type="L2", mask_size=0, **kwargs):
    """Independent brute-force reference for correctness cases."""
    del kwargs
    if x.dtype != np.uint8:
        raise TypeError("x must be uint8")
    mode = distance_type.upper()
    if mode not in ("L1", "L2", "CHESSBOARD"):
        raise ValueError("distance_type must be L1, L2, or CHESSBOARD")

    if mask_size not in (0, 3, 5):
        raise ValueError("mask_size must be 0, 3, or 5")
    effective_mask_size = mask_size if mode == "L2" else 3
    images = _images(x)
    result = np.empty(images.shape, dtype=np.float32)
    height, width = images.shape[-2:]
    sentinels = {
        "L1": height + width,
        "L2": np.hypot(height, width) if effective_mask_size == 0 else height + width,
        "CHESSBOARD": max(height, width),
    }
    for batch, image in enumerate(images):
        background = np.argwhere(image == 0)
        if mode == "L2" and effective_mask_size in (3, 5):
            result[batch] = _l2_chamfer(image, effective_mask_size)
            continue

        if background.size == 0:
            result[batch].fill(sentinels[mode])
            continue
        for row in range(height):
            for col in range(width):
                if image[row, col] == 0:
                    result[batch, row, col] = 0.0
                    continue
                delta = np.abs(background - np.array([row, col]))
                if mode == "L1":
                    value = np.min(delta[:, 0] + delta[:, 1])
                elif mode == "CHESSBOARD":
                    value = np.min(np.maximum(delta[:, 0], delta[:, 1]))
                else:
                    value = np.sqrt(
                        np.min(np.sum(delta.astype(np.float64) ** 2, axis=1))
                    )
                result[batch, row, col] = value
    return result.reshape(x.shape)
