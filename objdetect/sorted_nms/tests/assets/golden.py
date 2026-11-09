#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
from pathlib import Path

import numpy as np


def _load_customize_inputs():
    input_path = Path(__file__).with_name("input.py")
    spec = importlib.util.spec_from_file_location("sorted_nms_test_input", input_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.customize_inputs


__spec__ = {"sorted_nms": "SortedNmsTestSpec"}


class SortedNmsTestSpec:
    """Deterministic input generation and a sequential SortedNMS reference."""

    tolerance = {"int32": {"standard": "binary_equal"}}
    customize_inputs = staticmethod(_load_customize_inputs())

    @staticmethod
    def compare(*values, **kwargs):
        """Compare the logical output and the meaningful dynamic-shape metadata."""
        if len(values) != 4:
            return {
                "pass": False,
                "precision": "INVALID_OUTPUT_COUNT",
                "error_info": f"expected 2 outputs and 2 goldens, got {len(values)} values",
            }

        selected_output, shape_output, selected_golden, shape_golden = values
        output_shape = np.asarray(shape_output, dtype=np.uint64).reshape(-1)
        golden_shape = np.asarray(shape_golden, dtype=np.uint64).reshape(-1)
        if output_shape.size < 2 or golden_shape.size < 2:
            return {
                "pass": False,
                "precision": "INVALID_SHAPE_METADATA",
                "error_info": "dynamic-shape metadata must contain rank and at least one dimension",
            }

        # TTK marks the uint64 encoding in bit 31 of the first metadata word.
        rank_mask = np.uint64(0x7FFFFFFF)
        output_rank = int(output_shape[0] & rank_mask)
        golden_rank = int(golden_shape[0] & rank_mask)
        output_count = int(output_shape[1])
        golden_count = int(golden_shape[1])
        output_values = np.asarray(selected_output).reshape(-1)
        golden_values = np.asarray(selected_golden).reshape(-1)

        errors = []
        if output_rank != 1 or golden_rank != 1:
            errors.append(f"rank mismatch: output={output_rank}, golden={golden_rank}")
        if output_count != golden_count or golden_count != golden_values.size:
            errors.append(
                f"selected count mismatch: output={output_count}, "
                f"metadata_golden={golden_count}, golden={golden_values.size}"
            )
        if output_count > output_values.size:
            errors.append(
                f"selected count {output_count} exceeds output buffer {output_values.size}"
            )
        elif not np.array_equal(output_values[:output_count], golden_values):
            errors.append("selected_indices differ from golden")

        passed = not errors
        return {
            "pass": passed,
            "precision": "BINARY_EQUAL" if passed else "MISMATCH",
            "error_info": "; ".join(errors),
        }

    @staticmethod
    def golden(
        boxes,
        sorted_scores,
        input_indices,
        max_output_size,
        iou_threshold,
        score_threshold,
        *,
        offset=0,
        **kwargs,
    ):
        boxes_num = boxes.shape[0]
        if sorted_scores.size > 1 and np.any(sorted_scores[:-1] < sorted_scores[1:]):
            raise ValueError("sorted_scores must be sorted in non-increasing order")
        max_out = min(max(int(max_output_size.reshape(-1)[0]), 0), boxes_num)
        iou_thr = float(iou_threshold.reshape(-1)[0])
        score_thr = float(score_threshold.reshape(-1)[0])
        boxes_f32 = boxes.astype(np.float32)
        suppressed = np.zeros(boxes_num, dtype=np.bool_)
        selected = []

        for sorted_pos in range(boxes_num):
            if len(selected) >= max_out:
                break
            score = float(sorted_scores[sorted_pos])
            current = int(input_indices[sorted_pos])
            if (
                suppressed[sorted_pos]
                or score <= score_thr
                or current < 0
                or current >= boxes_num
            ):
                continue

            selected.append(current)
            current_box = boxes_f32[current]
            for next_pos in range(sorted_pos + 1, boxes_num):
                if suppressed[next_pos] or float(sorted_scores[next_pos]) <= score_thr:
                    continue
                next_index = int(input_indices[next_pos])
                if next_index < 0 or next_index >= boxes_num:
                    continue
                next_box = boxes_f32[next_index]
                width = max(
                    0.0,
                    min(current_box[2], next_box[2])
                    - max(current_box[0], next_box[0])
                    + offset,
                )
                height = max(
                    0.0,
                    min(current_box[3], next_box[3])
                    - max(current_box[1], next_box[1])
                    + offset,
                )
                intersection = width * height
                current_area = max(0.0, current_box[2] - current_box[0] + offset) * max(
                    0.0, current_box[3] - current_box[1] + offset
                )
                next_area = max(0.0, next_box[2] - next_box[0] + offset) * max(
                    0.0, next_box[3] - next_box[1] + offset
                )
                union = current_area + next_area - intersection
                overlap = intersection / union if union > 0.0 else 0.0
                if overlap > iou_thr:
                    suppressed[next_pos] = True

        return [np.asarray(selected, dtype=np.int32)]
