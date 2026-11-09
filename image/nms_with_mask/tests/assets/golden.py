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
"""TTK golden for NMSWithMask.

The reference is TensorFlow NonMaxSuppressionV4 (max_output_size = num_boxes,
score_threshold = -1): boxes are visited in descending score order and later
boxes with IoU > iou_threshold are suppressed.  selected_boxes is a copy of
box_scores, selected_idx is 0..num_boxes-1 and selected_mask marks the boxes
kept by NMS.

TensorFlow is imported with RTLD_DEEPBIND.  The TTK worker has already loaded
torch and the CANN runtime (libruntime.so is opened with RTLD_GLOBAL); their
C++ template weak symbols otherwise get merged with TensorFlow's clang-built
copies and SIGSEGV during TensorFlow's static initialization.  DEEPBIND keeps
each library on its own symbols while the flag is active for this import.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

__spec__ = {"nms_with_mask": "NmsWithMaskTestSpec"}
__golden__ = {
    "kernel": {"nms_with_mask": "nms_with_mask_golden"},
}

_RTLD_DEEPBIND = 0x00008


def _import_tensorflow():
    if "tensorflow" in sys.modules:
        return sys.modules["tensorflow"]
    old_flags = sys.getdlopenflags()
    sys.setdlopenflags(old_flags | _RTLD_DEEPBIND)
    try:
        import tensorflow as tf
    finally:
        sys.setdlopenflags(old_flags)
    return tf


def nms_with_mask_golden(box_scores, *, iou_threshold=0.5, **kwargs):
    """Generate the selected boxes, indices and mask for NMSWithMask."""
    del kwargs
    tf = _import_tensorflow()
    input_type = box_scores.dtype
    if "bfloat16" in str(input_type) or "float16" in str(input_type):
        box_scores = box_scores.astype(np.float32)
    num_boxes = box_scores.shape[0]
    boxes = box_scores[:, :4]
    scores = box_scores[:, 4]
    nms_res = tf.raw_ops.NonMaxSuppressionV4(
        boxes=boxes,
        scores=scores,
        max_output_size=num_boxes,
        iou_threshold=iou_threshold,
        score_threshold=-1,
    )
    selected_indices = nms_res.selected_indices.numpy()
    selected_mask = np.zeros(num_boxes, dtype=np.uint8)
    selected_mask[selected_indices] = 1
    if "bfloat16" in str(input_type) or "float16" in str(input_type):
        box_scores = box_scores.astype(input_type)
    return (box_scores, np.arange(0, num_boxes, dtype=np.int32), selected_mask)


class NmsWithMaskTestSpec:
    tolerance = {
        "float32": {"standard": "binary_equal"},
        "float16": {"standard": "binary_equal"},
        "bfloat16": {"standard": "binary_equal"},
        "int32": {"standard": "binary_equal"},
        "uint8": {"standard": "binary_equal"},
    }

    def golden(box_scores, *, iou_threshold=0.5, **kwargs):
        return nms_with_mask_golden(box_scores, iou_threshold=iou_threshold, **kwargs)

    def customize_inputs(box_scores, **kwargs):
        input_path = Path(__file__).with_name("inputs.py")
        module_spec = importlib.util.spec_from_file_location(
            "nms_with_mask_inputs", input_path
        )
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        return module.nms_with_mask_input(box_scores, **kwargs)
