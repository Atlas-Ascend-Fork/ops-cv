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
# Three-party golden for yolo (Kernel / GEIR path).
# Golden wrapper 接收 numpy.ndarray，third_party 接收 torch.Tensor；二者共享同一 _torch_compute 计算核心。
# dtype 提升/回写由 TTK --golden-mode Promote 外部处理，本文件不在内部做 float16->float32 提升或回写。

__spec__ = {
    "yolo": "YoloTestSpec",
}

import torch


def _torch_compute(x, boxes, coords, classes, yolo_version, softmax, background):
    """共享 Torch 计算核心 —— Tensor -> Tensor，保留调用方 dtype/device 与数学顺序。

    被 golden wrapper（numpy->torch 转入后）和 third_party __call__（直接接收 provider Tensor）共同调用。
    禁止：numpy/CPU 边界副作用、无条件 .cpu()/.numpy()、固定 dtype cast、float32 提升/回写。
    新建 Tensor 通过 torch.sigmoid/torch.softmax/torch.cat/torch.nn.functional.pad 等 API 自动继承输入 dtype/device。

    输入:
      x: torch.Tensor, shape (N, boxes*(coords+1+classes), H, W), dtype float16/float32
      boxes: int, 锚框数量
      coords: int, 坐标参数数（固定 4）
      classes: int, 类别数
      yolo_version: str, "V2" 或 "V3"
      softmax: bool, 是否做 softmax（仅 V2 有效）
      background: bool, obj 和 classes 操作类型（仅 V2 有效）

    返回:
      list[torch.Tensor]: [coord_data, obj_prob, classes_prob]
        coord_data:   (N, boxes*coords, H*W_aligned)
        obj_prob:     (N, boxes*H*W_aligned)
        classes_prob: (N, classes, boxes*H*W_aligned)
    """
    N, C, H, W = x.shape
    HW = H * W
    B = boxes

    # 步骤1: 坐标处理
    # x, y 做 sigmoid；w, h 做 move（注意输出排列 (x, y, h, w)，w/h 位置交换）
    coord_x = torch.sigmoid(x[:, 0:B, :, :])
    coord_y = torch.sigmoid(x[:, B : 2 * B, :, :])
    coord_w = x[:, 2 * B : 3 * B, :, :]
    coord_h = x[:, 3 * B : 4 * B, :, :]

    coord_data = torch.cat(
        [
            coord_x.reshape(N, B, HW),
            coord_y.reshape(N, B, HW),
            coord_h.reshape(N, B, HW),
            coord_w.reshape(N, B, HW),
        ],
        dim=1,
    )

    # 步骤2: 确定 yolo_mode（由 yolo_version + softmax + background 决定）
    if yolo_version.upper() == "V3":
        yolo_mode = "YOLO_MODE_1"
    else:
        if not softmax and not background:
            yolo_mode = "YOLO_MODE_1"
        elif softmax and not background:
            yolo_mode = "YOLO_MODE_2"
        elif not softmax and background:
            yolo_mode = "YOLO_MODE_3"
        else:
            yolo_mode = "YOLO_MODE_4"

    # 步骤3: 目标置信度与类别概率处理
    obj_raw = x[:, 4 * B : 5 * B, :, :]
    cls_raw = x[:, 5 * B : 5 * B + classes * B, :, :]
    cls_raw = cls_raw.reshape(N, classes, B, H, W)

    if yolo_mode == "YOLO_MODE_1":
        obj_prob = torch.sigmoid(obj_raw).reshape(N, B * HW)
        classes_prob = torch.sigmoid(cls_raw).reshape(N, classes, B * HW)
    elif yolo_mode == "YOLO_MODE_2":
        obj_prob = torch.sigmoid(obj_raw).reshape(N, B * HW)
        classes_prob = torch.softmax(cls_raw, dim=1).reshape(N, classes, B * HW)
    elif yolo_mode == "YOLO_MODE_3":
        obj_prob = obj_raw.reshape(N, B * HW)
        classes_prob = torch.sigmoid(cls_raw).reshape(N, classes, B * HW)
    else:
        all_raw = torch.cat(
            [
                obj_raw.unsqueeze(1),
                cls_raw,
            ],
            dim=1,
        )
        all_softmax = torch.softmax(all_raw, dim=1)
        obj_prob = all_softmax[:, 0, :, :, :].reshape(N, B * HW)
        classes_prob = all_softmax[:, 1:, :, :, :].reshape(N, classes, B * HW)

    # 步骤4: ceilx 内存对齐 padding（对齐 canndev 1.0 infershape）
    # CeilX(size, 32) = (size + 31) // 32 * 32，输出最后一维 = CeilX(...*2, 32) // 2
    hw_aligned = (HW * 2 + 32 + 31) // 32 * 32 // 2
    boxes_hw_aligned = (B * HW * 2 + 32 + 31) // 32 * 32 // 2

    coord_pad = hw_aligned - HW
    if coord_pad > 0:
        coord_data = torch.nn.functional.pad(
            coord_data, (0, coord_pad), mode="constant", value=0
        )

    obj_pad = boxes_hw_aligned - (B * HW)
    if obj_pad > 0:
        obj_prob = torch.nn.functional.pad(
            obj_prob, (0, obj_pad), mode="constant", value=0
        )

    classes_pad = boxes_hw_aligned - (B * HW)
    if classes_pad > 0:
        classes_prob = torch.nn.functional.pad(
            classes_prob, (0, classes_pad), mode="constant", value=0
        )

    return [coord_data, obj_prob, classes_prob]


class YoloTestSpec:
    """Kernel / GEIR 共用 TestSpec。

    Input (from def.cpp):
      x: (N, boxes*(coords+1+classes), H, W)  float16 / float32

    Attributes (all OPTIONAL with defaults):
      boxes: int (default 3)
      coords: int (default 4)
      classes: int (default 80)
      yolo_version: str (default "V3")
      softmax: bool (default False)
      background: bool (default False)
      softmaxtree: bool (default False, 固定未使用)

    Outputs:
      coord_data:   (N, boxes*coords, H*W_aligned)
      obj_prob:     (N, boxes*H*W_aligned)
      classes_prob: (N, classes, boxes*H*W_aligned)
    """

    def golden(
        x,
        *,
        boxes=3,
        coords=4,
        classes=80,
        yolo_version="V3",
        softmax=False,
        background=False,
        softmaxtree=False,
        **kwargs,
    ):
        """Golden wrapper —— numpy->torch 转入，调用共享核心，torch->numpy 转出。

        不做 float16->float32 提升，不做 dtype 回写（由 TTK --golden-mode Promote 外部处理）。
        softmaxtree 属性不影响计算逻辑，接收但不下传给共享核心。
        """
        x_torch = torch.from_numpy(x).contiguous()
        results = _torch_compute(
            x_torch, boxes, coords, classes, yolo_version, softmax, background
        )
        return [r.detach().cpu().numpy() for r in results]

    class ThirdPartyImpl:
        """third_party 类形式 —— 接收 provider Tensor，直接调用共享核心。

        参数绑定契约（参照 BallQuery / CosineSimilarity 模式）：
          __init__(**kwargs): TTK bind_params 按名匹配，属性池注入 kwargs。
            在此完成属性绑定；yolo_mode 决策与对齐常量预计算由共享核心自包含处理，
            避免与 golden wrapper 重复维护两套计算逻辑。
          __call__(x, **kwargs): 只保留纯 torch 计算（计入性能比对时间），调 _torch_compute。
        """

        def __init__(
            self,
            *,
            boxes=3,
            coords=4,
            classes=80,
            yolo_version="V3",
            softmax=False,
            background=False,
            softmaxtree=False,
            **kwargs,
        ):
            self.boxes = boxes
            self.coords = coords
            self.classes = classes
            self.yolo_version = yolo_version
            self.softmax = softmax
            self.background = background

        def __call__(self, x, **kwargs):
            return _torch_compute(
                x,
                self.boxes,
                self.coords,
                self.classes,
                self.yolo_version,
                self.softmax,
                self.background,
            )

    third_party = {"torch": ThirdPartyImpl}

    tolerance = {
        "float16": {"standard": "cross_check", "level": "L1"},
        "float32": {"standard": "cross_check", "level": "L1"},
    }
