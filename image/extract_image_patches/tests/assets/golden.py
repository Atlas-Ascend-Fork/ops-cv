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

import numpy as np


def _dtype(x):
    return getattr(x.dtype, "name", None) or str(x.dtype).split(".")[-1]


def _format(kwargs, attrs):
    formats = kwargs.get("input_ori_formats") or kwargs.get("input_formats")
    if formats:
        value = str(formats[0]).upper()
        if "NCHW" in value:
            return "NCHW"
        if "NHWC" in value:
            return "NHWC"
        raise ValueError(f"x format must be NHWC or NCHW, got {formats[0]}")
    nchw, nhwc = (
        all(values[1] == 1 for values in attrs),
        all(values[3] == 1 for values in attrs),
    )
    if nchw and nhwc:
        # [1, 1, X, 1] 型属性同时匹配两种布局（NCHW 且 W 维全 1，或 NHWC 且 H 维
        # 全 1），仅凭属性无法判定布局；静默猜测会在错误布局上生成参考值。
        raise ValueError(
            "cannot infer x format from ksizes/strides/rates = [1, 1, *, 1]; "
            "pass input_ori_formats or input_formats explicitly"
        )
    return "NCHW" if nchw else "NHWC"


def _attrs(name, values):
    if len(values) != 4:
        raise ValueError(f"{name} must contain 4 integers, got {values}")
    if any(
        not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
        for value in values
    ):
        raise TypeError(f"{name} must contain integers, got {values}")
    return tuple(int(value) for value in values)


def _validate(x, ksizes, strides, rates, padding, kwargs):
    if len(x.shape) != 4:
        raise ValueError(f"x must be 4D, got {x.shape}")
    # float64：cross_check 判据下调度器按 Promote 将 float32 输入提升为 float64
    # 后传入 golden，白名单须覆盖提升后的 dtype。
    if _dtype(x) not in ("float16", "float32", "float64", "bfloat16", "int8", "uint8"):
        raise TypeError(f"unsupported ExtractImagePatches dtype: {_dtype(x)}")
    ksizes, strides, rates = (
        _attrs("ksizes", ksizes),
        _attrs("strides", strides),
        _attrs("rates", rates),
    )
    fmt = _format(kwargs, (ksizes, strides, rates))
    if padding not in ("SAME", "VALID"):
        raise ValueError(f"padding must be SAME or VALID, got {padding}")
    h_idx, w_idx, c_idx = (1, 2, 3) if fmt == "NHWC" else (2, 3, 1)
    if any(values[0] != 1 or values[c_idx] != 1 for values in (ksizes, strides, rates)):
        raise ValueError("ksizes/strides/rates values for N and C dimensions must be 1")
    k_h, k_w = ksizes[h_idx], ksizes[w_idx]
    s_h, s_w = strides[h_idx], strides[w_idx]
    r_h, r_w = rates[h_idx], rates[w_idx]
    if k_h <= 0 or k_w <= 0 or s_h <= 0 or s_w <= 0 or r_h < 1 or r_w < 1:
        raise ValueError(
            "spatial ksizes/strides must be positive and spatial rates must be at least 1"
        )
    return fmt, [1, k_h, k_w, 1], [1, s_h, s_w, 1], [1, r_h, r_w, 1]


def _extract_image_patches_torch(x_nhwc, sizes, strides, rates, padding):
    """int8/uint8 的 Torch 参考实现。

    TensorFlow 的 ExtractImagePatches 仅注册浮点内核，int8/uint8 无法经由 TF
    生成参考；本算子为纯数据搬运算子，用 Torch 基础算子按下标公式直接搬运即可
    保持逐位一致。ndarray 与 Tensor 之间的转换仅作桥接，采样、越界判定与补零
    等参考计算全部由 Torch 算子完成。输入输出均为 NHWC，展平顺序为
    (p * kW + q) * C + c，SAME 越界补零。
    """
    import torch

    x = torch.as_tensor(x_nhwc)
    _, in_h, in_w, in_c = x.shape
    _, k_h, k_w, _ = sizes
    _, s_h, s_w, _ = strides
    _, r_h, r_w, _ = rates
    eff_h = (k_h - 1) * r_h + 1
    eff_w = (k_w - 1) * r_w + 1
    if padding == "SAME":
        out_h = -(-in_h // s_h)
        out_w = -(-in_w // s_w)
        pad_h_top = max((out_h - 1) * s_h + eff_h - in_h, 0) // 2
        pad_w_left = max((out_w - 1) * s_w + eff_w - in_w, 0) // 2
    else:
        out_h = max((in_h - eff_h + s_h) // s_h, 0)
        out_w = max((in_w - eff_w + s_w) // s_w, 0)
        pad_h_top = pad_w_left = 0
    i_idx = torch.arange(out_h).reshape(1, out_h, 1, 1, 1)
    j_idx = torch.arange(out_w).reshape(1, 1, out_w, 1, 1)
    p_idx = torch.arange(k_h).reshape(1, 1, 1, k_h, 1)
    q_idx = torch.arange(k_w).reshape(1, 1, 1, 1, k_w)
    h_pos = i_idx * s_h + p_idx * r_h - pad_h_top
    w_pos = j_idx * s_w + q_idx * r_w - pad_w_left
    valid = (h_pos >= 0) & (h_pos < in_h) & (w_pos >= 0) & (w_pos < in_w)
    h_safe = torch.where(valid, h_pos, torch.zeros_like(h_pos))
    w_safe = torch.where(valid, w_pos, torch.zeros_like(w_pos))
    # (N, out_h, out_w, k_h, k_w, C)：n 维经广播对齐全通道后整体 gather
    gathered = x[:, h_safe, w_safe, :]
    zero = torch.zeros((), dtype=x.dtype)
    result = torch.where(valid[..., None], gathered, zero)
    return result.reshape(x.shape[0], out_h, out_w, k_h * k_w * in_c)


def _extract_image_patches(x, sizes, strides, rates, padding, fmt):
    if _dtype(x) in ("int8", "uint8"):
        if hasattr(x, "numpy"):
            x = x.numpy()
        x_nhwc = x if fmt == "NHWC" else x.transpose(0, 2, 3, 1)
        output = _extract_image_patches_torch(
            x_nhwc, sizes, strides, rates, padding
        ).numpy()
        return output if fmt == "NHWC" else output.transpose(0, 3, 1, 2)

    import tensorflow as tf

    x = x if fmt == "NHWC" else tf.transpose(x, (0, 2, 3, 1))
    if hasattr(tf, "extract_image_patches"):
        output = tf.extract_image_patches(
            x, ksizes=sizes, strides=strides, rates=rates, padding=padding
        )
    else:
        output = tf.image.extract_patches(
            x, sizes=sizes, strides=strides, rates=rates, padding=padding
        )
    return output if fmt == "NHWC" else tf.transpose(output, (0, 3, 1, 2))


class ExtractImagePatchesSpec:
    def golden(x, *, ksizes, strides, rates, padding, **kwargs):
        fmt, sizes, strides, rates = _validate(
            x, ksizes, strides, rates, padding, kwargs
        )
        output = _extract_image_patches(x, sizes, strides, rates, padding, fmt)
        # TF 路径返回 EagerTensor，torch 路径（int8/uint8）桥接回 ndarray。
        return [output.numpy() if hasattr(output, "numpy") else np.asarray(output)]

    class TfThirdPartyImpl:
        def __init__(self, *, ksizes, strides, rates, padding, **kwargs):
            self.ksizes, self.strides, self.rates, self.padding = (
                ksizes,
                strides,
                rates,
                padding,
            )

        def __call__(self, x, **kwargs):
            fmt, sizes, strides, rates = _validate(
                x, self.ksizes, self.strides, self.rates, self.padding, kwargs
            )
            return [_extract_image_patches(x, sizes, strides, rates, self.padding, fmt)]

    third_party = {"tf": TfThirdPartyImpl}
    tolerance = {
        dtype: {"standard": "cross_check", "level": "L0"}
        for dtype in ("float16", "float32", "bfloat16")
    }
    tolerance.update(
        {dtype: {"standard": "binary_equal"} for dtype in ("int8", "uint8")}
    )


__spec__ = {"extract_image_patches": "ExtractImagePatchesSpec"}
