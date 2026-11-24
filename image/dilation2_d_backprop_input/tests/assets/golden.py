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

"""dilation2_d_backprop_input 的 Kernel、GEIR 和 TensorFlow E2E Golden。

Kernel/GEIR 的 golden 收到 numpy.ndarray，转 tf tensor 后调 tf.raw_ops.Dilation2DBackpropInput 计算，
结果转回 numpy；
third_party 的 tf vendor 收到 tf.Tensor（框架按 vendor 自动转换），直接调同一接口，无需手动转换。
TensorFlow E2E 按原生 API 签名接收参数并直接执行该接口。
无 torch 现成接口，以 tf.raw_ops.Dilation2DBackpropInput 作为三方标杆。
"""

__spec__ = {
    "dilation2_d_backprop_input": "Dilation2DBackpropInputKernelSpec",
    "tf.raw_ops.Dilation2DBackpropInput": "TfDilation2DBackpropInputSpec",
}

import numpy as np
import tensorflow as tf


class Dilation2DBackpropInputKernelSpec:
    """Kernel / GEIR 流程 — golden 收到 numpy.ndarray，third_party 收到 tf.Tensor"""

    def golden(
        x,
        filter,
        out_backprop,
        *,
        strides,
        rates,
        padding_mode="SAME",
        pads=(0, 0, 0, 0),
        ceil_mode=False,
        data_format="NHWC",
        **kwargs,
    ):
        x = np.ascontiguousarray(x)
        filter = np.ascontiguousarray(filter)
        out_backprop = np.ascontiguousarray(out_backprop)

        x_t = tf.constant(x)
        filter_t = tf.constant(filter)
        out_backprop_t = tf.constant(out_backprop)

        if data_format == "NCHW":
            x_nhwc = tf.transpose(x_t, [0, 2, 3, 1])
            filter_nhwc = tf.transpose(filter_t, [1, 2, 0])
            out_bp_nhwc = tf.transpose(out_backprop_t, [0, 2, 3, 1])
            stride_n, stride_c, stride_h, stride_w = strides
            rate_n, rate_c, rate_h, rate_w = rates
        else:
            x_nhwc = x_t
            filter_nhwc = filter_t
            out_bp_nhwc = out_backprop_t
            stride_n, stride_h, stride_w, stride_c = strides
            rate_n, rate_h, rate_w, rate_c = rates

        filter_h = int(filter_nhwc.shape[0])
        filter_w = int(filter_nhwc.shape[1])
        window_h = (filter_h - 1) * rate_h + 1
        window_w = (filter_w - 1) * rate_w + 1

        input_h = int(x_nhwc.shape[1])
        input_w = int(x_nhwc.shape[2])

        if padding_mode == "SAME":
            result_nhwc = tf.raw_ops.Dilation2DBackpropInput(
                input=x_nhwc,
                filter=filter_nhwc,
                out_backprop=out_bp_nhwc,
                strides=[stride_n, stride_h, stride_w, stride_c],
                rates=[rate_n, rate_h, rate_w, rate_c],
                padding="SAME",
            )
        elif padding_mode == "VALID":
            result_nhwc = tf.raw_ops.Dilation2DBackpropInput(
                input=x_nhwc,
                filter=filter_nhwc,
                out_backprop=out_bp_nhwc,
                strides=[stride_n, stride_h, stride_w, stride_c],
                rates=[rate_n, rate_h, rate_w, rate_c],
                padding="VALID",
            )
        else:
            pad_top, pad_bottom, pad_left, pad_right = pads
            if ceil_mode:
                out_h = (
                    input_h - window_h + pad_top + pad_bottom + stride_h - 1
                ) // stride_h + 1
                out_w = (
                    input_w - window_w + pad_left + pad_right + stride_w - 1
                ) // stride_w + 1
            else:
                out_h = (input_h - window_h + pad_top + pad_bottom) // stride_h + 1
                out_w = (input_w - window_w + pad_left + pad_right) // stride_w + 1
            out_h = max(out_h, 1)
            out_w = max(out_w, 1)
            pad_bottom = max((out_h - 1) * stride_h + window_h - input_h - pad_top, 0)
            pad_right = max((out_w - 1) * stride_w + window_w - input_w - pad_left, 0)

            min_val = tf.cast(-3.402823466e38, x_nhwc.dtype)
            padded_x = tf.pad(
                x_nhwc,
                [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                constant_values=min_val,
            )
            result_nhwc = tf.raw_ops.Dilation2DBackpropInput(
                input=padded_x,
                filter=filter_nhwc,
                out_backprop=out_bp_nhwc,
                strides=[stride_n, stride_h, stride_w, stride_c],
                rates=[rate_n, rate_h, rate_w, rate_c],
                padding="VALID",
            )
            result_nhwc = result_nhwc[
                :, pad_top : pad_top + input_h, pad_left : pad_left + input_w, :
            ]

        if data_format == "NCHW":
            result = tf.transpose(result_nhwc, [0, 3, 1, 2])
        else:
            result = result_nhwc

        return [result.numpy()]

    class ThirdPartyImpl:
        """tf vendor 三方标杆 — 入参为 tf.Tensor，与 golden 同后端保证语义一致。

        __init__ 前置：data_format 判定、strides/rates 重排、filter/window 维度提取、
        CALCULATED 模式 padding 计算与极小值常量准备、transpose perm 常量、
        pad 配置列表与结果切片索引、padding 分支标志。
        __call__ 只留纯计算：transpose → pad(仅CALCULATED) → tf.raw_ops → slice → transpose back。
        """

        def __init__(
            self,
            x,
            filter,
            out_backprop,
            *,
            strides,
            rates,
            padding_mode="SAME",
            pads=(0, 0, 0, 0),
            ceil_mode=False,
            data_format="NHWC",
            **kwargs,
        ):
            self.padding_mode = padding_mode
            self.is_nchw = data_format == "NCHW"
            self.is_calc_padding = padding_mode == "CALCULATED"

            if self.is_nchw:
                x_h, x_w = int(x.shape[2]), int(x.shape[3])
                filter_h, filter_w = int(filter.shape[1]), int(filter.shape[2])
                stride_n, stride_c, stride_h, stride_w = strides
                rate_n, rate_c, rate_h, rate_w = rates
            else:
                x_h, x_w = int(x.shape[1]), int(x.shape[2])
                filter_h, filter_w = int(filter.shape[0]), int(filter.shape[1])
                stride_n, stride_h, stride_w, stride_c = strides
                rate_n, rate_h, rate_w, rate_c = rates

            self.strides_nhwc = [stride_n, stride_h, stride_w, stride_c]
            self.rates_nhwc = [rate_n, rate_h, rate_w, rate_c]

            window_h = (filter_h - 1) * rate_h + 1
            window_w = (filter_w - 1) * rate_w + 1

            self.x_perm = [0, 2, 3, 1] if self.is_nchw else None
            self.filter_perm = [1, 2, 0] if self.is_nchw else None
            self.out_bp_perm = [0, 2, 3, 1] if self.is_nchw else None
            self.result_perm = [0, 3, 1, 2] if self.is_nchw else None

            if self.is_calc_padding:
                pad_top, pad_bottom, pad_left, pad_right = pads
                if ceil_mode:
                    out_h = (
                        x_h - window_h + pad_top + pad_bottom + stride_h - 1
                    ) // stride_h + 1
                    out_w = (
                        x_w - window_w + pad_left + pad_right + stride_w - 1
                    ) // stride_w + 1
                else:
                    out_h = (x_h - window_h + pad_top + pad_bottom) // stride_h + 1
                    out_w = (x_w - window_w + pad_left + pad_right) // stride_w + 1
                out_h = max(out_h, 1)
                out_w = max(out_w, 1)
                self.pad_top = pad_top
                self.pad_bottom = max(
                    (out_h - 1) * stride_h + window_h - x_h - pad_top, 0
                )
                self.pad_left = pad_left
                self.pad_right = max(
                    (out_w - 1) * stride_w + window_w - x_w - pad_left, 0
                )
                self.input_h = x_h
                self.input_w = x_w
                self.min_val = tf.cast(-3.402823466e38, x.dtype)
                self.pad_config = [
                    [0, 0],
                    [self.pad_top, self.pad_bottom],
                    [self.pad_left, self.pad_right],
                    [0, 0],
                ]
                self.out_slice = (
                    slice(None),
                    slice(self.pad_top, self.pad_top + self.input_h),
                    slice(self.pad_left, self.pad_left + self.input_w),
                    slice(None),
                )

            # 输入侧预处理前置到 __init__（__call__ 计入性能比对时间，只留纯计算）
            if self.is_nchw:
                self.x_nhwc = tf.transpose(x, self.x_perm)
                self.filter_nhwc = tf.transpose(filter, self.filter_perm)
                self.out_bp_nhwc = tf.transpose(out_backprop, self.out_bp_perm)
            else:
                self.x_nhwc = x
                self.filter_nhwc = filter
                self.out_bp_nhwc = out_backprop

            if self.is_calc_padding:
                self.input_for_compute = tf.pad(
                    self.x_nhwc, self.pad_config, constant_values=self.min_val
                )
                self.tf_padding = "VALID"
            else:
                self.input_for_compute = self.x_nhwc
                self.tf_padding = self.padding_mode

        def __call__(self, **kwargs):
            result_nhwc = tf.raw_ops.Dilation2DBackpropInput(
                input=self.input_for_compute,
                filter=self.filter_nhwc,
                out_backprop=self.out_bp_nhwc,
                strides=self.strides_nhwc,
                rates=self.rates_nhwc,
                padding=self.tf_padding,
            )
            if self.is_calc_padding:
                result_nhwc = result_nhwc[self.out_slice]
            if self.is_nchw:
                result = tf.transpose(result_nhwc, self.result_perm)
            else:
                result = result_nhwc

            return [result]

    third_party = {"tf": ThirdPartyImpl}
    tolerance = {"float32": {"standard": "cross_check", "level": "L1"}}


class TfDilation2DBackpropInputSpec:
    """TensorFlow E2E Golden，参数名与 tf.raw_ops 接口一致。"""

    @staticmethod
    def golden(input, filter, out_backprop, strides, rates, padding, name=None):
        return [
            tf.raw_ops.Dilation2DBackpropInput(
                input=input,
                filter=filter,
                out_backprop=out_backprop,
                strides=strides,
                rates=rates,
                padding=padding,
                name=name,
            )
        ]
