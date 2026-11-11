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
import torch


__spec__ = {
    "img_raw_decode_post_handle": "ImgRawDecodePostHandleTestSpec",
}


def _img_raw_decode_post_handle_compute(
    img_channel_0,
    img_channel_1,
    img_channel_2,
    img_channel_3,
    img_size,
    gamma,
    bayer_pattern,
):
    """Execute the SE section 7 formula on Torch tensors.

    This shared core keeps every temporary tensor on the input device.  The
    explicit float32 casts are required by SE section 5.5: the operator's
    uint16-to-float computation, log/exp chain, and rounding input all use
    float32 even when a framework caller supplies wider metadata.
    """
    h_out = int(img_size[0].item())
    w_out = int(img_size[1].item())
    h_in = h_out // 2
    w_in = w_out // 2

    processed_channels = []
    channels = (img_channel_0, img_channel_1, img_channel_2, img_channel_3)
    for channel_index, channel in enumerate(channels):
        channel_valid = channel[:h_in, :w_in].to(dtype=torch.float32)
        gamma_value = gamma[channel_index].to(
            device=channel_valid.device,
            dtype=torch.float32,
        )

        blc = channel_valid.new_tensor(56.0)
        blc_minus_one = channel_valid.new_tensor(55.0)
        scale = channel_valid.new_tensor(967.0)
        shift = channel_valid.new_tensor(64.0)

        value = channel_valid / shift
        mask = torch.clamp(value, min=blc_minus_one, max=blc) - blc_minus_one
        gamma_base = (value - blc) / scale * mask
        positive_exp = torch.exp(
            torch.log(torch.clamp(gamma_base, min=1.0e-38)) * gamma_value
        )
        # Mirror the kernel's piecewise 0 ** gamma handling; using -inf directly
        # would produce NaN when gamma is zero/negative.
        gamma_zero_exp = torch.ones_like(gamma_base)
        gamma_negative_exp = torch.full_like(gamma_base, float("inf"))
        d_exp = torch.where(
            gamma_base > 0,
            positive_exp,
            torch.where(
                gamma_value > 0,
                torch.zeros_like(gamma_base),
                torch.where(gamma_value == 0, gamma_zero_exp, gamma_negative_exp),
            ),
        )
        corrected = torch.where(
            mask > 0,
            (d_exp * scale + blc) * mask,
            torch.zeros_like(mask),
        )
        result = value * (value.new_tensor(1.0) - mask) + corrected
        processed_channels.append(
            torch.round(result).clamp(0, 65535).to(dtype=torch.int32)
        )

    ch0, ch1, ch2, ch3 = processed_channels
    raw_img = ch0.new_zeros((h_out, w_out), dtype=torch.int32)

    if bayer_pattern == "binning":
        raw_img[0::2, 0::2] = ch0
        raw_img[0::2, 1::2] = ch1
        raw_img[1::2, 0::2] = ch2
        raw_img[1::2, 1::2] = ch3
    elif bayer_pattern == "quad":
        raw_img[0::4, 0::2] = ch0[0::2, :]
        raw_img[0::4, 1::2] = ch1[0::2, :]
        raw_img[1::4, 0::2] = ch0[1::2, :]
        raw_img[1::4, 1::2] = ch1[1::2, :]
        raw_img[2::4, 0::2] = ch2[0::2, :]
        raw_img[2::4, 1::2] = ch3[0::2, :]
        raw_img[3::4, 0::2] = ch2[1::2, :]
        raw_img[3::4, 1::2] = ch3[1::2, :]
    else:
        raise ValueError(
            f"bayer_pattern must be either 'binning' or 'quad', got {bayer_pattern!r}"
        )

    return raw_img.to(dtype=torch.uint16)


class ImgRawDecodePostHandleTestSpec:
    """Kernel/GEIR TestSpec with an independently dispatched Torch provider."""

    def golden(
        img_channel_0,
        img_channel_1,
        img_channel_2,
        img_channel_3,
        img_size,
        gamma,
        *,
        bayer_pattern="binning",
        **kwargs,
    ):
        # uint16 -> int32: older torch rejects np.uint16 in from_numpy; values
        # fit int32 exactly and the core casts to float32 anyway.
        torch_inputs = tuple(
            torch.from_numpy(np.ascontiguousarray(channel.astype(np.int32)))
            for channel in (img_channel_0, img_channel_1, img_channel_2, img_channel_3)
        ) + (
            torch.from_numpy(np.ascontiguousarray(img_size)),
            torch.from_numpy(np.ascontiguousarray(gamma)),
        )
        result = _img_raw_decode_post_handle_compute(
            *torch_inputs,
            bayer_pattern=bayer_pattern,
        )
        return [result.detach().cpu().numpy()]

    class ThirdPartyImpl:
        def __init__(self, bayer_pattern="binning", **kwargs):
            self.bayer_pattern = bayer_pattern

        def __call__(
            self,
            img_channel_0,
            img_channel_1,
            img_channel_2,
            img_channel_3,
            img_size,
            gamma,
            **kwargs,
        ):
            result = _img_raw_decode_post_handle_compute(
                img_channel_0,
                img_channel_1,
                img_channel_2,
                img_channel_3,
                img_size,
                gamma,
                bayer_pattern=self.bayer_pattern,
            )
            return [result]

    # Explicit provider binding is required for portable remote GEIR dispatch.
    third_party = {"torch": ThirdPartyImpl}

    # TTK's native quant standard only accepts int4/int8 outputs.  raw_img is
    # uint16, so keep the honest binary_equal route and override it with this
    # custom quant-style comparator using the operator's documented atol=2.
    tolerance = {
        "uint16": {"standard": "binary_equal"},
    }

    @staticmethod
    def compare(raw_img, golden_raw_img, **kwargs):
        """Apply TTK quant-style integer accounting with uint16 atol=2.

        TTK quant integerizes to int32, counts ``abs(output - golden) > 1``
        and compares that ratio with ``ptol``.  This operator is not eligible
        for native quant because its output is uint16; its SE/CSV contract
        instead allows two LSBs, while still requiring no element beyond the
        limit (ptol=0).
        """
        output = np.asarray(raw_img)
        golden = np.asarray(golden_raw_img)
        if output.shape != golden.shape:
            return {
                "pass": False,
                "precision": 0.0,
                "error_info": f"shape mismatch: output={output.shape}, golden={golden.shape}",
                "metrics": {
                    "standard": "quant_style",
                    "abs_err_limit": 2,
                    "exceed_count": int(golden.size),
                    "ptol": 0.0,
                    "pass": False,
                },
            }

        output_i = output.astype(np.int32, copy=False)
        golden_i = golden.astype(np.int32, copy=False)
        diff = np.abs(output_i - golden_i)
        bad = np.flatnonzero(diff.reshape(-1) > 2)
        size = int(golden_i.size)
        exceed_count = int(bad.size)
        precision_fraction = (size - exceed_count) / size if size else 1.0
        ptol = 0.0
        passed = (1.0 - precision_fraction) <= ptol
        max_abs_diff = int(np.max(diff)) if diff.size else 0
        return {
            "pass": bool(passed),
            "precision": float(precision_fraction * 100.0),
            "error_info": (
                ""
                if passed
                else f"integer absolute error exceeds 2; count={exceed_count}, max={max_abs_diff}"
            ),
            "diff_indices": bad[:64].tolist(),
            "metrics": {
                "standard": "quant_style",
                "precision": f"{precision_fraction * 100.0}%",
                "abs_err_limit": 2,
                "exceed_count": exceed_count,
                "ptol": ptol,
                "max_abs_diff": max_abs_diff,
                "pass": bool(passed),
            },
        }
