# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate independent NumPy float64 golden arrays in the build directory."""

import sys
from pathlib import Path

import numpy as np


def convolve_axis(image, sigma, axis):
    radius = int(3 * sigma)
    taps = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-(taps * taps) / (2 * sigma * sigma))
    weights /= weights.sum()
    # NumPy symmetric padding repeats the edge, including when radius > dimension.
    padding = [(0, 0)] * image.ndim
    padding[axis] = (radius, radius)
    padded = np.pad(image, padding, mode="symmetric")
    result = np.zeros_like(image)
    for index, weight in enumerate(weights):
        slices = [slice(None)] * image.ndim
        slices[axis] = slice(index, index + image.shape[axis])
        result += weight * padded[tuple(slices)]
    return result


def reference(image):
    clipped = np.clip(image.astype(np.float64), 0, 255)
    x = clipped + 1
    msr = np.zeros_like(x)
    for sigma in (15, 80, 250):
        illumination = convolve_axis(convolve_axis(x, sigma, 1), sigma, 0)
        msr += (np.log(x) - np.log(illumination)) / 3
    if image.shape[2] == 1:
        return np.clip(30 * (msr + 6), 0, 255).astype(np.float32)
    crf = np.log(125 * x[:, :, :3]) - np.log(x[:, :, :3].sum(axis=2, keepdims=True))
    result = clipped.copy()
    result[:, :, :3] = np.clip(30 * (crf * msr[:, :, :3] + 6), 0, 255)
    return result.astype(np.float32)


def emit_array(name, data):
    values = ", ".join(f"{float(value):.9e}f" for value in data.ravel())
    return f"inline constexpr float {name}[] = {{{values}}};\n"


def generate_header():
    text = "#pragma once\nnamespace RetinexGolden {\n"
    cases = ((2, 3, 1), (3, 5, 3), (2, 33, 4))
    for height, width, channels in cases:
        count = height * width * channels
        image = (((np.arange(count, dtype=np.int64) * 73) % 331) - 35).astype(
            np.float32
        )
        image = image.reshape(height, width, channels)
        golden = reference(image)
        if not np.isfinite(golden).all() or not ((golden >= 0) & (golden <= 255)).all():
            raise ValueError("Golden violates output range")
        if channels == 4:
            np.testing.assert_array_equal(
                golden[:, :, 3], np.clip(image[:, :, 3], 0, 255)
            )
        text += emit_array(f"input{channels}", image)
        text += emit_array(f"output{channels}", golden)
    return text + "} // namespace RetinexGolden\n"


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(generate_header(), encoding="utf-8")
