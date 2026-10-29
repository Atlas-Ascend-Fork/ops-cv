# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# Copyright (c) 2026.
"""NumPy golden for GammaCorrection kernel tests.

Implements the three correctionMode pipelines so the host reference mirrors the
device kernel exactly:

  mode 0 (pure):  y = pow(x, gamma)               # NaN/Inf propagate
  mode 1 (clamp): t = clip(x, 0, M); y = clip(pow(t/M, gamma) * M, 0, M)
  mode 2 (sRGB):  t = clip(x, 0, M) / M; sRGB(t) * M; clip to [0, M]

where M = maxValue when provided (>0), else derived from the dtype
(1.0 for floats, 255.0 for 8-bit, 65535.0 for 16-bit integers).

Integer outputs follow the kernel's clamp-before-cast: NaN -> 0, saturate to the
dtype range, then round half to even (CAST_RINT). This applies to every mode,
including mode 0.
"""

import numpy as np

_PURE = 0
_CLAMP = 1
_SRGB = 2


_INT_RANGE = {
    np.dtype(np.uint8): (0.0, 255.0),
    np.dtype(np.int8): (-128.0, 127.0),
    np.dtype(np.uint16): (0.0, 65535.0),
    np.dtype(np.int16): (-32768.0, 32767.0),
}


def _default_max_value(dtype):
    if dtype == np.uint8 or dtype == np.int8:
        return 255.0
    if dtype == np.uint16 or dtype == np.int16:
        return 65535.0
    return 1.0


def _srgb_encode(t):
    """linear -> sRGB encoding curve on a [0,1] array."""
    t = np.asarray(t, dtype=np.float32)
    linear = 12.92 * t
    branch = 1.055 * np.power(t, 1.0 / 2.4) - 0.055
    return np.where(t <= 0.0031308, linear, branch)


def golden(x, gamma, correction_mode=_PURE, max_value=0.0):
    dtype = np.asarray(x).dtype
    max_value = float(max_value)
    if not (np.isfinite(max_value) and max_value > 0.0):
        max_value = _default_max_value(dtype)

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        xf = x.astype(np.float32)
        g = np.float32(gamma)

        if correction_mode == _PURE:
            result = np.power(xf, g)
        else:
            t = np.clip(xf, 0.0, max_value)
            if correction_mode == _CLAMP:
                p = np.power(t / max_value, g)
                p = np.where(np.isfinite(p), p, 0.0)  # NaN/Inf -> 0
                y = p * max_value
            else:  # _SRGB
                y = _srgb_encode(t / max_value) * max_value
            result = np.clip(np.where(np.isfinite(y), y, 0.0), 0.0, max_value)

        int_range = _INT_RANGE.get(np.dtype(dtype))
        if int_range is not None:
            result = np.where(np.isnan(result), 0.0, result)
            result = np.rint(np.clip(result, int_range[0], int_range[1]))

    return result.astype(dtype, copy=False)


if __name__ == "__main__":
    x = np.linspace(0.0, 1.0, 1031, dtype=np.float16)
    y = golden(x, 2.2)
    assert y.shape == x.shape
    assert np.isfinite(y).all()
    u16 = np.array([0, 32767, 32768, 65535], dtype=np.uint16)
    assert (golden(u16, 1.0) == u16).all()
    assert (golden(np.array([255], dtype=np.uint8), 2.2) == 255).all()
    assert (golden(np.array([-100], dtype=np.int16), 2.2) == 0).all()
    print("golden self-check OK", y.shape)
