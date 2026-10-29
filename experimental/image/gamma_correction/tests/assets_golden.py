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
"""Reusable NumPy reference for GammaCorrection.

Thin wrapper over tests/ut/op_kernel/golden.py so callers that only import the
shared assets module get the same mode-aware reference. See golden.golden() for
the per-mode math.
"""

import numpy as np

# Re-export the canonical implementation to avoid drift between the two copies.
from golden import golden as _kernel_golden  # noqa: E402  (path added by test runner)


def gamma_correction_golden(x, gamma, correction_mode=0, max_value=0.0):
    return _kernel_golden(
        x, gamma, correction_mode=correction_mode, max_value=max_value
    )


if __name__ == "__main__":
    values = np.array([0.0, 0.01, 0.25, 0.5, 1.0], dtype=np.float16)
    result = gamma_correction_golden(values, 2.2)
    assert result.shape == values.shape
    assert np.isfinite(result).all()
    print("golden self-check OK", result)
