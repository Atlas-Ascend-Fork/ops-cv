#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""torch_npu Scharr reference benchmark (cat-slice border + slice multiply-add).

Mirrors benchmark_aclnn_scharr.cpp: same case names/specs and the same CSV schema,
so tests/performance/summarize_scharr_performance.py can compute torch/custom speedups.

Run:  python3 benchmark_torch_npu_scharr.py [case_name]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch_npu  # noqa: F401

# Must stay in sync with benchmark_aclnn_scharr.cpp.
CASES = [
    ("uint8_64x64_X_REFLECT", "uint8", 64, 64, 1, "X", "REFLECT_101", 10, 100),
    ("uint8_128x128_X_REFLECT", "uint8", 128, 128, 1, "X", "REFLECT_101", 10, 100),
    ("uint8_256x256_X_REFLECT", "uint8", 256, 256, 1, "X", "REFLECT_101", 10, 100),
    ("uint8_256x256_Y_CONSTANT", "uint8", 256, 256, 1, "Y", "CONSTANT", 10, 100),
    ("uint8_512x512_X_REFLECT", "uint8", 512, 512, 1, "X", "REFLECT_101", 5, 20),
    ("uint8_512x512_Y_REPLICATE", "uint8", 512, 512, 1, "Y", "REPLICATE", 5, 20),
    ("uint8_1024x1024_X_REFLECT", "uint8", 1024, 1024, 1, "X", "REFLECT_101", 3, 10),
    ("uint8_720x1280_X_REFLECT", "uint8", 720, 1280, 1, "X", "REFLECT_101", 2, 10),
    ("uint8_1080x1920_X_REFLECT", "uint8", 1080, 1920, 1, "X", "REFLECT_101", 2, 10),
    ("uint8_2048x2048_X_REFLECT", "uint8", 2048, 2048, 1, "X", "REFLECT_101", 2, 10),
    ("uint8_2048x2048_Y_REPLICATE", "uint8", 2048, 2048, 1, "Y", "REPLICATE", 2, 10),
    ("uint8_257x4096_X_REFLECT", "uint8", 257, 4096, 1, "X", "REFLECT_101", 2, 10),
    ("uint8_721x1283_Y_CONSTANT", "uint8", 721, 1283, 1, "Y", "CONSTANT", 2, 10),
    ("uint8_batch4_512x512_X_REFLECT", "uint8", 512, 512, 4, "X", "REFLECT_101", 2, 10),
    ("fp16_64x64_X_REFLECT", "float16", 64, 64, 1, "X", "REFLECT_101", 10, 100),
    ("fp16_128x128_X_REFLECT", "float16", 128, 128, 1, "X", "REFLECT_101", 10, 100),
    ("fp16_256x256_X_REFLECT", "float16", 256, 256, 1, "X", "REFLECT_101", 10, 100),
    ("fp16_256x256_Y_CONSTANT", "float16", 256, 256, 1, "Y", "CONSTANT", 10, 100),
    ("fp16_512x512_X_REFLECT", "float16", 512, 512, 1, "X", "REFLECT_101", 5, 20),
    ("fp16_512x512_Y_REPLICATE", "float16", 512, 512, 1, "Y", "REPLICATE", 5, 20),
    ("fp16_1024x1024_X_REFLECT", "float16", 1024, 1024, 1, "X", "REFLECT_101", 3, 10),
    ("fp16_720x1280_X_REFLECT", "float16", 720, 1280, 1, "X", "REFLECT_101", 2, 10),
    ("fp16_1080x1920_X_REFLECT", "float16", 1080, 1920, 1, "X", "REFLECT_101", 2, 10),
    ("fp16_2048x2048_X_REFLECT", "float16", 2048, 2048, 1, "X", "REFLECT_101", 2, 10),
    ("fp16_2048x2048_Y_REPLICATE", "float16", 2048, 2048, 1, "Y", "REPLICATE", 2, 10),
    ("fp16_257x4096_X_REFLECT", "float16", 257, 4096, 1, "X", "REFLECT_101", 2, 10),
    ("fp16_721x1283_Y_CONSTANT", "float16", 721, 1283, 1, "Y", "CONSTANT", 2, 10),
    (
        "fp16_batch4_512x512_X_REFLECT",
        "float16",
        512,
        512,
        4,
        "X",
        "REFLECT_101",
        2,
        10,
    ),
    ("f32_64x64_X_REFLECT", "float32", 64, 64, 1, "X", "REFLECT_101", 10, 100),
    ("f32_128x128_X_REFLECT", "float32", 128, 128, 1, "X", "REFLECT_101", 10, 100),
    ("f32_256x256_X_REFLECT", "float32", 256, 256, 1, "X", "REFLECT_101", 10, 100),
    ("f32_256x256_Y_CONSTANT", "float32", 256, 256, 1, "Y", "CONSTANT", 10, 100),
    ("f32_512x512_X_REFLECT", "float32", 512, 512, 1, "X", "REFLECT_101", 5, 20),
    ("f32_512x512_Y_REPLICATE", "float32", 512, 512, 1, "Y", "REPLICATE", 5, 20),
    ("f32_1024x1024_X_REFLECT", "float32", 1024, 1024, 1, "X", "REFLECT_101", 3, 10),
    ("f32_720x1280_X_REFLECT", "float32", 720, 1280, 1, "X", "REFLECT_101", 2, 10),
    ("f32_1080x1920_X_REFLECT", "float32", 1080, 1920, 1, "X", "REFLECT_101", 2, 10),
    ("f32_2048x2048_X_REFLECT", "float32", 2048, 2048, 1, "X", "REFLECT_101", 2, 10),
    ("f32_2048x2048_Y_REPLICATE", "float32", 2048, 2048, 1, "Y", "REPLICATE", 2, 10),
    ("f32_257x4096_X_REFLECT", "float32", 257, 4096, 1, "X", "REFLECT_101", 2, 10),
    ("f32_721x1283_Y_CONSTANT", "float32", 721, 1283, 1, "Y", "CONSTANT", 2, 10),
    ("f32_batch4_512x512_X_REFLECT", "float32", 512, 512, 4, "X", "REFLECT_101", 2, 10),
]
PAD_MODE = {"REFLECT_101": "reflect", "REPLICATE": "replicate", "CONSTANT": "constant"}
TORCH_DTYPE = {
    "uint8": torch.float32,
    "float16": torch.float16,
    "float32": torch.float32,
}


def percentiles(sorted_values):
    def pick(fraction):
        index = min(
            int(np.ceil(fraction * len(sorted_values))) - 1, len(sorted_values) - 1
        )
        return sorted_values[index]

    return pick(0.5), pick(0.95)


def catpad(x, mode, height, width):
    if height >= 2:
        if mode == "reflect":
            top, bottom = x[:, :, 1:2, :], x[:, :, -2:-1, :]
        elif mode == "replicate":
            top, bottom = x[:, :, 0:1, :], x[:, :, -1:, :]
        else:
            top = torch.zeros_like(x[:, :, 0:1, :])
            bottom = torch.zeros_like(top)
    else:
        top = bottom = x[:, :, 0:1, :]
    xh = torch.cat([top, x, bottom], dim=2)
    if width >= 2:
        if mode == "reflect":
            left, right = xh[:, :, :, 1:2], xh[:, :, :, -2:-1]
        elif mode == "replicate":
            left, right = xh[:, :, :, 0:1], xh[:, :, :, -1:]
        else:
            left = torch.zeros_like(xh[:, :, :, 0:1])
            right = torch.zeros_like(left)
    else:
        left = right = xh[:, :, :, 0:1]
    return torch.cat([left, xh, right], dim=3)


def run_case(name, dtype, height, width, batch, direction, border, warmup, repeats):
    device = torch.device("npu:0")
    shape = (batch, height, width) if batch > 1 else (height, width)
    rng = np.random.default_rng(0)
    host = (
        rng.integers(0, 256, shape)
        if dtype == "uint8"
        else (rng.random(shape) * 256).astype(np.float32)
    )
    x = torch.from_numpy(np.ascontiguousarray(host)).to(device)
    mode = PAD_MODE[border]
    target = TORCH_DTYPE[dtype]

    def compute():
        xf = x.to(target) if x.dtype != target else x
        if xf.dim() == 2:
            xf = xf.unsqueeze(0)
        xf = xf.unsqueeze(1)
        xp = catpad(xf, mode, height, width)
        if direction == "X":
            a = xp[:, :, 0:-2, 2:] - xp[:, :, 0:-2, 0:-2]
            b = xp[:, :, 1:-1, 2:] - xp[:, :, 1:-1, 0:-2]
            c = xp[:, :, 2:, 2:] - xp[:, :, 2:, 0:-2]
        else:
            a = xp[:, :, 2:, 0:-2] - xp[:, :, 0:-2, 0:-2]
            b = xp[:, :, 2:, 1:-1] - xp[:, :, 0:-2, 1:-1]
            c = xp[:, :, 2:, 2:] - xp[:, :, 0:-2, 2:]
        return 3.0 * a + 10.0 * b + 3.0 * c

    for _ in range(warmup):
        compute()
    torch.npu.synchronize()
    host_samples, device_samples = [], []
    for _ in range(repeats):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        host_start = time.perf_counter()
        start_event.record()
        compute()
        end_event.record()
        torch.npu.synchronize()
        host_samples.append((time.perf_counter() - host_start) * 1e6)
        device_samples.append(start_event.elapsed_time(end_event) * 1000.0)
    host_samples.sort()
    device_samples.sort()
    host_p50, host_p95 = percentiles(host_samples)
    device_p50, device_p95 = percentiles(device_samples)
    print(
        "%s,%s,%dx%d,%s,%s,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d"
        % (
            name,
            dtype,
            height,
            width,
            direction,
            border,
            repeats,
            host_samples[0],
            host_p50,
            host_p95,
            sum(host_samples) / len(host_samples),
            device_samples[0],
            device_p50,
            device_p95,
            sum(device_samples) / len(device_samples),
            0,
        )
    )


def main() -> int:
    requested = sys.argv[1] if len(sys.argv) > 1 else ""
    print(
        "case,dtype,shape,direction,border,repeats,host_min_us,host_p50_us,"
        "host_p95_us,host_mean_us,device_min_us,device_p50_us,device_p95_us,"
        "device_mean_us,workspace_bytes"
    )
    found = requested == ""
    for case in CASES:
        if requested and requested != case[0]:
            continue
        found = True
        run_case(*case)
    if not found:
        print(f"unknown benchmark case: {requested}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
