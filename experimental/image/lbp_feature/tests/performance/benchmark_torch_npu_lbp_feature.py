#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""torch_npu LbpFeature reference benchmark.

Mirrors benchmark_aclnn_lbp_feature.cpp: same case names/specs and the same CSV schema,
so tests/performance/summarize_lbp_feature_performance.py can compute torch/custom speedups.

Two reference implementations are provided (chosen with --mode):
  vectorized (default): all neighbors expanded into batched index gathers / elementwise
                        calls (best-effort torch implementation, primary baseline);
  loop:                 one F.grid_sample per neighbor (natural naive implementation).

Run:  python3 benchmark_torch_npu_lbp_feature.py [case_name] [--mode vectorized|loop]
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

FLT_EPS = float(np.finfo(np.float32).eps)

# Must stay in sync with benchmark_aclnn_lbp_feature.cpp.
# (name, dtype, height, width, batch, radius, neighbors, warmup, repeats)
CASES = [
    ("u8_64x64_r1n8", "uint8", 64, 64, 1, 1, 8, 10, 100),
    ("u8_64x64_r1n4", "uint8", 64, 64, 1, 1, 4, 10, 100),
    ("u8_64x64_r2n12", "uint8", 64, 64, 1, 2, 12, 10, 100),
    ("u8_64x64_r3n16", "uint8", 64, 64, 1, 3, 16, 10, 100),
    ("u8_128x128_r1n8", "uint8", 128, 128, 1, 1, 8, 10, 100),
    ("u8_128x128_r2n16", "uint8", 128, 128, 1, 2, 16, 10, 100),
    ("u8_256x256_r1n8", "uint8", 256, 256, 1, 1, 8, 10, 100),
    ("u8_256x256_r3n24", "uint8", 256, 256, 1, 3, 24, 10, 100),
    ("u8_512x512_r1n8", "uint8", 512, 512, 1, 1, 8, 5, 20),
    ("u8_512x512_r2n12", "uint8", 512, 512, 1, 2, 12, 5, 20),
    ("u8_720x1280_r1n8", "uint8", 720, 1280, 1, 1, 8, 2, 10),
    ("u8_1080x1920_r1n8", "uint8", 1080, 1920, 1, 1, 8, 2, 10),
    ("u8_2048x2048_r1n8", "uint8", 2048, 2048, 1, 1, 8, 2, 10),
    ("u8_2048x2048_r2n16", "uint8", 2048, 2048, 1, 2, 16, 2, 10),
    ("u8_257x4096_r1n8", "uint8", 257, 4096, 1, 1, 8, 2, 10),
    ("u8_721x1283_r1n8", "uint8", 721, 1283, 1, 1, 8, 2, 10),
    ("u8_batch4_512x512_r1n8", "uint8", 512, 512, 4, 1, 8, 2, 10),
    ("u8_33x45_r1n8", "uint8", 33, 45, 1, 1, 8, 10, 100),
    ("u8_63x65_r4n16", "uint8", 63, 65, 1, 4, 16, 10, 100),
    ("u8_300x100_r2n12", "uint8", 300, 100, 1, 2, 12, 10, 100),
    ("f32_64x64_r1n8", "float32", 64, 64, 1, 1, 8, 10, 100),
    ("f32_64x64_r1n4", "float32", 64, 64, 1, 1, 4, 10, 100),
    ("f32_64x64_r2n12", "float32", 64, 64, 1, 2, 12, 10, 100),
    ("f32_64x64_r3n16", "float32", 64, 64, 1, 3, 16, 10, 100),
    ("f32_128x128_r1n8", "float32", 128, 128, 1, 1, 8, 10, 100),
    ("f32_128x128_r2n16", "float32", 128, 128, 1, 2, 16, 10, 100),
    ("f32_256x256_r1n8", "float32", 256, 256, 1, 1, 8, 10, 100),
    ("f32_256x256_r3n24", "float32", 256, 256, 1, 3, 24, 10, 100),
    ("f32_512x512_r1n8", "float32", 512, 512, 1, 1, 8, 5, 20),
    ("f32_512x512_r2n12", "float32", 512, 512, 1, 2, 12, 5, 20),
    ("f32_720x1280_r1n8", "float32", 720, 1280, 1, 1, 8, 2, 10),
    ("f32_1080x1920_r1n8", "float32", 1080, 1920, 1, 1, 8, 2, 10),
    ("f32_2048x2048_r1n8", "float32", 2048, 2048, 1, 1, 8, 2, 10),
    ("f32_2048x2048_r2n16", "float32", 2048, 2048, 1, 2, 16, 2, 10),
    ("f32_257x4096_r1n8", "float32", 257, 4096, 1, 1, 8, 2, 10),
    ("f32_721x1283_r1n8", "float32", 721, 1283, 1, 1, 8, 2, 10),
    ("f32_batch4_512x512_r1n8", "float32", 512, 512, 4, 1, 8, 2, 10),
    ("f32_33x45_r1n8", "float32", 33, 45, 1, 1, 8, 10, 100),
    ("f32_63x65_r4n16", "float32", 63, 65, 1, 4, 16, 10, 100),
    ("f32_300x100_r2n12", "float32", 300, 100, 1, 2, 12, 10, 100),
]


def neighbor_geometry(radius, neighbors):
    """整数角点 + 双线性权重（与算子 tiling 同一数学）。"""
    geometry = []
    for k in range(neighbors):
        angle = 2.0 * math.pi * k / neighbors
        x = radius * math.cos(angle)
        y = -radius * math.sin(angle)
        fx, fy = math.floor(x), math.floor(y)
        cx, cy = math.ceil(x), math.ceil(y)
        tx, ty = x - fx, y - fy
        weights = ((1.0 - tx) * (1.0 - ty), tx * (1.0 - ty), (1.0 - tx) * ty, tx * ty)
        geometry.append(
            (
                (fy, fx, weights[0]),
                (fy, cx, weights[1]),
                (cy, fx, weights[2]),
                (cy, cx, weights[3]),
            )
        )
    return geometry


def percentiles(sorted_values):
    def pick(fraction):
        index = min(
            int(np.ceil(fraction * len(sorted_values))) - 1, len(sorted_values) - 1
        )
        return sorted_values[index]

    return pick(0.5), pick(0.95)


def build_vectorized(device, height, width, batch, radius, neighbors):
    """预构建（4n*N,）int32 采样索引与（1,4n,1）权重，返回每轮计算闭包。"""
    out_h, out_w = height - 2 * radius, width - 2 * radius
    rows = (np.arange(out_h, dtype=np.int64) + radius)[:, None]
    cols = (np.arange(out_w, dtype=np.int64) + radius)[None, :]
    corner_indices = []
    weight_values = []
    shifts = []
    for k, corners in enumerate(neighbor_geometry(radius, neighbors)):
        for dy, dx, weight in corners:
            rr = np.clip(rows + dy, 0, height - 1)
            cc = np.clip(cols + dx, 0, width - 1)
            corner_indices.append((rr * width + cc).reshape(-1))
            weight_values.append(weight)
        shifts.append(1 << k)
    flat_index = torch.from_numpy(
        np.ascontiguousarray(np.concatenate(corner_indices)).astype(np.int32)
    ).to(device)
    weights = torch.tensor(weight_values, dtype=torch.float32, device=device).reshape(
        1, -1, 1
    )
    shift_values = torch.tensor(shifts, dtype=torch.int32, device=device).reshape(
        1, -1, 1
    )
    center_index = torch.from_numpy(
        np.ascontiguousarray(
            (
                (np.arange(out_h, dtype=np.int64) + radius)[:, None] * width
                + (np.arange(out_w, dtype=np.int64) + radius)[None, :]
            ).reshape(-1)
        ).astype(np.int32)
    ).to(device)

    def compute(x):
        xf = x.reshape(batch, height * width).float()
        samples = torch.index_select(xf, 1, flat_index).reshape(
            batch, 4 * neighbors, -1
        )
        neighbor = (samples * weights).reshape(batch, neighbors, 4, -1).sum(dim=2)
        center = torch.index_select(xf, 1, center_index).unsqueeze(1)  # (B, 1, N)
        bits = ((neighbor > center) | ((neighbor - center).abs() < FLT_EPS)).to(
            torch.int32
        )
        return (bits * shift_values).sum(dim=1)  # (B, N) int32

    return compute


def build_loop(device, height, width, batch, radius, neighbors):
    """逐邻居 grid_sample 版（自然写法）。"""
    out_h, out_w = height - 2 * radius, width - 2 * radius
    grids = []
    for k in range(neighbors):
        angle = 2.0 * math.pi * k / neighbors
        ys = np.arange(out_h)[:, None] + radius - radius * math.sin(angle)
        xs = np.arange(out_w)[None, :] + radius + radius * math.cos(angle)
        gx = np.broadcast_to(xs * 2.0 / max(width - 1, 1) - 1.0, (out_h, out_w))
        gy = np.broadcast_to(ys * 2.0 / max(height - 1, 1) - 1.0, (out_h, out_w))
        grids.append(torch.from_numpy(np.stack([gx, gy], axis=-1).astype(np.float32)))
    grid_tensors = [g.to(device) for g in grids]

    def compute(x):
        xf = x.float()
        base = xf[None, None] if xf.dim() == 2 else xf[:, None]
        center = base[:, :, radius : radius + out_h, radius : radius + out_w]
        out = torch.zeros((batch, out_h, out_w), dtype=torch.int32, device=x.device)
        for k, grid in enumerate(grid_tensors):
            neighbor = F.grid_sample(
                base,
                grid.unsqueeze(0).repeat(batch, 1, 1, 1),
                mode="bilinear",
                align_corners=True,
                padding_mode="border",
            )
            neighbor = neighbor[:, 0, :, :]
            bits = (
                (neighbor > center[:, 0]) | ((neighbor - center[:, 0]).abs() < FLT_EPS)
            ).to(torch.int32)
            out += bits * (1 << k)
        return out.reshape(batch, -1)

    return compute


def time_compute(compute, x, warmup, repeats):
    for _ in range(warmup):
        compute(x)
    torch.npu.synchronize()
    host_samples, device_samples = [], []
    for _ in range(repeats):
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        host_start = time.perf_counter()
        start_event.record()
        compute(x)
        end_event.record()
        torch.npu.synchronize()
        host_samples.append((time.perf_counter() - host_start) * 1e6)
        device_samples.append(start_event.elapsed_time(end_event) * 1000.0)
    host_samples.sort()
    device_samples.sort()
    return host_samples, device_samples


def run_case(
    name, dtype, height, width, batch, radius, neighbors, warmup, repeats, mode
):
    device = torch.device("npu:0")
    shape = (batch, height, width) if batch > 1 else (height, width)
    rng = np.random.default_rng(0)
    host = (
        rng.integers(0, 256, shape, dtype=np.uint8)
        if dtype == "uint8"
        else (rng.random(shape) * 256).astype(np.float32)
    )
    x = torch.from_numpy(np.ascontiguousarray(host)).to(device)

    if mode == "best":
        # 每例取“循环 grid_sample 版”与“向量化 gather 版”中更快者作为对标基线，
        # 保证对照的是最强的合理 torch 实现（避免挑弱实现带来的口径争议）。
        best = None
        for candidate in ("vectorized", "loop"):
            builder = build_vectorized if candidate == "vectorized" else build_loop
            compute = builder(device, height, width, batch, radius, neighbors)
            host_samples, device_samples = time_compute(compute, x, warmup, repeats)
            if (
                best is None
                or device_samples[len(device_samples) // 2] < best[2][len(best[2]) // 2]
            ):
                best = (candidate, host_samples, device_samples)
        host_samples, device_samples = best[1], best[2]
    else:
        builder = build_vectorized if mode == "vectorized" else build_loop
        compute = builder(device, height, width, batch, radius, neighbors)
        host_samples, device_samples = time_compute(compute, x, warmup, repeats)

    host_p50, host_p95 = percentiles(host_samples)
    device_p50, device_p95 = percentiles(device_samples)
    print(
        "%s,%s,%dx%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d"
        % (
            name,
            dtype,
            height,
            width,
            radius,
            neighbors,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("case", nargs="?", default="", help="run a single case by name")
    parser.add_argument(
        "--mode", choices=("best", "vectorized", "loop"), default="best"
    )
    args = parser.parse_args()

    print(
        "case,dtype,shape,radius,neighbors,repeats,host_min_us,host_p50_us,"
        "host_p95_us,host_mean_us,device_min_us,device_p50_us,device_p95_us,"
        "device_mean_us,workspace_bytes"
    )
    found = args.case == ""
    for case in CASES:
        if args.case and args.case != case[0]:
            continue
        found = True
        run_case(*case, mode=args.mode)
    if not found:
        print(f"unknown benchmark case: {args.case}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
