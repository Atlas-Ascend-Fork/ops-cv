# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# NOTE: Portions of this code were AI-generated and have been
# technically reviewed for functional accuracy and security
# ----------------------------------------------------------------------------------------------------------
"""Same-definition FP32 FIR baselines: full 2D and separable public NPU conv2d.

CPU coefficient/index construction is setup, while cast and border gathers,
convolutions and final branch addition are included in every invocation.
"""

import json
import math
import os
import sys
import subprocess
from pathlib import Path
import numpy as np


def worker(source, kind, destination):
    import torch
    import torch_npu
    import torch.nn.functional as F

    z = np.load(source)
    x = z["x"]
    s = float(z["sigma"])
    r = int(z["radius"])
    border = str(z["border"])
    warmup = int(z["warmup"])
    repeats = int(z["repeats"])
    torch.npu.set_device(0)
    torch.npu.conv.allow_hf32 = False
    torch.npu.matmul.allow_hf32 = False
    src = torch.from_numpy(x.copy()).to("npu:0").reshape(-1, 1, *x.shape[-2:])
    taps = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-0.5 * (taps / s) ** 2)
    g /= g.sum()
    d = (taps * taps / s**4 - 1 / s**2) * g

    def weight(a):
        return torch.tensor(a.astype(np.float32), device="npu:0")

    wg = weight(g)
    wd = weight(d)
    kernel = weight(d[:, None] * g[None, :] + g[:, None] * d[None, :]).reshape(
        1, 1, 2 * r + 1, 2 * r + 1
    )
    index = []
    masks = []
    for n in x.shape[-2:]:
        raw = np.arange(-r, n + r)
        valid = (raw >= 0) & (raw < n)
        if border == "REFLECT":
            q = raw % (2 * n)
            mapped = np.where(q < n, q, 2 * n - 1 - q)
        elif border == "REFLECT_101":
            if n == 1:
                mapped = np.zeros_like(raw)
            else:
                q = raw % (2 * n - 2)
                mapped = np.where(q < n, q, 2 * n - 2 - q)
        else:
            mapped = np.clip(raw, 0, n - 1)
        index.append(torch.tensor(mapped, device="npu:0"))
        masks.append(weight(valid))

    def pad(v, axis):
        result = v.index_select(axis, index[axis + 2])
        if border == "CONSTANT":
            result = result * masks[axis + 2].reshape(
                (1, 1, -1, 1) if axis == -2 else (1, 1, 1, -1)
            )
        return result

    def call():
        v = src.float()
        if kind == "full2d":
            return F.conv2d(pad(pad(v, -2), -1), kernel)
        padded = pad(v, -2)
        a = F.conv2d(padded, wd.reshape(1, 1, -1, 1))
        b = F.conv2d(padded, wg.reshape(1, 1, -1, 1))
        return F.conv2d(pad(a, -1), wg.reshape(1, 1, 1, -1)) + F.conv2d(
            pad(b, -1), wd.reshape(1, 1, 1, -1)
        )

    # Validate this actual path before measuring. Every output is FP32.
    out = call()
    torch.npu.synchronize()
    actual = out.cpu().numpy().reshape(x.shape)
    checks = []
    for ref in (z["golden"], z["independent"]):
        e = np.abs(actual.astype(np.float64) - ref)
        b = 2**-16 + 2**-10 * np.abs(ref)
        checks.append(
            dict(
                matched_ratio=float(np.mean(e <= b)),
                max_abs=float(e.max()),
                rmse=float(np.sqrt(np.mean(e * e))),
                rmse_bound=float(np.sqrt(np.mean(b * b))),
            )
        )
    passed = (
        actual.dtype == np.float32
        and np.isfinite(actual).all()
        and all(
            c["matched_ratio"] == 1
            and c["max_abs"] <= 0.01
            and c["rmse"] <= c["rmse_bound"]
            for c in checks
        )
    )
    result = dict(
        implementation=kind,
        precision_pass=bool(passed),
        precision_checks=checks,
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        allow_hf32=False,
        device_physical=4,
        scope="device_event_resident_full_log_ms",
    )
    if passed:
        for _ in range(warmup):
            out = call()
            torch.npu.synchronize()
        samples = []
        for _ in range(repeats):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            out = call()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        result["samples_ms"] = samples
    np.save(str(destination) + ".npy", actual)
    Path(str(destination) + ".json").write_text(json.dumps(result, indent=2))


def benchmark(x, sigma, radius, truncate, border, golden, independent, warmup, repeats):
    from profile_tasks import collect, summarize

    s = float(np.float32(sigma))
    r = (
        radius
        if radius >= 0
        else int(math.floor(s * float(np.float32(truncate)) + 0.5))
    )
    directory = Path(os.environ["LOG_OUTPUT"])
    tag = os.environ["LOG_CASE_ID"]
    source = directory / f"0-{tag}.baseline-input.npz"
    np.savez(
        source,
        x=x,
        sigma=s,
        radius=r,
        border=border,
        warmup=warmup,
        repeats=repeats,
        golden=golden,
        independent=independent,
    )
    candidates = []
    for kind in ("full2d", "separable"):
        dst = directory / f"0-{tag}.{kind}"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            str(source),
            kind,
            str(dst),
        ]
        env = os.environ.copy()
        env.pop("LOG_BASELINE", None)
        if os.environ.get("LOG_PROFILE") == "1":
            collect(cmd, str(dst) + "-prof")
        else:
            p = subprocess.run(
                cmd, env=env, capture_output=True, text=True, timeout=240
            )
            Path(str(dst) + ".log").write_text(p.stdout + p.stderr)
            if p.returncode:
                raise RuntimeError(f"NPU baseline {kind} failed: {dst}.log")
        if not Path(str(dst) + ".json").exists():
            raise RuntimeError(f"missing baseline result: {dst}")
        result = json.loads(Path(str(dst) + ".json").read_text())
        result["output_path"] = str(dst) + ".npy"
        if result["precision_pass"] and os.environ.get("LOG_PROFILE") == "1":
            result["kernel"] = summarize(
                str(dst) + "-prof",
                repeats,
                terminal="conv" if kind == "full2d" else "add",
            )
        candidates.append(result)
    qualified = [v for v in candidates if v["precision_pass"]]
    if not qualified:
        raise RuntimeError(
            f"neither NPU conv baseline passes frozen precision: {candidates}"
        )
    selected = min(
        qualified,
        key=lambda v: v["kernel"]["median_us"]
        if "kernel" in v
        else float(np.median(v["samples_ms"])),
    )
    result = selected.copy()
    result["candidates"] = candidates
    result["output"] = np.load(selected["output_path"])
    return result


if __name__ == "__main__":
    worker(*sys.argv[1:])
