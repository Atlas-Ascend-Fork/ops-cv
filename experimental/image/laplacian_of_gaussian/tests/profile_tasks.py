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
"""Native msprof task timing for ACLNN executables and Python NPU workers."""

import csv
import json
import subprocess
import statistics
from pathlib import Path


def collect(cmd, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    command = [
        "msprof",
        "--output=" + str(directory),
        "--ai-core=on",
        "--task-time=on",
        "--ascendcl=on",
    ] + list(map(str, cmd))
    p = subprocess.run(command, capture_output=True, text=True, timeout=300)
    (directory / "application.log").write_text(p.stdout + p.stderr)
    (directory / "command.json").write_text(json.dumps(command, indent=2))
    if p.returncode or "Traceback (most recent call last)" in p.stdout + p.stderr:
        raise RuntimeError(f"msprof application failed: {directory}/application.log")
    return directory


def summarize(directory, repeats, custom=False, terminal="add"):
    files = list(Path(directory).rglob("op_summary_*.csv"))
    if len(files) != 1:
        raise RuntimeError(f"expected one op_summary CSV, got {files}")
    with files[0].open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["Op Type"] = row.get("OP Type", row.get("Op Type", ""))
    rows.sort(key=lambda r: float(r.get("Task Start Time(us)", "0").strip()))
    # op_summary contains device tasks; memory copies and event records are excluded.
    rows = [
        r
        for r in rows
        if r.get("Task Type", "").upper()
        in ("AI_CORE", "AI_VECTOR_CORE", "AI_CPU", "MIX_AIC", "MIX_AIV", "AI_MIX_CORE")
    ]
    if not rows:
        raise RuntimeError(f"no recognized device task rows in {files[0]}")
    timekey = "Task Duration(us)"
    if timekey not in rows[0]:
        raise RuntimeError(f"missing {timekey}: {list(rows[0])}")
    if custom:
        rows = [
            r
            for r in rows
            if "laplacian"
            in (r.get("Op Type", "") + " " + r.get("Op Name", "")).lower()
        ]
        if len(rows) < repeats:
            raise RuntimeError("missing custom LoG tasks")
        groups = [[r] for r in rows[-repeats:]]
    else:
        # Every complete FIR invocation terminates in a single Add kernel.
        # Keep exactly the final repeats invocations, after precision and warmup.
        ends = [
            i
            for i, r in enumerate(rows)
            if (
                r.get("Op Type", "").lower() in ("add", "addv2")
                if terminal == "add"
                else "conv" in r.get("Op Type", "").lower()
            )
        ]
        if len(ends) < repeats + 1:
            raise RuntimeError(
                f"missing FIR terminal Add tasks in {files[0]}: {[r.get('Op Type') for r in rows]}"
            )
        # Conv2D may have an output TransData kernel after it. Infer the
        # complete stationary period and align to the final device task,
        # thereby including every border/cast/format/compute/output task.
        period = ends[-1] - ends[-2]
        if period < 1 or len(rows) < period * repeats:
            raise RuntimeError("incomplete FIR task periods")
        groups = [
            rows[
                len(rows) - period * (repeats - j) : len(rows)
                - period * (repeats - j - 1)
                if j < repeats - 1
                else len(rows)
            ]
            for j in range(repeats)
        ]
        signatures = [[r.get("Op Type") for r in group] for group in groups]
        if any(s != signatures[0] for s in signatures):
            raise RuntimeError("inconsistent FIR task sequence")
    samples = [sum(float(r[timekey]) for r in group) for group in groups]
    if not all(v > 0 for v in samples):
        raise RuntimeError("nonpositive kernel duration")
    out = dict(
        scope="sum_device_kernel_us_complete_log",
        samples_us=samples,
        median_us=statistics.median(samples),
        min_us=min(samples),
        max_us=max(samples),
        tasks_per_call=len(groups[0]),
        task_types=[r.get("Op Type") for r in groups[0]],
        csv=str(files[0]),
        raw_rows=groups,
    )
    (Path(directory) / "kernel_summary.json").write_text(json.dumps(out, indent=2))
    return out
