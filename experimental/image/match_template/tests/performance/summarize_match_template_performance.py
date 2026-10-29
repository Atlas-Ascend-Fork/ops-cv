#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Summarize three-run MatchTemplate ACLNN and torch_npu benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = {row["case"]: row for row in csv.DictReader(stream)}
    if not rows:
        raise ValueError(f"no benchmark rows in {path}")
    return rows


def median_metric(
    runs: list[dict[str, dict[str, str]]], case: str, metric: str
) -> float:
    return statistics.median(float(run[case][metric]) for run in runs)


def output_windows(image: str, template: str) -> int:
    batch, image_h, image_w, _ = (int(value) for value in image.split("x"))
    template_h, template_w = (int(value) for value in template.split("x"))
    return batch * (image_h - template_h + 1) * (image_w - template_w + 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory containing custom_run1..3.csv and torch_run1..3.csv",
    )
    parser.add_argument(
        "--output", type=Path, help="Output directory; defaults to --input"
    )
    parser.add_argument("--gate", type=float, default=1.05)
    args = parser.parse_args()
    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)

    custom_runs = [
        read_rows(args.input / f"custom_run{run}.csv") for run in range(1, 4)
    ]
    torch_runs = [read_rows(args.input / f"torch_run{run}.csv") for run in range(1, 4)]
    expected_cases = set(custom_runs[0])
    for name, runs in (("custom", custom_runs), ("torch", torch_runs)):
        for index, run in enumerate(runs, start=1):
            if set(run) != expected_cases:
                missing = sorted(expected_cases - set(run))
                extra = sorted(set(run) - expected_cases)
                raise ValueError(
                    f"{name} run {index} case mismatch: missing={missing}, extra={extra}"
                )

    rows: list[dict[str, object]] = []
    for case in custom_runs[0]:
        custom = custom_runs[0][case]
        torch = torch_runs[0][case]
        for field in ("dtype", "image", "template", "method"):
            if custom[field] != torch[field]:
                raise ValueError(f"{case}: custom/torch {field} mismatch")
        custom_host = median_metric(custom_runs, case, "host_p50_us")
        custom_device = median_metric(custom_runs, case, "device_p50_us")
        torch_host = median_metric(torch_runs, case, "host_p50_us")
        torch_device = median_metric(torch_runs, case, "device_p50_us")
        host_speedup = torch_host / custom_host
        device_speedup = torch_device / custom_device
        windows = output_windows(custom["image"], custom["template"])
        passed = host_speedup >= args.gate and device_speedup >= args.gate
        rows.append(
            {
                "case": case,
                "dtype": custom["dtype"],
                "image": custom["image"],
                "template": custom["template"],
                "method": int(custom["method"]),
                "custom_host_p50_us": f"{custom_host:.3f}",
                "custom_device_p50_us": f"{custom_device:.3f}",
                "torch_host_p50_us": f"{torch_host:.3f}",
                "torch_device_p50_us": f"{torch_device:.3f}",
                "host_speedup": f"{host_speedup:.6f}",
                "device_speedup": f"{device_speedup:.6f}",
                "custom_mwindows_per_second": f"{windows / custom_device:.6f}",
                "workspace_bytes": int(custom["workspace_bytes"]),
                "gate": args.gate,
                "status": "passed" if passed else "failed",
            }
        )

    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    failed = [row for row in rows if row["status"] == "failed"]
    report = {
        "run_count": 3,
        "case_count": len(rows),
        "gate": args.gate,
        "passed": len(rows) - len(failed),
        "failed": len(failed),
        "status_by_dtype": {
            dtype: dict(Counter(row["status"] for row in rows if row["dtype"] == dtype))
            for dtype in sorted({str(row["dtype"]) for row in rows})
        },
        "failed_cases": [row["case"] for row in failed],
        "worst_device_speedups": [
            row["case"]
            for row in sorted(rows, key=lambda item: float(item["device_speedup"]))[:10]
        ],
        "worst_host_speedups": [
            row["case"]
            for row in sorted(rows, key=lambda item: float(item["host_speedup"]))[:10]
        ],
    }
    with (output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
