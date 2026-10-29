#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Summarize three performance runs while retaining runtime-failed cases."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return {row["case"]: row for row in csv.DictReader(stream)}


def median(runs: list[dict[str, dict[str, str]]], case: str, metric: str) -> float:
    return statistics.median(float(run[case][metric]) for run in runs)


def output_windows(image: str, template: str) -> int:
    batch, image_h, image_w, _ = (int(value) for value in image.split("x"))
    template_h, template_w = (int(value) for value in template.split("x"))
    return batch * (image_h - template_h + 1) * (image_w - template_w + 1)


def read_failures(input_directory: Path) -> dict[str, set[str]]:
    failures: dict[str, set[str]] = {}
    for run in range(1, 4):
        with (input_directory / f"custom_failures_r{run}.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            for row in csv.DictReader(stream):
                failures.setdefault(row["case"], set()).add(row["error"])
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", type=float, default=1.05)
    args = parser.parse_args()
    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)

    with args.matrix.open(newline="", encoding="utf-8-sig") as stream:
        matrix = list(csv.DictReader(stream))
    custom_runs = [read_rows(args.input / f"custom_r{run}.csv") for run in range(1, 4)]
    torch_runs = [read_rows(args.input / f"torch_r{run}.csv") for run in range(1, 4)]
    failures = read_failures(args.input)

    rows: list[dict[str, object]] = []
    for expected in matrix:
        case = expected["case"]
        common = {
            "case": case,
            "dtype": expected["dtype"],
            "image": expected["image"],
            "template": expected["template"],
            "method": int(expected["method"]),
        }
        if any(case not in run for run in custom_runs):
            errors = sorted(failures.get(case, {"missing custom measurement"}))
            rows.append(
                {
                    **common,
                    "custom_host_p50_us": "",
                    "custom_device_p50_us": "",
                    "torch_host_p50_us": f"{median(torch_runs, case, 'host_p50_us'):.3f}",
                    "torch_device_p50_us": f"{median(torch_runs, case, 'device_p50_us'):.3f}",
                    "host_speedup": "",
                    "device_speedup": "",
                    "custom_mwindows_per_second": "",
                    "workspace_bytes": "",
                    "gate": args.gate,
                    "status": "runtime_failed",
                    "error": " | ".join(errors),
                }
            )
            continue

        custom = custom_runs[0][case]
        torch = torch_runs[0][case]
        for field in ("dtype", "image", "template", "method"):
            if custom[field] != torch[field]:
                raise ValueError(f"{case}: custom/torch {field} mismatch")
        custom_host = median(custom_runs, case, "host_p50_us")
        custom_device = median(custom_runs, case, "device_p50_us")
        torch_host = median(torch_runs, case, "host_p50_us")
        torch_device = median(torch_runs, case, "device_p50_us")
        host_speedup = torch_host / custom_host
        device_speedup = torch_device / custom_device
        windows = output_windows(custom["image"], custom["template"])
        passed = host_speedup >= args.gate and device_speedup >= args.gate
        rows.append(
            {
                **common,
                "custom_host_p50_us": f"{custom_host:.3f}",
                "custom_device_p50_us": f"{custom_device:.3f}",
                "torch_host_p50_us": f"{torch_host:.3f}",
                "torch_device_p50_us": f"{torch_device:.3f}",
                "host_speedup": f"{host_speedup:.6f}",
                "device_speedup": f"{device_speedup:.6f}",
                "custom_mwindows_per_second": f"{windows / custom_device:.6f}",
                "workspace_bytes": int(custom["workspace_bytes"]),
                "gate": args.gate,
                "status": "passed" if passed else "performance_failed",
                "error": "",
            }
        )

    summary_path = output / "performance_full_v18_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(str(row["status"]) for row in rows)
    report = {
        "run_count": 3,
        "case_count": len(rows),
        "gate": args.gate,
        "status": dict(counts),
        "status_by_dtype": {
            dtype: dict(
                Counter(str(row["status"]) for row in rows if row["dtype"] == dtype)
            )
            for dtype in sorted({str(row["dtype"]) for row in rows})
        },
        "nonpassing_cases": [row["case"] for row in rows if row["status"] != "passed"],
    }
    with (output / "performance_full_v18_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
