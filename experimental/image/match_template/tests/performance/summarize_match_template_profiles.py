#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Create machine-readable summaries from MatchTemplate msopprof reports."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


METHOD_CASES = (
    "f32_mode_sqdiff",
    "f32_mode_sqdiff_normed",
    "f32_mode_ccorr",
    "f32_mode_ccorr_normed",
    "f32_mode_ccoeff",
    "f32_mode_ccoeff_normed",
)
REPRESENTATIVE_CASES = (
    "f32_color_ccoeff_normed",
    "fp16_probe_n1_512_c3_t16_ccoeff_normed",
)


def find_csv(root: Path, name: str) -> Path:
    matches = list(root.glob(f"OPPROF_*/{name}"))
    if len(matches) != 1:
        raise ValueError(f"expected one {name} below {root}, found {len(matches)}")
    return matches[0]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def floats(rows: list[dict[str, str]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if row.get(field) not in (None, "", "NA")]


def mean(rows: list[dict[str, str]], field: str) -> float:
    values = floats(rows, field)
    return statistics.fmean(values) if values else 0.0


def total(rows: list[dict[str, str]], field: str) -> float:
    return sum(floats(rows, field))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, help="Output directory; defaults to --input"
    )
    args = parser.parse_args()
    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)

    method_rows = []
    for method, case in enumerate(METHOD_CASES):
        rows = read_csv(find_csv(args.input / f"{case}_BasicInfo", "OpBasicInfo.csv"))
        method_rows.append(
            {
                "case": case,
                "method": method,
                "task_duration_us": float(rows[0]["Task Duration(us)"]),
                "block_dim": int(rows[0]["Block Dim"]),
            }
        )

    representative_rows = []
    for case in REPRESENTATIVE_CASES:
        root = args.input / f"{case}_Roofline"
        basic = read_csv(find_csv(root, "OpBasicInfo.csv"))[0]
        arithmetic = read_csv(find_csv(root, "ArithmeticUtilization.csv"))
        memory = read_csv(find_csv(root, "Memory.csv"))
        memory_ub = read_csv(find_csv(root, "MemoryUB.csv"))
        pipe = read_csv(find_csv(root, "PipeUtilization.csv"))
        l2 = read_csv(find_csv(root, "L2Cache.csv"))
        resource = read_csv(find_csv(root, "ResourceConflictRatio.csv"))
        core_times = floats(arithmetic, "aiv_time(us)")
        core_mean = statistics.fmean(core_times)
        representative_rows.append(
            {
                "case": case,
                "task_duration_us": float(basic["Task Duration(us)"]),
                "block_dim": int(basic["Block Dim"]),
                "core_time_min_us": min(core_times),
                "core_time_max_us": max(core_times),
                "core_time_mean_us": core_mean,
                "core_time_cv": statistics.pstdev(core_times) / core_mean,
                "core_time_max_over_min": max(core_times) / min(core_times),
                "vector_ratio_mean": mean(arithmetic, "aiv_vec_ratio"),
                "vector_fp32_ratio_mean": mean(arithmetic, "aiv_vec_fp32_ratio"),
                "vector_misc_ratio_mean": mean(arithmetic, "aiv_vec_misc_ratio"),
                "mte2_ratio_mean": mean(pipe, "aiv_mte2_ratio"),
                "mte3_ratio_mean": mean(pipe, "aiv_mte3_ratio"),
                "scalar_ratio_mean": mean(pipe, "aiv_scalar_ratio"),
                "scalar_vector_stall_us_mean": mean(
                    pipe, "aiv_scalar_vector_stall_time(us)"
                ),
                "scalar_wait_us_mean": mean(pipe, "aiv_scalar_wait_time(us)"),
                "main_memory_read_bw_gbs_mean": mean(
                    memory, "aiv_main_mem_read_bw(GB/s)"
                ),
                "main_memory_write_bw_gbs_mean": mean(
                    memory, "aiv_main_mem_write_bw(GB/s)"
                ),
                "gm_to_ub_kb_total": total(memory, "GM_to_UB_datas(KB)"),
                "ub_to_gm_kb_total": total(memory, "UB_to_GM_datas(KB)"),
                "ub_vector_read_bw_gbs_mean": mean(
                    memory_ub, "aiv_ub_read_bw_vector(GB/s)"
                ),
                "ub_vector_write_bw_gbs_mean": mean(
                    memory_ub, "aiv_ub_write_bw_vector(GB/s)"
                ),
                "l2_hit_rate_mean": mean(l2, "aiv_total_hit_rate(%)"),
                "vector_conflict_ratio_mean": mean(
                    resource, "aiv_vec_total_cflt_ratio"
                ),
            }
        )

    with (output / "method_basic_info.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=method_rows[0].keys())
        writer.writeheader()
        writer.writerows(method_rows)
    with (output / "representative_profile_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=representative_rows[0].keys())
        writer.writeheader()
        writer.writerows(representative_rows)
    report = {
        "method_profiles": method_rows,
        "representative_profiles": representative_rows,
    }
    with (output / "profile_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=True)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
