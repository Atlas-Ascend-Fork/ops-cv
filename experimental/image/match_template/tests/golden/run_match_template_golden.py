#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate deterministic MatchTemplate inputs and validate ACLNN outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


METHOD_NAMES = (
    "sqdiff",
    "sqdiff_normed",
    "ccorr",
    "ccorr_normed",
    "ccoeff",
    "ccoeff_normed",
)
CV_METHODS = (
    cv2.TM_SQDIFF,
    cv2.TM_SQDIFF_NORMED,
    cv2.TM_CCORR,
    cv2.TM_CCORR_NORMED,
    cv2.TM_CCOEFF,
    cv2.TM_CCOEFF_NORMED,
)
NUMPY_DTYPES = {
    "float32": np.float32,
    "float16": np.float16,
    "uint8": np.uint8,
}
EPSILON = np.float32(1e-12)


@dataclass(frozen=True)
class Case:
    name: str
    dtype: str
    batch: int
    image_h: int
    image_w: int
    channels: int
    template_h: int
    template_w: int
    method: int
    scenario: str
    seed: int
    force_spec: bool = False
    position: tuple[int, int] | None = None


def add_matrix(
    cases: list[Case],
    scenario: str,
    shape: tuple[int, int, int, int, int, int],
    dtypes: tuple[str, ...] = ("float32", "float16", "uint8"),
    methods: tuple[int, ...] = tuple(range(6)),
    force_spec: bool = False,
) -> None:
    batch, image_h, image_w, channels, template_h, template_w = shape
    for dtype in dtypes:
        for method in methods:
            name = f"{scenario}_{dtype}_{METHOD_NAMES[method]}"
            cases.append(
                Case(
                    name,
                    dtype,
                    batch,
                    image_h,
                    image_w,
                    channels,
                    template_h,
                    template_w,
                    method,
                    scenario,
                    1000 + len(cases),
                    force_spec,
                )
            )


def build_cases() -> tuple[Case, ...]:
    cases: list[Case] = []
    add_matrix(cases, "one_by_one", (1, 9, 11, 1, 1, 1), force_spec=True)
    add_matrix(cases, "equal_image", (1, 7, 9, 1, 7, 9))
    add_matrix(cases, "color_random", (1, 15, 18, 3, 4, 5))
    add_matrix(cases, "non_aligned", (1, 23, 29, 1, 6, 7))
    add_matrix(cases, "negative_random", (1, 17, 19, 1, 5, 7), ("float32", "float16"))
    add_matrix(cases, "batch2", (2, 13, 17, 3, 3, 4))
    add_matrix(cases, "zeros", (1, 11, 13, 1, 3, 5), methods=(1, 3, 5), force_spec=True)
    add_matrix(
        cases, "ties", (1, 12, 14, 1, 3, 4), methods=(0, 2, 4, 5), force_spec=True
    )

    positions = ((0, 0), (0, 15), (14, 0), (14, 15), (7, 8))
    for dtype in ("float32", "uint8"):
        for method in (0, 5):
            for index, position in enumerate(positions):
                name = f"planted_{index}_{dtype}_{METHOD_NAMES[method]}"
                cases.append(
                    Case(
                        name,
                        dtype,
                        1,
                        18,
                        20,
                        1,
                        4,
                        5,
                        method,
                        "planted",
                        2000 + len(cases),
                        False,
                        position,
                    )
                )

    for method in (0, 5):
        cases.append(
            Case(
                f"near_tie_float32_{METHOD_NAMES[method]}",
                "float32",
                1,
                20,
                24,
                1,
                4,
                5,
                method,
                "near_tie",
                3000 + method,
            )
        )

    add_matrix(cases, "gm_fallback", (1, 250, 250, 1, 249, 249), methods=(0, 5))
    # These shapes select the C3 vector schedule when the target UB is sufficient.
    # Small C1 ties do not exercise the indexed hardware ReduceMax in that path.
    for scenario in ("c3_constant_ties", "c3_pattern_ties"):
        add_matrix(
            cases,
            scenario,
            (1, 512, 512, 3, 16, 16),
            ("float16", "uint8"),
            (2, 3, 4, 5),
            force_spec=True,
        )
    add_matrix(
        cases,
        "c3_tail_best",
        (1, 512, 512, 3, 16, 16),
        ("float16",),
        (2,),
        force_spec=True,
    )
    return tuple(cases)


def encode(values: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "uint8":
        return np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return values.astype(NUMPY_DTYPES[dtype])


def make_inputs(case: Case) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(case.seed)
    image_shape = (case.batch, case.image_h, case.image_w, case.channels)
    template_shape = (1, case.template_h, case.template_w, case.channels)
    if case.scenario == "zeros":
        image = np.zeros(image_shape, dtype=np.float32)
        templ = np.zeros(template_shape, dtype=np.float32)
    elif case.scenario in ("ties", "c3_constant_ties"):
        image = np.full(image_shape, 2.0, dtype=np.float32)
        templ = np.full(template_shape, 2.0, dtype=np.float32)
    elif case.scenario == "c3_tail_best":
        image = np.zeros(image_shape, dtype=np.float32)
        templ = np.ones(template_shape, dtype=np.float32)
        # The unique maximum at [496,496] is in FP16 CCORR's final short
        # tile, which falls back from C3 to the generic Vector score buffer.
        image[:, -case.template_h :, -case.template_w :, :] = templ[0]
    elif case.scenario == "c3_pattern_ties":
        image = np.zeros(image_shape, dtype=np.float32)
        templ = np.zeros(template_shape, dtype=np.float32)
        # A nonconstant binary signature keeps sums exact in FP32 for both
        # FP16 and UINT8 inputs; one nonzero per row/channel makes the direct
        # independent correlation inexpensive without FFT tie-rounding noise.
        pattern_rng = np.random.default_rng(20261020)
        for y in range(case.template_h):
            for channel in range(case.channels):
                x = int(pattern_rng.integers(case.template_w))
                templ[0, y, x, channel] = 1.0
        templ[0, 0, :, 0] = 0.0
        templ[0, 0, 0, 0] = 1.0
        templ[0, -1, :, -1] = 0.0
        templ[0, -1, -1, -1] = 1.0
        # Multiple winners within one row, beyond column 256, and in another
        # tile row. The first winner is deliberately not output index zero.
        for y, x in ((5, 17), (5, 81), (5, 273), (37, 17), (37, 273)):
            image[0, y : y + case.template_h, x : x + case.template_w] = templ[0]
    elif case.scenario == "negative_random":
        image = rng.normal(0.0, 2.0, image_shape).astype(np.float32)
        templ = rng.normal(0.0, 2.0, template_shape).astype(np.float32)
    elif case.scenario == "equal_image":
        image = rng.uniform(0.0, 8.0, image_shape).astype(np.float32)
        templ = image[0:1].copy()
    elif case.scenario == "planted":
        image = rng.uniform(0.0, 4.0, image_shape).astype(np.float32)
        templ = rng.uniform(16.0, 32.0, template_shape).astype(np.float32)
        y, x = case.position or (0, 0)
        image[0, y : y + case.template_h, x : x + case.template_w, :] = templ[0]
    elif case.scenario == "near_tie":
        image = rng.uniform(-2.0, 2.0, image_shape).astype(np.float32)
        templ = rng.uniform(-4.0, 4.0, template_shape).astype(np.float32)
        image[0, 2 : 2 + case.template_h, 3 : 3 + case.template_w, :] = templ[0]
        candidate = templ[0].copy()
        candidate[0, 0, 0] += np.float32(0.125)
        image[0, 12 : 12 + case.template_h, 15 : 15 + case.template_w, :] = candidate
    elif case.dtype == "uint8":
        image = rng.integers(0, 256, image_shape, dtype=np.uint8).astype(np.float32)
        templ = rng.integers(0, 256, template_shape, dtype=np.uint8).astype(np.float32)
    else:
        image = rng.uniform(0.0, 8.0, image_shape).astype(np.float32)
        templ = rng.uniform(0.0, 8.0, template_shape).astype(np.float32)
    return encode(image, case.dtype), encode(templ, case.dtype)


def normalize(
    numerator: np.float32,
    product: np.float32,
    lower: float,
    upper: float,
    zero_value: float,
) -> np.float32:
    if product <= 0.0:
        return np.float32(zero_value)
    denominator = np.float32(np.sqrt(product))
    if denominator <= EPSILON:
        return np.float32(zero_value)
    return np.float32(np.clip(np.float32(numerator / denominator), lower, upper))


def spec_reference(image: np.ndarray, templ: np.ndarray, method: int) -> np.ndarray:
    image_f32 = image.astype(np.float32)
    template_f32 = templ[0].astype(np.float32)
    batch, image_h, image_w, _ = image_f32.shape
    template_h, template_w, _ = template_f32.shape
    output_h = image_h - template_h + 1
    output_w = image_w - template_w + 1
    spatial = np.float32(template_h * template_w)
    output = np.empty((batch, output_h, output_w), dtype=np.float32)
    template_square = np.sum(template_f32 * template_f32, dtype=np.float32)
    template_sum = np.sum(template_f32, axis=(0, 1), dtype=np.float32)
    template_channel_square = np.sum(
        template_f32 * template_f32, axis=(0, 1), dtype=np.float32
    )
    template_variance = np.sum(
        template_channel_square - template_sum * template_sum / spatial,
        dtype=np.float32,
    )
    template_variance = np.float32(max(float(template_variance), 0.0))

    for n in range(batch):
        for y in range(output_h):
            for x in range(output_w):
                window = image_f32[n, y : y + template_h, x : x + template_w, :]
                difference = window - template_f32
                square_difference = np.sum(difference * difference, dtype=np.float32)
                cross = np.sum(window * template_f32, dtype=np.float32)
                image_square = np.sum(window * window, dtype=np.float32)
                if method == 0:
                    value = square_difference
                elif method == 1:
                    zero_value = 0.0 if square_difference <= EPSILON else 1.0
                    value = normalize(
                        square_difference,
                        np.float32(image_square * template_square),
                        0.0,
                        1.0,
                        zero_value,
                    )
                elif method == 2:
                    value = cross
                elif method == 3:
                    value = normalize(
                        cross,
                        np.float32(image_square * template_square),
                        -1.0,
                        1.0,
                        0.0,
                    )
                else:
                    image_sum = np.sum(window, axis=(0, 1), dtype=np.float32)
                    image_channel_square = np.sum(
                        window * window, axis=(0, 1), dtype=np.float32
                    )
                    coefficient = np.sum(
                        np.sum(window * template_f32, axis=(0, 1), dtype=np.float32)
                        - image_sum * template_sum / spatial,
                        dtype=np.float32,
                    )
                    if method == 4:
                        value = coefficient
                    else:
                        image_variance = np.sum(
                            image_channel_square - image_sum * image_sum / spatial,
                            dtype=np.float32,
                        )
                        image_variance = np.float32(max(float(image_variance), 0.0))
                        value = normalize(
                            coefficient,
                            np.float32(image_variance * template_variance),
                            -1.0,
                            1.0,
                            0.0,
                        )
                output[n, y, x] = value
    return output


def opencv_reference(image: np.ndarray, templ: np.ndarray, method: int) -> np.ndarray:
    scores = []
    for batch_index in range(image.shape[0]):
        image_item = image[batch_index]
        template_item = templ[0]
        if image_item.shape[-1] == 1:
            image_item = image_item[..., 0]
            template_item = template_item[..., 0]
        scores.append(cv2.matchTemplate(image_item, template_item, CV_METHODS[method]))
    return np.stack(scores).astype(np.float32)


def c3_tie_reference(image: np.ndarray, templ: np.ndarray, method: int) -> np.ndarray:
    """Independent FP64 spatial formulas for the large, exact C3 tie fixtures."""
    image_f64 = image.astype(np.float64)
    template_f64 = templ[0].astype(np.float64)
    template_h, template_w, _ = template_f64.shape
    spatial = template_h * template_w
    output_h = image.shape[1] - template_h + 1
    output_w = image.shape[2] - template_w + 1

    def window_sum(values: np.ndarray) -> np.ndarray:
        integral = np.pad(values, ((0, 0), (1, 0), (1, 0), (0, 0)))
        integral = integral.cumsum(axis=1).cumsum(axis=2)
        return (
            integral[:, template_h:, template_w:]
            - integral[:, :-template_h, template_w:]
            - integral[:, template_h:, :-template_w]
            + integral[:, :-template_h, :-template_w]
        )

    image_sum = window_sum(image_f64)
    image_square = window_sum(image_f64 * image_f64).sum(axis=-1)
    template_sum = template_f64.sum(axis=(0, 1))
    template_square = np.square(template_f64).sum()
    if np.all(template_f64 == template_f64[0, 0]):
        cross = (image_sum * template_f64[0, 0]).sum(axis=-1)
    else:
        cross = np.zeros((image.shape[0], output_h, output_w), dtype=np.float64)
        for y, x, channel in np.argwhere(template_f64 != 0):
            cross += (
                image_f64[:, y : y + output_h, x : x + output_w, channel]
                * template_f64[y, x, channel]
            )
    if method == 2:
        return cross.astype(np.float32)
    if method == 3:
        numerator = cross
        product = image_square * template_square
    else:
        numerator = cross - (image_sum * template_sum).sum(axis=-1) / spatial
        if method == 4:
            return numerator.astype(np.float32)
        image_variance = np.maximum(
            image_square - np.square(image_sum).sum(axis=-1) / spatial, 0.0
        )
        template_variance = max(
            template_square - np.square(template_sum).sum() / spatial, 0.0
        )
        product = image_variance * template_variance
    denominator = np.sqrt(product)
    normalized = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > EPSILON,
    )
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def best_from_score(score: np.ndarray, method: int) -> tuple[np.ndarray, np.ndarray]:
    flat = score.reshape(score.shape[0], -1)
    indices = np.argmin(flat, axis=1) if method <= 1 else np.argmax(flat, axis=1)
    values = flat[np.arange(flat.shape[0]), indices].astype(np.float32)
    locations = np.stack(
        (indices % score.shape[2], indices // score.shape[2]), axis=1
    ).astype(np.int64)
    return values, locations


def max_abs(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(actual.astype(np.float64) - expected.astype(np.float64)), initial=0.0
        )
    )


def max_relative(actual: np.ndarray, expected: np.ndarray) -> float:
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    denominator = np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
    return float(np.max(difference / denominator, initial=0.0))


def score_passes(actual: np.ndarray, expected: np.ndarray, method: int) -> bool:
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    if method in (1, 3, 5):
        return bool(np.all(difference <= 1e-4))
    # OpenCV's CCOEFF/CCORR/SQDIFF accumulation order can differ from the
    # device's deterministic FP32 reduction for low-magnitude scores. Keep a
    # fixed absolute floor while retaining the relative precision requirement.
    expected_abs = np.abs(expected.astype(np.float64))
    tolerance = np.maximum(1e-3, expected_abs * 1e-3)
    return bool(np.all(difference <= tolerance))


def run_case(
    case: Case, runner: Path, output_root: Path, reuse_actual: bool = False
) -> dict[str, object]:
    case_root = output_root / "raw" / case.name
    case_root.mkdir(parents=True, exist_ok=True)
    image, templ = make_inputs(case)
    image_path = case_root / "image.bin"
    template_path = case_root / "template.bin"
    output_prefix = case_root / "actual"
    image.tofile(image_path)
    templ.tofile(template_path)
    command = [
        str(runner),
        case.dtype,
        str(case.batch),
        str(case.image_h),
        str(case.image_w),
        str(case.channels),
        str(case.template_h),
        str(case.template_w),
        str(case.method),
        str(image_path),
        str(template_path),
        str(output_prefix),
    ]
    output_files = (
        Path(str(output_prefix) + ".score.f32"),
        Path(str(output_prefix) + ".best_score.f32"),
        Path(str(output_prefix) + ".best_location.i64"),
    )
    if reuse_actual and all(path.is_file() for path in output_files):
        runner_return_code = 0
        runner_stderr = ""
    else:
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        runner_return_code = completed.returncode
        runner_stderr = completed.stderr.strip()
    base = {
        "case": case.name,
        "scenario": case.scenario,
        "dtype": case.dtype,
        "image_shape": f"{case.batch}x{case.image_h}x{case.image_w}x{case.channels}",
        "template_shape": f"1x{case.template_h}x{case.template_w}x{case.channels}",
        "method": case.method,
        "method_name": METHOD_NAMES[case.method],
        "runner_return_code": runner_return_code,
    }
    if runner_return_code != 0:
        return {
            **base,
            "golden_source": "not_run",
            "score_max_abs_error": math.inf,
            "score_max_relative_error": math.inf,
            "best_score_max_abs_error": math.inf,
            "location_match": False,
            "location_consistent": False,
            "best_score_consistent": False,
            "normalized_range_ok": False,
            "finite": False,
            "status": "failed",
            "runner_stderr": runner_stderr,
        }

    output_h = case.image_h - case.template_h + 1
    output_w = case.image_w - case.template_w + 1
    actual_score = np.fromfile(
        str(output_prefix) + ".score.f32", dtype=np.float32
    ).reshape(case.batch, output_h, output_w)
    actual_best = np.fromfile(str(output_prefix) + ".best_score.f32", dtype=np.float32)
    actual_location = np.fromfile(
        str(output_prefix) + ".best_location.i64", dtype=np.int64
    ).reshape(case.batch, 2)

    use_spec = case.force_spec or case.dtype == "float16"
    if case.scenario in ("c3_constant_ties", "c3_pattern_ties", "c3_tail_best"):
        golden_source = (
            "numpy_fp64_c3_tail"
            if case.scenario == "c3_tail_best"
            else "numpy_fp64_c3_ties"
        )
        expected_score = c3_tie_reference(image, templ, case.method)
    else:
        golden_source = "spec_fp32" if use_spec else f"opencv_{cv2.__version__}"
        expected_score = (
            spec_reference(image, templ, case.method)
            if use_spec
            else opencv_reference(image, templ, case.method)
        )
    expected_best, expected_location = best_from_score(expected_score, case.method)
    actual_extreme, actual_first_location = best_from_score(actual_score, case.method)
    finite = bool(np.isfinite(actual_score).all() and np.isfinite(actual_best).all())
    if case.method == 1:
        range_ok = bool(np.all((actual_score >= -1e-5) & (actual_score <= 1.0 + 1e-5)))
    elif case.method in (3, 5):
        range_ok = bool(
            np.all((actual_score >= -1.0 - 1e-5) & (actual_score <= 1.0 + 1e-5))
        )
    else:
        range_ok = True
    score_ok = score_passes(actual_score, expected_score, case.method)
    best_ok = score_passes(actual_best, expected_best, case.method)
    location_ok = bool(np.array_equal(actual_location, expected_location))
    location_consistent = bool(np.array_equal(actual_location, actual_first_location))
    best_consistent = bool(np.array_equal(actual_best, actual_extreme))
    passed = (
        finite
        and range_ok
        and score_ok
        and best_ok
        and location_ok
        and location_consistent
        and best_consistent
    )
    return {
        **base,
        "golden_source": golden_source,
        "score_max_abs_error": f"{max_abs(actual_score, expected_score):.9g}",
        "score_max_relative_error": f"{max_relative(actual_score, expected_score):.9g}",
        "best_score_max_abs_error": f"{max_abs(actual_best, expected_best):.9g}",
        "location_match": location_ok,
        "location_consistent": location_consistent,
        "best_score_consistent": best_consistent,
        "normalized_range_ok": range_ok,
        "finite": finite,
        "status": "passed" if passed else "failed",
        "runner_stderr": runner_stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", help="Run one exact case name")
    parser.add_argument("--scenario", help="Run one scenario")
    parser.add_argument(
        "--limit", type=int, help="Limit selected cases for smoke testing"
    )
    parser.add_argument(
        "--reuse-actual",
        action="store_true",
        help="Reuse existing raw runner outputs and only regenerate the report",
    )
    args = parser.parse_args()

    all_cases = build_cases()
    selected = [
        case
        for case in all_cases
        if (args.case is None or case.name == args.case)
        and (args.scenario is None or case.scenario == args.scenario)
    ]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        raise SystemExit("no cases selected")
    args.output.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, case in enumerate(selected, start=1):
        row = run_case(
            case, args.runner.resolve(), args.output.resolve(), args.reuse_actual
        )
        rows.append(row)
        print(f"[{index}/{len(selected)}] {case.name}: {row['status']}", flush=True)

    with (args.output / "results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "case_count": len(rows),
        "passed": sum(row["status"] == "passed" for row in rows),
        "failed": sum(row["status"] != "passed" for row in rows),
        "cases": [asdict(case) for case in selected],
    }
    with (args.output / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=True)
    print(
        json.dumps({key: summary[key] for key in ("case_count", "passed", "failed")}),
        flush=True,
    )
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
