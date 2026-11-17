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
import os
import sys
import json
import subprocess
import hashlib
from pathlib import Path
import numpy as np
import scipy
from scipy.ndimage import gaussian_laplace
from importlib.machinery import SourceFileLoader

direct = (
    SourceFileLoader("independent", str(Path(__file__).with_name("golden.py")))
    .load_module()
    .direct
)
W = Path(os.environ["LOG_OUTPUT"])
TOOLS = Path(__file__).parent
assert scipy.__version__ == "1.13.1" and np.__version__ == "1.26.4"

rows = [
    ("U-R", "uint8", [17, 31], 1, -1, "REFLECT", "noise"),
    ("U-M", "uint8", [2, 5, 7], 3, -1, "REFLECT_101", "corners-zero"),
    ("U-N", "uint8", [2, 1, 1, 31], 0.5, 2, "REPLICATE", "ramp-constant"),
    ("U-C", "uint8", [1, 1], 1, -1, "CONSTANT", "constant"),
    ("H-R", "float16", [2, 1, 7, 13], 1, -1, "REFLECT", "step-zero"),
    ("H-M", "float16", [1, 31], 0.5, -1, "REFLECT_101", "ramp"),
    ("H-N", "float16", [2, 31, 1], 3, 12, "REPLICATE", "pulse-constant"),
    ("H-C", "float16", [5, 7], 1, 4, "CONSTANT", "corners-center"),
    ("F-R", "float32", [2, 9, 35], 3, -1, "REFLECT", "noise-zero"),
    ("F-M", "float32", [2, 1, 1, 1], 1, 4, "REFLECT_101", "signed-constant"),
    ("F-N", "float32", [17, 31], 1, -1, "REPLICATE", "constant"),
    ("F-C", "float32", [2, 1, 7, 13], 0.5, 2, "CONSTANT", "checker-step"),
    ("Z-U", "uint8", [3, 5], 1, 0, "REFLECT", "constant"),
    ("Z-H", "float16", [2, 3, 5], 0.5, 0, "CONSTANT", "noise"),
    ("Z-F", "float32", [2, 1, 3, 5], 3, 0, "REFLECT_101", "signed-step"),
    ("G10", "float16", [17, 31], 10, -1, "REFLECT", "constant-checker"),
    ("G20", "float32", [17, 31], 20, -1, "REFLECT", "constant-pulse"),
    (
        "T-minus",
        "float32",
        [5, 7],
        float(np.nextafter(np.float32(1.125), np.float32(-np.inf))),
        -1,
        "REFLECT",
        "ramp",
    ),
    ("T-half", "float32", [5, 7], 1.125, -1, "REFLECT", "ramp"),
    (
        "T-plus",
        "float32",
        [5, 7],
        float(np.nextafter(np.float32(1.125), np.float32(np.inf))),
        -1,
        "REFLECT",
        "ramp",
    ),
]
if os.environ.get("LOG_SUITE") == "perf":
    rows = [
        (
            "P-" + tag + "-" + scale,
            dt,
            shape,
            1,
            4,
            ("REPLICATE" if tag == "F" and scale == "small" else border),
            "noise",
        )
        for tag, dt, border in [
            ("U", "uint8", "REFLECT"),
            ("H", "float16", "REFLECT_101"),
            ("F", "float32", "CONSTANT"),
        ]
        for scale, shape in [("small", [17, 31]), ("medium", [128, 129])]
    ]
if os.environ.get("LOG_SUITE") == "extended":
    shapes = [
        [1, 1],
        [1, 33],
        [33, 1],
        [3, 31],
        [3, 32],
        [3, 33],
        [5, 129],
        [5, 160],
        [5, 161],
    ]
    specs = [(shape, r) for shape in shapes for r in (0, 1, 4)]
    specs += [([2, 3], r) for r in (7, 123)]
    specs += [
        (shape, 4)
        for shape in (
            [2, 7, 161],
            [2, 1, 7, 161],
            [7, 4097],
            [128, 4095],
            [128, 4096],
            [128, 4097],
            [128, 8193],
        )
    ]
    rows = [
        (
            "E-" + str(i) + "-" + dt + "-" + border,
            dt,
            shape,
            1,
            r,
            border,
            ["noise", "constant", "ramp", "checker", "corners-center", "step", "pulse"][
                i % 7
            ],
        )
        for dt in ("uint8", "float16", "float32")
        for border in ("REFLECT", "REFLECT_101", "REPLICATE", "CONSTANT")
        for i, (shape, r) in enumerate(specs)
    ]
if os.environ.get("LOG_SUITE") == "stability":
    rows = [
        ("S-" + str(i) + "-" + dt + "-" + border, dt, shape, 1, r, border, kind)
        for dt in ("uint8", "float16", "float32")
        for border in ("REFLECT", "REFLECT_101", "REPLICATE", "CONSTANT")
        for i, (shape, r, kind) in enumerate(
            [
                ([3, 1], 1, "constant-pulse"),
                ([1, 7], 4, "constant-pulse"),
                ([3, 31], 1, "checker"),
                ([3, 32], 4, "constant-pulse"),
                ([3, 33], 7, "ramp"),
                ([3, 63], 4, "step"),
                ([3, 64], 4, "checker"),
                ([3, 65], 4, "constant-pulse"),
                ([2, 1, 3, 129], 4, "noise"),
                ([2, 3], 123, "constant-pulse"),
            ]
        )
    ]
if os.environ.get("LOG_SUITE") == "stability":
    rows += [
        (
            "S-replay-" + dt + "-" + border + "-" + variant,
            dt,
            [128, 8193],
            1,
            4,
            border,
            "replay-" + variant,
        )
        for dt in ("uint8", "float16", "float32")
        for border in ("REFLECT", "REFLECT_101", "REPLICATE", "CONSTANT")
        for variant in ("original", "flip", "roll", "newseed")
    ]
if os.environ.get("LOG_SUITE") == "large":
    base = [([512, 512], 1), ([512, 512], 3), ([1080, 1920], 1), ([1080, 1920], 3)]
    rows = [
        (
            "L-" + dt + "-" + str(sh[0]) + "x" + str(sh[1]) + "-s" + str(sg),
            dt,
            sh,
            float(sg),
            4,
            border,
            "noise",
        )
        for dt, border in [
            ("uint8", "REFLECT"),
            ("float16", "REFLECT_101"),
            ("float32", "REPLICATE"),
        ]
        for sh, sg in base
    ]
    rows += [
        ("L-extreme-uint8-s8", "uint8", [512, 512], 8.0, 4, "REFLECT", "checker"),
        ("L-extreme-f32-s03", "float32", [1080, 1920], 0.3, 4, "CONSTANT", "pulse"),
        ("L-extreme-f16-r0", "float16", [1024, 1024], 1.0, 0, "REFLECT_101", "ramp"),
    ]
if os.environ.get("LOG_SUITE") == "perf2":
    sizes = [[17, 31], [64, 64], [128, 129], [256, 256], [512, 512], [1080, 1920]]
    rows = [
        (
            "Q-" + dt + "-" + str(sh[0]) + "x" + str(sh[1]),
            dt,
            sh,
            1.0,
            4,
            "REFLECT",
            "noise",
        )
        for dt in ("uint8", "float16", "float32")
        for sh in sizes
    ]
    rows += [
        ("Q-" + dt + "-b" + bd, dt, [128, 129], 1.0, 4, bd, "noise")
        for dt in ("uint8", "float16", "float32")
        for bd in ("REFLECT_101", "REPLICATE", "CONSTANT")
    ]
    rows += [
        (
            "Q-" + dt + "-s" + str(sg).replace(".", "p"),
            dt,
            [512, 512],
            float(sg),
            4,
            "REFLECT",
            "noise",
        )
        for dt in ("uint8", "float16", "float32")
        for sg in (0.5, 3, 8)
    ]
    rows += [("Q-f32-4k", "float32", [2160, 3840], 1.0, 4, "REFLECT", "noise")]
mode = {
    "REFLECT": "reflect",
    "REFLECT_101": "mirror",
    "REPLICATE": "nearest",
    "CONSTANT": "constant",
}
results = []
server = None
if os.environ.get("LOG_SUITE") in ("extended", "stability"):
    server = subprocess.Popen(
        [str(W / "0-runner"), "--server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=open(W / "0-server.stderr", "w"),
        text=True,
        bufsize=1,
    )
    while True:
        ready = server.stdout.readline()
        if not ready:
            raise RuntimeError("persistent runner initialization failed")
        if ready.startswith("aclrtCreateStream"):
            break


def metrics(a, b, mask=None):
    if mask is not None:
        a = a[mask]
        b = b[mask]
    if not b.size:
        return None
    err = np.abs(a - b)
    bound = 2**-16 + 2**-10 * np.abs(b)
    return dict(
        max_abs=float(err.max()),
        rmse=float(np.sqrt(np.mean(err**2))),
        relative_l2=float(
            np.linalg.norm(a - b) / max(np.linalg.norm(b), 2**-16 * np.sqrt(b.size))
        ),
        failed_pixels=int(np.count_nonzero(err > bound)),
        matched_ratio=float(np.mean(err <= bound)),
        rmse_bound=float(np.sqrt(np.mean(bound**2))),
    )


def run(cid, x, s, r, t, border):
    ip = W / f"0-{cid}.input.bin"
    op = W / f"0-{cid}.output.bin"
    lp = W / f"0-{cid}.log"
    x.tofile(ip)
    cmd = [
        "timeout",
        "180",
        str(W / "0-runner"),
        str({"uint8": 4, "float16": 1, "float32": 0}[str(x.dtype)]),
        ",".join(map(str, x.shape)),
        repr(s),
        str(r),
        repr(t),
        border,
        str(ip),
        str(op),
        str(W / f"0-{cid}.maps"),
    ]
    if server:
        server.stdin.write(" ".join(cmd[3:]) + " " + str(lp) + "\n")
        server.stdin.flush()
        done = server.stdout.readline().strip()
        if not done.startswith("CASE_DONE "):
            raise RuntimeError("persistent runner failed: " + done)
        import types

        p = types.SimpleNamespace(
            returncode=int(done.split()[1]), stdout=lp.read_text(), stderr=""
        )
    else:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=190)
    lp.write_text(
        json.dumps(cmd)
        + "\n"
        + p.stdout
        + p.stderr
        + "\nEXIT_CODE="
        + str(p.returncode)
        + "\n"
    )
    rec = {
        "new_device_run": True,
        "evidence_source": str(W),
        "id": cid,
        "dtype": str(x.dtype),
        "shape": list(x.shape),
        "sigma": s,
        "radius_arg": r,
        "truncate": t,
        "border": border,
        "command": cmd,
        "returncode": p.returncode,
        "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
    }
    se = float(np.float32(s))
    te = float(np.float32(t))
    rr = r if r >= 0 else int(np.floor(se * te + 0.5))
    rec.update(
        effective_sigma=se,
        effective_sigma_hex=se.hex(),
        effective_truncate=te,
        effective_radius=rr,
    )
    gold = np.stack(
        [
            gaussian_laplace(
                v.astype(np.float64),
                sigma=se,
                output=np.float64,
                mode=mode[border],
                cval=0.0,
                truncate=te,
                radius=rr,
            )
            for v in x.reshape(-1, *x.shape[-2:])
        ]
    ).reshape(x.shape)
    independent, g, d = direct(x, se, rr, border)
    S = float(
        np.max(np.abs(x.astype(float))) * 2 * np.sum(np.abs(g)) * np.sum(np.abs(d))
    )
    rec["double_golden_max_abs"] = float(np.max(np.abs(independent - gold)))
    rec["double_golden_bound"] = 1e-12 + 1e-11 * S
    rec["double_golden_pass"] = bool(
        np.all(np.abs(independent - gold) <= rec["double_golden_bound"])
    )
    np.save(W / f"0-{cid}.golden.npy", gold)
    np.save(W / f"0-{cid}.independent.npy", independent)
    rec["K"] = 2 * rr + 1
    rec["A"] = float(np.max(np.abs(x.astype(float))))
    rec["C"] = float(2 * np.sum(np.abs(g)) * np.sum(np.abs(d)))
    gam = 2 * rec["K"] * 2**-24 / (1 - 2 * rec["K"] * 2**-24)
    rec["roundoff_estimate"] = S * (2 * gam + gam * gam)
    if p.returncode or not op.exists() or op.stat().st_size != x.size * 4:
        rec["status"] = "FAIL_API"
        return rec, None
    a = np.fromfile(op, np.float32).reshape(x.shape).astype(np.float64)
    rec["input_readonly"] = "input_readonly=1" in p.stdout
    rec["workspace_records"] = [
        line for line in p.stdout.splitlines() if line.startswith("workspace=")
    ]
    rec["output_bytes"] = op.stat().st_size
    rec["preflight"] = str((W / "preflight.json"))
    rec["maps"] = str(W / f"0-{cid}.maps")
    rec["log"] = str(lp)
    rec["repeat_bitwise_equal"] = "repeat_bitwise=1" in p.stdout
    rec["mapping_verified"] = "mapping physical=4 user=0 logic=4 back=0" in p.stdout
    rec["api_success_counts"] = {
        term: sum(
            term in line and line.endswith("=0") for line in p.stdout.splitlines()
        )
        for term in [
            "aclnnLaplacianOfGaussianGetWorkspaceSize",
            "aclnnLaplacianOfGaussian(",
            "aclrtSynchronizeStream(stream)",
            "aclrtMemcpy(output.data()",
        ]
    }
    assert (
        rec["input_readonly"]
        and rec["mapping_verified"]
        and rec["repeat_bitwise_equal"]
        and all(v == 2 for v in rec["api_success_counts"].values())
    )
    rec["finite"] = bool(np.all(np.isfinite(a)) and np.all(np.isfinite(gold)))
    rec["metrics"] = metrics(a, gold)
    h, w = x.shape[-2:]
    y, z = np.indices((h, w))
    edge = np.broadcast_to((y < rr) | (y >= h - rr) | (z < rr) | (z >= w - rr), x.shape)
    rec["independent_metrics"] = metrics(a, independent)
    rec["edge"] = metrics(a, gold, edge)
    rec["interior"] = metrics(a, gold, ~edge)
    rec["near_zero"] = metrics(a, gold, np.abs(gold) <= 2**-16)
    rec["failed_coordinates"] = np.argwhere(
        np.abs(a - gold) > 2**-16 + 2**-10 * np.abs(gold)
    ).tolist()
    if rr == 0:
        rec["analytic_max_abs"] = float(
            np.max(np.abs(gold + 2 * x.astype(float) / (se * se)))
        )
    m = rec["metrics"]
    rec["status"] = (
        "PASS"
        if rec["independent_metrics"]["failed_pixels"] == 0
        and rec["independent_metrics"]["max_abs"] <= 0.01
        and rec["double_golden_pass"]
        and rec["finite"]
        and m["failed_pixels"] == 0
        and m["max_abs"] <= 0.01
        and m["rmse"] <= m["rmse_bound"]
        and rec["independent_metrics"]["rmse"]
        <= rec["independent_metrics"]["rmse_bound"]
        and np.all(np.isfinite(independent))
        else "FAIL_PRECISION"
    )
    if rec["status"] == "PASS" and os.environ.get("LOG_BENCH") == "1":
        bench = cmd.copy()
        bench[2] = str(W / "0-benchmark")
        bench += [os.environ["LOG_WARMUP"], os.environ["LOG_REPEATS"]]
        if os.environ.get("LOG_PROFILE") == "1":
            from profile_tasks import collect, summarize

            prof = collect(bench, W / f"0-{cid}.custom-prof")
            rec["kernel"] = summarize(prof, int(os.environ["LOG_REPEATS"]), custom=True)
            # msprof returns success even on some app failures: require its TIMING record.
            text = (prof / "application.log").read_text()
            import types

            bp = types.SimpleNamespace(returncode=0, stdout=text, stderr="")
        else:
            bp = subprocess.run(bench, capture_output=True, text=True, timeout=190)
        (W / f"0-{cid}.benchmark.log").write_text(bp.stdout + bp.stderr)
        if bp.returncode:
            raise RuntimeError(f"benchmark failed: {cid}; see log")
        rec["timing"] = json.loads(
            next(v[7:] for v in bp.stdout.splitlines() if v.startswith("TIMING "))
        )
        print(
            cid,
            "custom event ms: median={median_ms:.6f} min={min_ms:.6f} max={max_ms:.6f}".format(
                **rec["timing"]
            ),
            flush=True,
        )
        if os.environ.get("LOG_VERBOSE") == "1":
            print(cid, rec["timing"], flush=True)
        if os.environ.get("LOG_BASELINE"):
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "baseline", os.environ["LOG_BASELINE"]
            )
            plugin = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin)
            os.environ["LOG_CASE_ID"] = cid
            rec["baseline"] = plugin.benchmark(
                x.copy(),
                s,
                r,
                t,
                border,
                gold.copy(),
                independent.copy(),
                int(os.environ["LOG_WARMUP"]),
                int(os.environ["LOG_REPEATS"]),
            )
            baseline_output = np.asarray(rec["baseline"].pop("output"))
            if (
                baseline_output.shape != x.shape
                or baseline_output.dtype != np.float32
                or not np.all(np.isfinite(baseline_output))
            ):
                raise RuntimeError("invalid baseline output")
            for reference in (gold, independent):
                bm = metrics(baseline_output.astype(np.float64), reference)
                if (
                    bm["failed_pixels"]
                    or bm["max_abs"] > 0.01
                    or bm["rmse"] > bm["rmse_bound"]
                ):
                    raise RuntimeError("baseline precision failed")
            if (
                rec["baseline"].get("precision_pass") is not True
                or rec["baseline"].get("scope") != "device_event_resident_full_log_ms"
            ):
                raise RuntimeError("baseline precision/scope rejected")
            vals = rec["baseline"]["samples_ms"]
            assert len(vals) == int(os.environ["LOG_REPEATS"]) and all(
                np.isfinite(v) and v > 0 for v in vals
            )
            rec["event_ratio"] = float(np.median(vals) / rec["timing"]["median_ms"])
            if "kernel" in rec and "kernel" in rec["baseline"]:
                rec["speedup"] = (
                    rec["baseline"]["kernel"]["median_us"] / rec["kernel"]["median_us"]
                )
            print(
                cid,
                "baseline=" + rec["baseline"].get("implementation", "plugin"),
                "event_ratio={:.4f}".format(rec["event_ratio"]),
                (
                    "kernel_ratio={:.4f}".format(rec["speedup"])
                    if "speedup" in rec
                    else ""
                ),
                flush=True,
            )
    return rec, a


for i, (cid, dt, shape, s, r, border, kind) in enumerate(rows):
    if os.environ.get("LOG_SUITE") == "smoke" and cid not in ["U-R", "H-R", "F-R"]:
        continue
    h, w = shape[-2:]
    n = int(np.prod(shape) // (h * w))
    y, z = np.indices((h, w))
    ramp = 3 * y + z
    step = np.where(z < w // 2, -1, 1) if w > 1 else np.where(y < h // 2, -1, 1)
    checker = np.where((y + z) % 2, -1, 1)
    rng = np.random.Generator(np.random.PCG64(20261020 + i))
    x = np.zeros((n, h, w), float)
    if kind.startswith("noise"):
        x = (
            rng.integers(0, 256, size=(n, h, w), dtype=np.uint8)
            if dt == "uint8"
            else rng.uniform(-255, 255, size=(n, h, w))
        ).astype(float)
    if "ramp" in kind:
        x[0] = ramp if dt == "uint8" else ramp - ramp.mean()
    if kind.startswith("step"):
        x[0] = step
    if kind.startswith("corners"):
        for yy, zz in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}:
            x[0, yy, zz] = 255 if dt == "uint8" else 1
        if "center" in kind:
            x[0, h // 2, w // 2] = 1
    if kind == "checker":
        x[:] = 37 + checker
    if kind == "pulse":
        x[0, h // 2, w // 2] = 255 if dt == "uint8" else 1
    if kind == "pulse-constant":
        x[0, h // 2, w // 2] = 1
    if kind.endswith("-constant"):
        x[1] = 37
    if kind == "constant":
        x[:] = 37
    if kind == "signed-constant":
        x[0] = 37
        x[1] = -37
    if kind == "checker-step":
        x[0] = checker
        x[1] = step
    if kind == "signed-step":
        x[0] = step
        x[1] = -step
    if kind == "constant-checker":
        x[:] = 37 + checker
    if kind == "constant-pulse":
        x[:] = 37
        x[0, h // 2, w // 2] += 1
    if kind.endswith("-zero"):
        x[1] = 0
    if kind.startswith("replay-"):
        seed = 20261020 + 107 + (1 if kind == "replay-newseed" else 0)
        x = np.random.Generator(np.random.PCG64(seed)).integers(
            0, 256, size=(n, h, w), dtype=np.uint8
        )
        if kind == "replay-flip":
            x = x[..., ::-1].copy()
        if kind == "replay-roll":
            x = np.roll(x, 17, axis=-1)
    x = x.astype(dt).reshape(shape)
    t = 5.0 if cid == "Z-H" else 3.0 if cid == "Z-F" else 4.0
    rec, a = run(cid, x, s, r, t, border)
    rec["output_dtype"] = "float32"
    rec["seed"] = 20261020 + i
    rec["generation"] = kind
    results.append(rec)
    print(cid, rec["status"], flush=True)
    if (
        os.environ.get("LOG_SUITE") != "smoke"
        and cid in ["U-M", "H-R", "F-R"]
        and a is not None
    ):
        flat = x.reshape(-1, h, w)
        af = a.reshape(-1, h, w)
        for j, sample in enumerate(flat):
            sub, sa = run(f"{cid}-single{j}", sample, s, r, t, border)
            print(sub["id"], sub["status"], flush=True)
            sub["batch_bitwise_equal"] = sa is not None and bool(
                np.array_equal(sa, af[j])
            )
            results.append(sub)
        mod = x.copy().reshape(-1, h, w)
        mod[0] = 0 if np.any(mod[0]) else 37
        sub, sa = run(f"{cid}-modified", mod.reshape(shape), s, r, t, border)
        print(sub["id"], sub["status"], flush=True)
        sub["other_batch_bitwise_equal"] = sa is not None and bool(
            np.array_equal(sa.reshape(-1, h, w)[1:], af[1:])
        )
        results.append(sub)
    (W / "0-閫愮敤渚嬬粨鏋?json").write_text(json.dumps(results, indent=2))
    if rec["status"].startswith("BLOCKED"):
        break
if server:
    server.stdin.write("QUIT\n")
    server.stdin.flush()
    server.wait(timeout=30)
sys.exit(
    0
    if len(results)
    == (
        {"smoke": 3, "perf": 6}.get(
            os.environ.get("LOG_SUITE"),
            len(rows)
            if os.environ.get("LOG_SUITE") in ("extended", "stability")
            else 29,
        )
    )
    and all(
        v["status"] == "PASS"
        and v.get("batch_bitwise_equal", True)
        and v.get("other_batch_bitwise_equal", True)
        for v in results
    )
    else 2
)
