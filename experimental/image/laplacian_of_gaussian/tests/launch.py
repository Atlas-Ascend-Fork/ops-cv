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
"""Repository-local ACLNN validation and resident device-event timing."""

import argparse
import os
import sys
import subprocess
import tempfile
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--mode", choices=["test", "benchmark"], default="test")
p.add_argument(
    "--suite",
    choices=["smoke", "full", "perf", "extended", "stability", "large", "perf2"],
    default="smoke",
    help="smoke: 3 dtypes; full: 29 frozen representatives",
)
g = p.add_mutually_exclusive_group()
g.add_argument("--package", type=Path)
g.add_argument("--opp-path", type=Path)
p.add_argument(
    "--output", type=Path, help="new directory; default: unique directory under /tmp"
)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--repeats", type=int, default=10)
p.add_argument(
    "--baseline",
    type=Path,
    help="Python plugin; see README protocol (same NPU, same definition)",
)
p.add_argument(
    "--profile",
    action="store_true",
    help="collect native msprof device task times; preserve raw PROF",
)
p.add_argument(
    "--verbose", action="store_true", help="show extraction and compilation logs"
)
a = p.parse_args()
# Internal controls are owned by this invocation, never inherited.
for key in list(os.environ):
    if key.startswith("LOG_"):
        os.environ.pop(key)

if a.warmup < 1 or a.repeats < 1:
    p.error("warmup and repeats must be positive")
tools = Path(__file__).resolve().parent
root = tools.parents[3]
w = (
    a.output.resolve()
    if a.output
    else Path(tempfile.mkdtemp(prefix="log-" + a.mode + "-"))
)
if a.output:
    w.mkdir(parents=True, exist_ok=False)
print("Results:", w, flush=True)


def run(cmd, name):
    r = subprocess.run(list(map(str, cmd)), capture_output=True, text=True, timeout=180)
    (w / name).write_text(r.stdout + r.stderr)
    if a.verbose:
        print(r.stdout + r.stderr, flush=True)
    if r.returncode:
        raise RuntimeError(f"{name}: exit {r.returncode}; diagnostics: {w / name}")


try:
    import numpy
    import scipy

    if numpy.__version__ != "1.26.4" or scipy.__version__ != "1.13.1":
        raise RuntimeError(
            "requires numpy==1.26.4 scipy==1.13.1; use a dedicated venv or PYTHONPATH target, do not overwrite shared packages"
        )
    if a.opp_path:
        base = a.opp_path.resolve()
    else:
        packages = (
            [a.package.resolve()]
            if a.package
            else sorted((root / "build_out").glob("cann-ops-*.run"))
        )
        if len(packages) != 1:
            raise RuntimeError(
                f"expected one build_out package, found {len(packages)}; specify --package"
            )
        pkg = packages[0]
        (w / "package.json").write_text(
            json.dumps(
                {
                    "path": str(pkg),
                    "sha256": hashlib.sha256(pkg.read_bytes()).hexdigest(),
                },
                indent=2,
            )
        )
        run(["bash", pkg, "--noexec", "--extract=" + str(w / "package")], "extract.log")
        base = w / "package"
    vendors = sorted({v.parents[2] for v in base.rglob("op_api/lib/libcust_opapi.so")})
    if base.joinpath("op_api/lib/libcust_opapi.so").is_file():
        vendors = [base]
    if len(vendors) != 1:
        raise RuntimeError(
            f"expected one vendor with libcust_opapi.so, found {vendors}; specify --opp-path vendor directory"
        )
    vendor = vendors[0]
    cann = Path(
        os.environ.get("ASCEND_INSTALL_PATH") or os.environ.get("ASCEND_HOME_PATH", "")
    )
    if not (cann / "include/acl/acl.h").is_file():
        raise RuntimeError(
            "source CANN set_env.sh; set ASCEND_INSTALL_PATH to toolkit root"
        )
    os.environ.update(
        ASCEND_RT_VISIBLE_DEVICES="4",
        ASCEND_CUSTOM_OPP_PATH=str(vendor),
        LOG_OUTPUT=str(w),
        LOG_SUITE=a.suite,
        LOG_BENCH=str(int(a.mode == "benchmark")),
        LOG_WARMUP=str(a.warmup),
        LOG_REPEATS=str(a.repeats),
        ASCEND_SLOG_PRINT_TO_STDOUT="0",
        ASCEND_GLOBAL_LOG_LEVEL="3",
    )
    os.environ["LD_LIBRARY_PATH"] = (
        str(vendor / "op_api/lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    )
    for key, sub in [
        ("TMPDIR", "tmp"),
        ("XDG_CACHE_HOME", "cache"),
        ("ASCEND_CACHE_PATH", "cache"),
        ("ASCEND_WORK_PATH", "runtime"),
    ]:
        (w / sub).mkdir(exist_ok=True)
        os.environ[key] = str(w / sub)
    if a.baseline:
        os.environ["LOG_BASELINE"] = str(a.baseline.resolve())
    os.environ["LOG_VERBOSE"] = str(int(a.verbose))
    os.environ["LOG_PROFILE"] = str(int(a.profile))
    (w / "configuration.json").write_text(
        json.dumps(
            {
                "mode": a.mode,
                "suite": a.suite,
                "vendor": str(vendor),
                "cann": str(cann),
                "python": sys.executable,
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
                "scipy_file": scipy.__file__,
                "baseline": str(a.baseline) if a.baseline else "custom-only",
                "warmup": a.warmup,
                "repeats": a.repeats,
                "scope": "device_event_resident_full_log_ms",
            },
            indent=2,
        )
    )
    for source, target in [("runner.cpp", "0-runner")] + (
        [("benchmark.cpp", "0-benchmark")] if a.mode == "benchmark" else []
    ):
        run(
            [
                "g++",
                "-std=c++17",
                "-D_GLIBCXX_USE_CXX11_ABI=0",
                tools / source,
                "-I" + str(vendor / "op_api/include"),
                "-I" + str(cann / "include"),
                "-L" + str(vendor / "op_api/lib"),
                "-L" + str(cann / "lib64"),
                "-Wl,-rpath," + str(vendor / "op_api/lib"),
                "-lcust_opapi",
                "-lopapi",
                "-lascendcl",
                "-lnnopbase",
                "-o",
                w / target,
            ],
            target + ".compile.log",
        )
    run([sys.executable, tools / "preflight.py", w / "preflight.json"], "preflight.log")
    print(
        "Device physical 4 healthy and idle; "
        + ("custom-only, no speedup" if not a.baseline else "baseline plugin enabled"),
        flush=True,
    )
    r = subprocess.run([sys.executable, tools / "cases.py"])
    sys.exit(r.returncode)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(2)
