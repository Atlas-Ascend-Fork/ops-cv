# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host-only CLI regressions for compile-time grouping temporary files."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "ci"
    / "op_compile_time_grouping.sh"
)
BASH = shutil.which("bash")
TOOLS_AVAILABLE = all(
    shutil.which(tool)
    for tool in ("bash", "awk", "cat", "grep", "mktemp", "rm", "sort")
)
INTERMEDIATES = (
    "result_cv.csv",
    "result_image.csv",
    "result_objdetect.csv",
    "result0.csv",
    "result1.csv",
)
OUTPUTS = ("result.csv", "op_compile_group.yaml")
SENTINEL = b"existing content must be preserved\n"


@unittest.skipUnless(TOOLS_AVAILABLE, "Bash and POSIX text utilities are required")
class TestCompileTimeGrouping(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_workdir(self, name):
        workdir = self.root / name
        workdir.mkdir()
        (workdir / "private temp directory").mkdir()
        return workdir

    def seed_existing_files(self, workdir):
        for name in INTERMEDIATES + OUTPUTS:
            (workdir / name).write_bytes(SENTINEL)

    def assert_preserved(self, workdir, names):
        for name in names:
            self.assertTrue(
                (workdir / name).is_file(), f"Existing file was removed: {name}"
            )
            self.assertEqual((workdir / name).read_bytes(), SENTINEL, name)

    def run_cli(self, workdir, args):
        env = os.environ.copy()
        env["TMPDIR"] = str(workdir / "private temp directory")
        result = subprocess.run(
            [BASH, str(SCRIPT), *map(str, args)],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
        self.assertEqual(list((workdir / "private temp directory").iterdir()), [])
        return result

    def make_input(self, workdir, categories, unknown=False):
        code = workdir / "code"
        code.mkdir()
        lines = []
        for category in categories:
            name = f"{category}_sample"
            (code / category / name).mkdir(parents=True)
            lines.append(f"--main_func={name}_apt --exe_time=10\n")
        if unknown:
            lines.append("--main_func=unknown_op --exe_time=10\n")
        log = workdir / "build.log"
        log.write_text("".join(lines), encoding="utf-8")
        return ["-l", log, "-c", code]

    def test_help_preserves_existing_files(self):
        workdir = self.make_workdir("help")
        self.seed_existing_files(workdir)
        result = self.run_cli(workdir, ["--help"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Usage:", result.stdout)
        self.assert_preserved(workdir, INTERMEDIATES + OUTPUTS)

    def test_invalid_option_preserves_existing_files(self):
        workdir = self.make_workdir("invalid-option")
        self.seed_existing_files(workdir)
        result = self.run_cli(workdir, ["--unknown"])
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("Unknown option:", result.stdout)
        self.assert_preserved(workdir, INTERMEDIATES + OUTPUTS)

    def test_missing_arguments_preserve_existing_files(self):
        workdir = self.make_workdir("missing-arguments")
        self.seed_existing_files(workdir)
        result = self.run_cli(workdir, [])
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("--log and --code are required", result.stdout)
        self.assert_preserved(workdir, INTERMEDIATES + OUTPUTS)

    def test_existing_intermediates_do_not_change_normal_outputs(self):
        for categories in (("cv",), ("image",), ("objdetect",), ("image", "objdetect")):
            with self.subTest(categories=categories):
                name = "-".join(categories)
                clean = self.make_workdir(name + "-clean")
                result = self.run_cli(clean, self.make_input(clean, categories))
                self.assertEqual(result.returncode, 0, result.stderr)
                expected = {name: (clean / name).read_bytes() for name in OUTPUTS}
                workdir = self.make_workdir(name + "-existing")
                self.seed_existing_files(workdir)
                result = self.run_cli(workdir, self.make_input(workdir, categories))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assert_preserved(workdir, INTERMEDIATES)
                # Normal invocations continue replacing the two documented output files.
                for output in OUTPUTS:
                    self.assertEqual((workdir / output).read_bytes(), expected[output])

    def test_failed_grouping_preserves_files_and_cleans_temporary_directory(self):
        workdir = self.make_workdir("failed-grouping")
        self.seed_existing_files(workdir)
        args = self.make_input(workdir, ("image",), unknown=True)
        result = self.run_cli(workdir, args)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("operator not found in any category", result.stdout)
        self.assert_preserved(workdir, INTERMEDIATES + OUTPUTS)


if __name__ == "__main__":
    unittest.main()
