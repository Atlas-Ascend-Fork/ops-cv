#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "parse_changed_files.py"


class KernelUtChangeSelectionTest(unittest.TestCase):
    def run_parser(self, changed_paths, experimental=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in changed_paths:
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            changed = root / "changed.txt"
            changed.write_text("\n".join(changed_paths) + "\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(SCRIPT),
                    str(changed),
                    "TRUE" if experimental else "FALSE",
                ],
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            related, _, soc = result.stdout.strip().partition("&")
            return (set() if related == "set()" else ast.literal_eval(related)), soc

    def test_kernel_data_generator_and_comparator_trigger_kernel_ut(self):
        base = "image/col2_im_v2/tests/ut/op_kernel/col2_im_v2_data/"
        for filename in ("gen_data.py", "compare_data.py"):
            with self.subTest(filename=filename):
                selected, soc = self.run_parser([base + filename])
                self.assertEqual(selected, {"OP_KERNEL_UT"})
                self.assertEqual(soc, "")

    def test_kernel_cpp_still_triggers_kernel_ut(self):
        selected, _ = self.run_parser(
            ["image/col2_im_v2/tests/ut/op_kernel/test_col2_im_v2.cpp"]
        )
        self.assertEqual(selected, {"OP_KERNEL_UT"})

    def test_experimental_kernel_python_triggers_kernel_ut(self):
        path = "experimental/image/upsample_nearest3d/tests/ut/op_kernel/upsample_nearest3d_data/gen_data.py"
        selected, _ = self.run_parser([path], experimental=True)
        self.assertEqual(selected, {"OP_KERNEL_UT"})

    def test_unrelated_python_and_markdown_do_not_trigger_kernel_ut(self):
        selected, _ = self.run_parser(
            ["scripts/helper.py", "image/col2_im_v2/README.md"]
        )
        self.assertEqual(selected, set())


if __name__ == "__main__":
    unittest.main()
