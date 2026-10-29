# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression tests for example checking of ACLNN-exposing operators."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CHECK_SCRIPT = Path(__file__).resolve().parents[1] / "check_examples_ut.py"


class CheckExamplesAclnnTest(unittest.TestCase):
    def test_aclnn_detection(self):
        # A missing example should fail only when the operator exposes ACLNN.
        cases = (
            ("dir_root", "directory", "root", "", False),
            ("dir_nested", "directory", "nested", "", False),
            ("comment_root", "none", "root", "comment", False),
            ("comment_nested", "none", "nested", "comment", False),
            ("real_root", "file", "root", "", True),
            ("real_nested", "file", "nested", "", True),
            ("real_cmake", "none", "root", "real", True),
            ("real_inner", "none", "nested", "inner", True),
            ("multiline", "none", "root", "multiline", True),
            ("comment_plus_real", "none", "root", "both", True),
            ("exclude", "none", "root", "exclude", False),
            ("no_api", "none", "root", "", False),
        )
        declarations = {
            "comment": "  # Alternative: ACLNNTYPE aclnn\n"
            "add_all_modules_sources(OPTYPE test ACLNNTYPE aclnn_exclude)\n",
            "real": "add_all_modules_sources(OPTYPE test ACLNNTYPE aclnn)\n",
            "inner": "add_all_modules_sources(OPTYPE test ACLNNTYPE aclnn_inner)\n",
            "multiline": "add_all_modules_sources(OPTYPE test ACLNNTYPE\n  aclnn)\n",
            "both": "# ACLNNTYPE aclnn\n"
            "add_all_modules_sources(OPTYPE test ACLNNTYPE aclnn)\n",
            "exclude": "add_all_modules_sources(OPTYPE test ACLNNTYPE aclnn_exclude)\n",
            "": "",
        }
        for name, kind, where, declaration, exposes_aclnn in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "cmake").mkdir()
                (root / "cmake/variables.cmake").write_text(
                    'set(OP_CATEGORY_LIST "image")\n', encoding="utf-8"
                )
                op = root / "image/test_op"
                (op / "op_host").mkdir(parents=True)
                (op / "tests/ut").mkdir(parents=True)
                (op / "op_host/test_op_def.cpp").write_text(
                    "OP_ADD(TestOp)\n", encoding="utf-8"
                )
                (op / "tests/ut/test.cpp").write_text("// UT\n", encoding="utf-8")
                api = op / ("op_api" if where == "root" else "op_host/op_api")
                if kind != "none":
                    api.mkdir(parents=True)
                    if kind == "directory":
                        (api / "aclnn_support").mkdir()
                    else:
                        (api / "aclnn_real.cpp").write_text(
                            "// API\n", encoding="utf-8"
                        )
                cmake = op / (
                    "CMakeLists.txt" if where == "root" else "op_host/CMakeLists.txt"
                )
                cmake.write_text(declarations[declaration], encoding="utf-8")
                pr_filelist = root / "pr.txt"
                pr_filelist.write_text(
                    "image/test_op/op_host/test_op_def.cpp\n", encoding="utf-8"
                )
                result = subprocess.run(
                    [
                        sys.executable,
                        str(CHECK_SCRIPT),
                        str(root),
                        "-p",
                        str(pr_filelist),
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                    check=False,
                )
                self.assertEqual(
                    result.returncode, int(exposes_aclnn), result.stdout + result.stderr
                )
                if exposes_aclnn:
                    self.assertIn("test_op(examples)", result.stdout)


if __name__ == "__main__":
    unittest.main()
