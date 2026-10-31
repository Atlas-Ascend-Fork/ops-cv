#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Finite-value and exit-status checks for IntegralImage result comparison."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import compare_data


class CompareDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def compare(self, actual, expected, dtype=np.float32):
        np.asarray(actual, dtype=dtype).tofile(self.path / "case_output_sat.bin")
        np.asarray(expected, dtype=dtype).tofile(self.path / "case_golden_sat.bin")
        with contextlib.redirect_stdout(io.StringIO()):
            return compare_data.compare_one(
                self.tmp.name, "case", "sat", dtype, (1e-5, 1e-3)
            )

    def test_finite_values(self):
        self.assertTrue(self.compare([1.000001, 2.0], [1.0, 2.0]))

    def test_nonfinite_output(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=value):
                self.assertFalse(self.compare([value, 1.0], [1.0, 1.0]))

    def test_nonfinite_reference(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=value):
                self.assertFalse(self.compare([1.0, 1.0], [value, 1.0]))
                self.assertFalse(self.compare([value], [value]))

    def test_value_mismatch(self):
        self.assertFalse(self.compare([2.0], [1.0]))
        self.assertFalse(self.compare([2], [1], np.int32))

    def test_integer_exact(self):
        self.assertTrue(
            self.compare([-2147483648, 2147483647], [-2147483648, 2147483647], np.int32)
        )

    def test_length_mismatch(self):
        self.assertFalse(self.compare([1.0], [1.0, 2.0]))

    def test_cli_exit_code(self):
        self.compare([np.nan], [1.0])
        result = subprocess.run(
            [
                sys.executable,
                str(Path(compare_data.__file__)),
                self.tmp.name,
                "case",
                "float32",
                "0",
                "0",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("compare result: False", result.stdout)


if __name__ == "__main__":
    unittest.main()
