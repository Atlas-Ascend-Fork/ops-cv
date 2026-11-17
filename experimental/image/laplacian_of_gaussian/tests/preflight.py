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
import subprocess
import json
import sys
from pathlib import Path

rows = []
for typ in ["health", "usages", "proc-mem"]:
    cmd = ["npu-smi", "info", "-t", typ, "-i", "4", "-c", "0"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    rows.append(dict(cmd=cmd, rc=p.returncode, stdout=p.stdout, stderr=p.stderr))
Path(sys.argv[1]).write_text(json.dumps(rows, indent=2))
assert all(r["rc"] == 0 for r in rows)
assert "OK" in rows[0]["stdout"]
assert any(
    t in rows[2]["stdout"] for t in ["No process in device.", "No running processes"]
)
for line in rows[1]["stdout"].splitlines():
    if any(k in line for k in ["Aicore Usage", "Aivector Usage", "Aicpu Usage"]):
        assert line.split(":")[-1].strip() == "0"
print(json.dumps(rows, indent=2))
