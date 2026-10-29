# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

import os
import sys
import re
import logging

OP_API_UT = "OP_API_UT"
OP_HOST_UT = "OP_HOST_UT"
OP_GRAPH_UT = "OP_GRAPH_UT"
OP_KERNEL_UT = "OP_KERNEL_UT"
OP_KERNEL_AICPU_UT = "OP_KERNEL_AICPU_UT"
ALL_UT = "ALL_UT"

NEW_OPS_PATH = [
    "image",
    "objdetect",
    # 添加更多算子路径
]
NEW_EXPERIMENTAL_OPS_PATH = [
    "experimental/image",
    "experimental/objdetect",
    # 添加更多算子路径
]
COMM_FILES = [
    "common/src"
    # 添加更多公共路径
]
SOC_MAPPING = {"arch35": "ascend950"}


class FileChangeInfo:
    def __init__(
        self,
        op_api_changed_files=None,
        op_host_changed_files=None,
        op_graph_changed_files=None,
        op_kernel_changed_files=None,
        op_kernel_aicpu_changed_files=None,
        comm_changed_files=None,
        soc_info=None,
    ):
        self.op_api_changed_files = (
            [] if op_api_changed_files is None else op_api_changed_files
        )
        self.op_host_changed_files = (
            [] if op_host_changed_files is None else op_host_changed_files
        )
        self.op_graph_changed_files = (
            [] if op_graph_changed_files is None else op_graph_changed_files
        )
        self.op_kernel_changed_files = (
            [] if op_kernel_changed_files is None else op_kernel_changed_files
        )
        self.op_kernel_aicpu_changed_files = op_kernel_aicpu_changed_files or []
        self.comm_changed_files = (
            [] if comm_changed_files is None else comm_changed_files
        )
        self.soc_info = set() if soc_info is None else soc_info


def get_file_change_info_from_ci(changed_file_info_from_ci, ops_path):
    """
    get file change info from ci, ci will write `git diff > /or_filelist.txt`
    :param changed_file_info_from_ci: git diff result file from ci
    :return: None or FileChangeInfo
    """
    or_file_path = os.path.realpath(changed_file_info_from_ci)
    if not os.path.exists(or_file_path):
        logging.error(
            "[ERROR] change file does not exist, can not get file change info in this pull request."
        )
        return None

    def _compile_patterns(ops_path):
        p = {}
        p["host"] = re.compile(rf"({'|'.join(ops_path)})/.*/op_host/.*\.(cc|cpp|h)$")
        p["api"] = re.compile(rf"({'|'.join(ops_path)})/.*/op_api/.*\.(cc|cpp|h)$")
        p["kernel"] = re.compile(
            rf"({'|'.join(ops_path)})/.*/op_kernel/.*\.(cc|cpp|h)$"
        )
        p["kernel_aicpu"] = re.compile(
            rf"({'|'.join(ops_path)})/.*/op_kernel_aicpu/.*\.(cc|cpp|h)$"
        )
        p["graph"] = re.compile(rf"({'|'.join(ops_path)})/.*/op_graph/.*\.(cc|cpp|h)$")
        p["host_test"] = re.compile(
            rf"({'|'.join(ops_path)})/.*/tests/ut/op_host/.*\.(cc|cpp|txt)$"
        )
        p["api_test"] = re.compile(
            rf"({'|'.join(ops_path)})/.*/tests/ut/op_api/.*\.(cc|cpp|txt|py)$"
        )
        p["graph_test"] = re.compile(
            rf"({'|'.join(ops_path)})/.*/tests/ut/op_graph/.*\.(cc|cpp|txt)$"
        )
        p["kernel_test"] = re.compile(
            rf"({'|'.join(ops_path)})/.*/tests/ut/op_kernel/.*\.(cc|cpp|txt|py)$"
        )
        p["kernel_aicpu_test"] = re.compile(
            rf"({'|'.join(ops_path)})/.*/tests/ut/op_kernel_aicpu/.*\.(cc|cpp|txt)$"
        )
        p["comm"] = re.compile(rf"^({'|'.join(COMM_FILES)})")
        p["soc"] = re.compile(rf"({'|'.join(re.escape(key) for key in SOC_MAPPING)})")
        return p

    with open(or_file_path) as or_f:
        lines = or_f.readlines()
        op_api_changed_files = []
        op_host_changed_files = []
        op_graph_changed_files = []
        op_kernel_changed_files = []
        op_kernel_aicpu_changed_files = []
        comm_changed_files = []
        soc_info = set()
        patterns = _compile_patterns(ops_path)

        for line in lines:
            line = line.strip()
            ext = os.path.splitext(line)[-1].lower()
            if ext in (".md",):
                continue
            if not os.path.exists(line):
                continue
            if patterns["api"].match(line) or patterns["api_test"].match(line):
                op_api_changed_files.append(line)
            elif patterns["host"].match(line) or patterns["host_test"].match(line):
                op_host_changed_files.append(line)
            elif patterns["kernel"].match(line) or patterns["kernel_test"].match(line):
                op_kernel_changed_files.append(line)
            elif patterns["kernel_aicpu"].match(line) or patterns[
                "kernel_aicpu_test"
            ].match(line):
                op_kernel_aicpu_changed_files.append(line)
            elif patterns["graph"].match(line) or patterns["graph_test"].match(line):
                op_graph_changed_files.append(line)
            elif patterns["comm"].match(line):
                comm_changed_files.append(line)
            soc_match = patterns["soc"].search(line)
            if soc_match:
                matched_key = soc_match.group(1)
                soc_info.add(SOC_MAPPING[matched_key])

    return FileChangeInfo(
        op_host_changed_files=op_host_changed_files,
        op_api_changed_files=op_api_changed_files,
        op_graph_changed_files=op_graph_changed_files,
        op_kernel_changed_files=op_kernel_changed_files,
        op_kernel_aicpu_changed_files=op_kernel_aicpu_changed_files,
        comm_changed_files=comm_changed_files,
        soc_info=soc_info,
    )


def get_change_relate_ut_dir_list(changed_file_info_from_ci, is_experimental):
    if is_experimental == "TRUE":
        ops_path = NEW_EXPERIMENTAL_OPS_PATH
    else:
        ops_path = NEW_OPS_PATH
    file_change_info = get_file_change_info_from_ci(changed_file_info_from_ci, ops_path)
    if not file_change_info:
        logging.info("[INFO] not found file change info, run all c++.")
        return None

    def _get_relate_ut_list_by_file_change():
        relate_ut = set()
        if len(file_change_info.op_host_changed_files) > 0:
            relate_ut.add(OP_HOST_UT)
        if len(file_change_info.op_api_changed_files) > 0:
            relate_ut.add(OP_API_UT)
        if len(file_change_info.op_graph_changed_files) > 0:
            relate_ut.add(OP_GRAPH_UT)
        if len(file_change_info.op_kernel_changed_files) > 0:
            relate_ut.add(OP_KERNEL_UT)
        if len(file_change_info.op_kernel_aicpu_changed_files) > 0:
            relate_ut.add(OP_KERNEL_AICPU_UT)
        if len(file_change_info.comm_changed_files) > 0:
            relate_ut.add(ALL_UT)
        return relate_ut

    try:
        relate_uts = _get_relate_ut_list_by_file_change()
    except BaseException as e:
        logging.error(e.args)
        return None
    return f"{str(relate_uts)}&{','.join(file_change_info.soc_info)}"


if __name__ == "__main__":
    print(get_change_relate_ut_dir_list(sys.argv[1], sys.argv[2]))
