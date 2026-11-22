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

NEW_OPS_PATH = [
    "image",
    "objdetect",
    # 添加更多算子路径
]

# ONNX 插件 framework 路径：插件实现源与对应 UT 都在 common/ 下，
# 这些算子可能没有独立的 op_host/op_kernel 实现目录，故不能走 image/objdetect 的目录校验，
# 而是从文件名提取算子名（test_<op>_onnx_plugin.cpp / <op>_onnx_plugin.cpp -> <op>）。
# 变更这些文件时应触发对应 framework UT 编译。
FRAMEWORK_PLUGIN_DIR = "common/src/framework"
FRAMEWORK_UT_DIR = "common/tests/ut/framework"
_ONNX_PLUGIN_NAME_RE = re.compile(r"^(?:test_)?(.+)_onnx_plugin$")


class OperatorChangeInfo:
    def __init__(self, changed_operators=None, operator_file_map=None):
        self.changed_operators = [] if changed_operators is None else changed_operators
        self.operator_file_map = {} if operator_file_map is None else operator_file_map


def _extract_onnx_plugin_op_name(file_name):
    """从 ONNX 插件文件名提取算子名：test_<op>_onnx_plugin.cpp / <op>_onnx_plugin.cpp -> <op>"""
    name, _ = os.path.splitext(os.path.basename(file_name))
    match = _ONNX_PLUGIN_NAME_RE.match(name)
    return match.group(1) if match else ""


def _extract_framework_onnx_plugin_op(file_path):
    """识别 common/src/framework 与 common/tests/ut/framework 下的 ONNX 插件文件，
    返回算子名；非该类路径或文件不存在时返回空串。"""
    clean_path = file_path.lstrip("/")
    if not (
        clean_path.startswith(FRAMEWORK_PLUGIN_DIR + "/")
        or clean_path.startswith(FRAMEWORK_UT_DIR + "/")
    ):
        return ""
    if not os.path.exists(clean_path):
        return ""
    return _extract_onnx_plugin_op_name(file_path)


def extract_operator_name(file_path, is_experimental):
    # 优先处理 framework ONNX 插件路径（common/src/framework、common/tests/ut/framework）：
    # 这类文件按文件名提取算子名，不经 image/objdetect 目录校验。
    fw_op_name = _extract_framework_onnx_plugin_op(file_path)
    if fw_op_name:
        return fw_op_name

    clean_path = file_path.lstrip("/")
    path_parts = clean_path.split("/")
    default_name = ""
    operator_name = ""
    domain = ""
    if is_experimental == "TRUE":
        if len(path_parts) >= 3:
            domain = path_parts[1]
            operator_name = path_parts[2]
            if operator_name == "common" or not os.path.exists(
                f"experimental/{domain}/{operator_name}"
            ):
                return default_name
    else:
        if len(path_parts) >= 2:
            domain = path_parts[0]
            operator_name = path_parts[1]
            if operator_name == "common" or not os.path.exists(
                f"{domain}/{operator_name}"
            ):
                return default_name
    if domain in NEW_OPS_PATH:
        return operator_name
    return default_name


def get_operator_info_from_ci(changed_file_info_from_ci, is_experimental):
    """
    get operator change info from ci, ci will write `git diff > /or_filelist.txt`
    :param changed_file_info_from_ci: git diff result file from ci
    :return: None or OperatorChangeInf
    """
    or_file_path = os.path.realpath(changed_file_info_from_ci)
    if not os.path.exists(or_file_path):
        logging.error(
            "[ERROR] change file does not exist, can not get file change info in this pull request."
        )
        return None
    with open(or_file_path) as or_f:
        lines = or_f.readlines()
        changed_operators = set()
        operator_file_map = {}

        for line in lines:
            line = line.strip()
            ext = os.path.splitext(line)[-1].lower()
            if ext in (".md",):
                continue
            operator_name = extract_operator_name(line, is_experimental)
            if not operator_name:
                continue
            changed_operators.add(operator_name)
            if operator_name not in operator_file_map:
                operator_file_map[operator_name] = []
            operator_file_map[operator_name].append(line)

    return OperatorChangeInfo(
        changed_operators=list(changed_operators), operator_file_map=operator_file_map
    )


def get_change_ops_list(changed_file_info_from_ci, is_experimental):
    ops_change_info = get_operator_info_from_ci(
        changed_file_info_from_ci, is_experimental
    )
    if not ops_change_info:
        logging.info("[INFO] not found ops change info, run all c++.")
        return None

    return ";".join(ops_change_info.changed_operators)


if __name__ == "__main__":
    print(get_change_ops_list(sys.argv[1], sys.argv[2]))
