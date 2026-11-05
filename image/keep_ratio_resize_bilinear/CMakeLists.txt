# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------------------------------------

# resize 算子（ONNX Resize 的 GE 适配算子）锚点目录：
# 算子原型注册于 CANN 内置 IR，本目录仅提供图优化融合规则（op_graph/fusion_pass）。
file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(NOT (UT_TEST_ALL OR OP_HOST_UT OR OP_API_UT OR OP_KERNEL_UT OR OP_GRAPH_UT))
    list(REMOVE_ITEM CURRENT_DIRS tests)
endif()

foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()

add_all_modules_sources(DEPENDENCIES resize_bilinear_v2)
