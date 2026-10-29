# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_HOST_UT)
    # The shared SDK fixture carries metadata in Tensor objects. Guard against
    # reintroducing Tensor-dependent inference that those runtime tests cannot detect.
    set(match_template_infer_source "${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/match_template_infershape.cpp")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${match_template_infer_source}")
    file(READ "${match_template_infer_source}" match_template_infer_code)
    if(match_template_infer_code MATCHES "GetInputTensor[ \t\r\n]*\\(")
        message(FATAL_ERROR "MatchTemplate InferShape must use GetInputDesc for dtype, not GetInputTensor.")
    endif()

    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()
