# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

if(UT_TEST_ALL OR OP_KERNEL_UT)
    set(retinex_enhance_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/retinex_enhance_tiling.cpp)
    AddOpTestCase(retinex_enhance "ascend910B1" "" "${retinex_enhance_tiling_files}")
    if(TARGET retinex_enhance_Ascend910B1_cases_obj)
        # This operator uses a plain C++ tiling struct, rather than a registered
        # TilingDataBase definition. Reuse its declaration for the common UT target.
        add_custom_command(TARGET retinex_enhance_Ascend910B1_tiling_tmp POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/retinex_enhance_tiling.h
                ${CMAKE_CURRENT_BINARY_DIR}/retinex_enhance_tiling_data.h
            VERBATIM)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        set(golden_header ${CMAKE_CURRENT_BINARY_DIR}/retinex_enhance_golden.h)
        add_custom_command(
            OUTPUT ${golden_header}
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/gen_golden.py ${golden_header}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/gen_golden.py
            VERBATIM)
        add_custom_target(retinex_enhance_golden DEPENDS ${golden_header})
        add_dependencies(retinex_enhance_Ascend910B1_cases_obj retinex_enhance_golden)
        target_include_directories(retinex_enhance_Ascend910B1_cases_obj PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    endif()
endif()
