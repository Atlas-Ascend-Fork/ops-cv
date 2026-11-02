# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_API_UT)
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND NOT "upsample_nearest3d" IN_LIST ASCEND_OP_NAME)
        return()
    endif()
    if(NOT TARGET ${OP_API_MODULE_NAME}_cases_obj)
        add_library(${OP_API_MODULE_NAME}_cases_obj OBJECT)
    endif()
    file(GLOB UPSAMPLE_NEAREST3D_OP_API_CASES
        ${CMAKE_CURRENT_SOURCE_DIR}/test_aclnn_*.cpp
        ${OPS_CV_DIR}/common/stub/*)
    target_sources(${OP_API_MODULE_NAME}_cases_obj PRIVATE ${UPSAMPLE_NEAREST3D_OP_API_CASES})
endif()
