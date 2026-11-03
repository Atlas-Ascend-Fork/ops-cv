# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)
add_all_modules_sources(OPTYPE lbp_feature ACLNNTYPE aclnn)

# 新增以下内容：如果开启了 UT 测试编译，则进入 tests 子目录
if(ENABLE_UT_EXEC OR ENABLE_TEST)
    add_subdirectory(tests)
endif()
