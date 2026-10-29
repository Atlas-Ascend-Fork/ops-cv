# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(good_features_to_track_tiling_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/good_features_to_track_tiling.cpp
        )
    # 内核入口是模板并用 DTYPE_IMAGE 实例化，必须显式给出
    AddOpTestCase(good_features_to_track "ascend910B1" "-DDTYPE_IMAGE=float" "${good_features_to_track_tiling_files}")
endif()
