/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_PIXEL_SHUFFLE_H_
#define ACLNN_PIXEL_SHUFFLE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnPixelShuffleGetWorkspaceSize(
    const aclTensor* x, int64_t scaleFactor, const aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnPixelShuffle(void* workspace, uint64_t workspaceSize,
                                                                     aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_PIXEL_SHUFFLE_H_
