/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_check.h
 * \brief
 */

#ifndef CV_COMMON_ACLNN_CHECK_H
#define CV_COMMON_ACLNN_CHECK_H

#include <set>
#include "opdev/platform.h"

namespace op {
static inline bool IsRegBase()
{
    const static std::set<NpuArch> regbaseArch = {NpuArch::DAV_3510, NpuArch::DAV_9201};
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    return regbaseArch.find(curArch) != regbaseArch.end();
}

static inline bool IsRegBase(NpuArch arch)
{
    const static std::set<NpuArch> regbaseArch = {NpuArch::DAV_3510, NpuArch::DAV_9201};
    return regbaseArch.find(arch) != regbaseArch.end();
}
} // namespace op
#endif
