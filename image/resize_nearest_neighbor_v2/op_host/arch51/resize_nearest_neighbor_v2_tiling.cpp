/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file resize_nearest_neighbor_v2.cc
 * \brief resize_nearest_neighbor_v2
 */
#include "../arch35/resize_nearest_neighbor_v2_tiling_base.h"
#include "op_host/tiling_util.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"

namespace optiling {
ge::graphStatus Tiling4ResizeNearestNeighborV2(gert::TilingContext* context)
{
    // get compile info ptr
    const ResizeNearestNeighborV2CompileInfo* compileInfo = static_cast<const ResizeNearestNeighborV2CompileInfo*>(
        context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return Tiling4ResizeNearestNeighborV2ForAscendC(context, compileInfo);
}

static ge::graphStatus TilingPrepare4ResizeNearestNeighborV2(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Start TilingPrepare4ResizeNearestNeighborV2.");
    auto compileInfo = context->GetCompiledInfo<ResizeNearestNeighborV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->core_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->core_num <= 0), OP_LOGE(context->GetNodeName(), "core num invalid."),
                return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    OP_CHECK_IF((compileInfo->ubSize <= 0), OP_LOGE(context->GetNodeName(), "ub size invalid."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// register tiling interface of the ResizeNearestNeighborV2 op.
IMPL_OP_OPTILING(ResizeNearestNeighborV2)
    .Tiling(Tiling4ResizeNearestNeighborV2)
    .TilingParse<ResizeNearestNeighborV2CompileInfo>(TilingPrepare4ResizeNearestNeighborV2);
} // namespace optiling
