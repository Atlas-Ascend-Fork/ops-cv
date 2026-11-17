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
 * \file stack_group_points_tiling.cpp
 * \brief Tiling implementation for stack_group_points operator
 */

#include <limits>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/shape_util.h"
#include "stack_group_points_tiling.h"

namespace optiling {

// ========== 常量 ==========
constexpr int64_t PER_CORE_MIN = 1024;               // 每核最小元素数，对齐到 32
constexpr uint32_t DCACHE_SIZE = 32 * 1024;          // DCache 预留 32KB
constexpr uint32_t STATIC_UB_ESTIMATE = 0;           // 无静态 UB
constexpr int64_t WORKSPACE_SIZE = 16 * 1024 * 1024; // 系统工作空间 16MB

// ========== 输入索引常量（SE §2.2） ==========
static constexpr int32_t kFeaturesIdx = 0;
static constexpr int32_t kFeaturesBatchCntIdx = 1;
static constexpr int32_t kIndicesIdx = 2;
static constexpr int32_t kIndicesBatchCntIdx = 3;
static constexpr int32_t kOutputIdx = 0;

// ========== 维度索引常量 ==========
static constexpr int32_t kFeaturesCIdx = 1;      // features dim[1] = C
static constexpr int32_t kIndicesNsampleIdx = 1; // indices dim[1] = nsample
static constexpr int32_t kBatchCntBIdx = 0;      // batch_cnt dim[0] = B

// ========== Tiling Key 常量（与 op_kernel/arch35/stack_group_points_tiling_key.h 一致） ==========
static constexpr int32_t kTilingKeyFp32 = 0;
static constexpr int32_t kTilingKeyFp16 = 1;

struct StackGroupPointsCompileInfo {};

// ========== CheckedMul（非负 int64 溢出安全乘法，溢出返回 false） ==========
static bool CheckedMul(int64_t lhs, int64_t rhs)
{
    return !(lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs));
}

static bool CheckedMul(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (!CheckedMul(lhs, rhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

// ========== ValidateDtype（SE §2.2 + §2.6） ==========
static ge::graphStatus ValidateDtype(gert::TilingContext* context)
{
    auto opName = context->GetNodeName();
    auto featuresDesc = context->GetInputDesc(kFeaturesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresDesc);
    auto featuresDtype = featuresDesc->GetDataType();
    OP_CHECK_IF(featuresDtype != ge::DT_FLOAT16 && featuresDtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(
                    opName, "features", std::to_string(static_cast<int32_t>(featuresDtype)).c_str(), "float16/float32"),
                return ge::GRAPH_FAILED);

    auto fbcDesc = context->GetInputDesc(kFeaturesBatchCntIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, fbcDesc);
    OP_CHECK_IF(
        fbcDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE(opName, "features_batch_cnt",
                                  std::to_string(static_cast<int32_t>(fbcDesc->GetDataType())).c_str(), "int32"),
        return ge::GRAPH_FAILED);

    auto indicesDesc = context->GetInputDesc(kIndicesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesDesc);
    OP_CHECK_IF(
        indicesDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE(opName, "indices",
                                  std::to_string(static_cast<int32_t>(indicesDesc->GetDataType())).c_str(), "int32"),
        return ge::GRAPH_FAILED);

    auto ibcDesc = context->GetInputDesc(kIndicesBatchCntIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, ibcDesc);
    OP_CHECK_IF(
        ibcDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPE(opName, "indices_batch_cnt",
                                  std::to_string(static_cast<int32_t>(ibcDesc->GetDataType())).c_str(), "int32"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ========== ValidateOutputDesc（输出描述符一致性校验） ==========
// 校验输出 desc 与 kernel 写入约定一致：y.dtype 须为 float16/float32 且与 features
// 同型；y rank=3 且声明元素数不小于写入量 m*c*nsample。
static ge::graphStatus ValidateOutputDesc(gert::TilingContext* context, int64_t m, int64_t c, int64_t nsample)
{
    auto opName = context->GetNodeName();

    // 1. 输出 dtype 域校验：y 须在算子支持域内
    auto outputDesc = context->GetOutputDesc(kOutputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    auto yDtype = outputDesc->GetDataType();
    OP_CHECK_IF(
        yDtype != ge::DT_FLOAT16 && yDtype != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPE(opName, "y", std::to_string(static_cast<int32_t>(yDtype)).c_str(), "float16/float32"),
        return ge::GRAPH_FAILED);

    // 2. 输出与输入同型校验：kernel 按 features dtype 写元素
    auto featuresDesc = context->GetInputDesc(kFeaturesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresDesc);
    OP_CHECK_IF(yDtype != featuresDesc->GetDataType(),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    opName, "y", std::to_string(static_cast<int32_t>(yDtype)).c_str(),
                    "y dtype must be the same as features dtype (kernel writes elements of features dtype)"),
                return ge::GRAPH_FAILED);

    // 3. 输出 shape 一致性校验：rank=3 + 声明元素数不低于 kernel 写入量 m*c*nsample
    auto outputShape = context->GetOutputShape(kOutputIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    const gert::Shape& yShape = outputShape->GetStorageShape();
    OP_CHECK_IF(yShape.GetDimNum() != 3,
                OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "y", (std::to_string(yShape.GetDimNum()) + "D").c_str(), "3D"),
                return ge::GRAPH_FAILED);

    // 零维前置扫描：任一维为 0 时声明元素数确定为 0（0 与任何维值相乘必为 0），
    // 须按真实元素数参与下溢比较；仅当全维均为正时，
    // 乘法溢出才可作为容量必然充足处理
    bool hasZeroDim = false;
    for (size_t i = 0; i < yShape.GetDimNum(); i++) {
        if (yShape.GetDim(i) == 0) {
            hasZeroDim = true;
            break;
        }
    }
    int64_t declaredElems = 0;
    if (!hasZeroDim) {
        declaredElems = 1;
        for (size_t i = 0; i < yShape.GetDimNum(); i++) {
            int64_t next = 0;
            if (!CheckedMul(declaredElems, yShape.GetDim(i), next)) {
                return ge::GRAPH_SUCCESS; // 全正维下声明元素数溢出 int64（必然 >= 写入量），视为内存安全
            }
            declaredElems = next;
        }
    }
    // kernel 写入量 = m*c*nsample：乘积溢出时下方主流程 CheckedMul 会以溢出语义拒绝，
    // 此处跳过下溢比较（无法建立安全的比较基准）
    int64_t writeElems = 0;
    if (!CheckedMul(m, c, writeElems) || !CheckedMul(writeElems, nsample, writeElems)) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(declaredElems < writeElems,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName, "y",
                    ("y declared elements=" + std::to_string(declaredElems) +
                     ", kernel write elements=" + std::to_string(writeElems))
                        .c_str(),
                    "y declared element count must not be less than m * c * nsample to avoid out-of-bound write"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ========== ValidateShape（SE §2.5） ==========
static ge::graphStatus ValidateShape(gert::TilingContext* context)
{
    auto opName = context->GetNodeName();
    auto featuresShape = context->GetInputShape(kFeaturesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresShape);
    OP_CHECK_IF(
        featuresShape->GetStorageShape().GetDimNum() != 2,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            opName, "features", (std::to_string(featuresShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);

    auto fbcShape = context->GetInputShape(kFeaturesBatchCntIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, fbcShape);
    OP_CHECK_IF(
        fbcShape->GetStorageShape().GetDimNum() != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "features_batch_cnt",
                                     (std::to_string(fbcShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
        return ge::GRAPH_FAILED);

    auto indicesShape = context->GetInputShape(kIndicesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    OP_CHECK_IF(
        indicesShape->GetStorageShape().GetDimNum() != 2,
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "indices",
                                     (std::to_string(indicesShape->GetStorageShape().GetDimNum()) + "D").c_str(), "2D"),
        return ge::GRAPH_FAILED);

    auto ibcShape = context->GetInputShape(kIndicesBatchCntIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, ibcShape);
    OP_CHECK_IF(
        ibcShape->GetStorageShape().GetDimNum() != 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(opName, "indices_batch_cnt",
                                     (std::to_string(ibcShape->GetStorageShape().GetDimNum()) + "D").c_str(), "1D"),
        return ge::GRAPH_FAILED);

    // 维度值边界校验（防止 kernel 中除零：index % nsample, index / nsample % c）
    OP_CHECK_IF(
        featuresShape->GetStorageShape().GetDim(kFeaturesCIdx) <= 0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName, "features",
            ("features dim[1]=" + std::to_string(featuresShape->GetStorageShape().GetDim(kFeaturesCIdx))).c_str(),
            "features dim[1] (C) must be > 0 to avoid division by zero in kernel"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        indicesShape->GetStorageShape().GetDim(kIndicesNsampleIdx) <= 0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName, "indices",
            ("indices dim[1]=" + std::to_string(indicesShape->GetStorageShape().GetDim(kIndicesNsampleIdx))).c_str(),
            "indices dim[1] (nsample) must be > 0 to avoid division by zero in kernel"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        ibcShape->GetStorageShape().GetDim(kBatchCntBIdx) <= 0,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName, "indices_batch_cnt",
            ("indices_batch_cnt dim[0]=" + std::to_string(ibcShape->GetStorageShape().GetDim(kBatchCntBIdx))).c_str(),
            "indices_batch_cnt dim[0] (B) must be > 0"),
        return ge::GRAPH_FAILED);

    // 跨输入等长校验：两个 batch 计数张量描述同一批划分，批数（dim[0]）必须一致；
    // 该约束同时确保按 b 个元素读取计数张量值时不越界
    int64_t fbcB = fbcShape->GetStorageShape().GetDim(kBatchCntBIdx);
    int64_t b = ibcShape->GetStorageShape().GetDim(kBatchCntBIdx);
    std::string fbcShapeMsg = "features_batch_cnt dim0=" + std::to_string(fbcB) +
                              ", indices_batch_cnt dim0=" + std::to_string(b);
    OP_CHECK_IF(fbcB != b,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(),
                                                       "features_batch_cnt and indices_batch_cnt", fbcShapeMsg.c_str(),
                                                       "features_batch_cnt dim0 must equal indices_batch_cnt dim0"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}
// ========== ValidateElementOverflow（元素数乘积溢出校验，写侧总量经出参回传主流程） ==========
// totalElements 供主流程核数划分与 FillTilingData 消费
static ge::graphStatus ValidateElementOverflow(gert::TilingContext* context, int64_t n, int64_t c, int64_t m,
                                               int64_t nsample, int64_t& totalElements)
{
    // featLen = n * c 是 kernel 读侧越界判据（kernel.h 无溢出防护），溢出会产生回绕的
    // 假判据，此处前置校验（对齐下方 m*c*nsample 的 CheckedMul 既有模式）
    OP_CHECK_IF(!CheckedMul(n, c),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "features",
                                                       ("n=" + std::to_string(n) + ", c=" + std::to_string(c)).c_str(),
                                                       "features total elements n * c overflows int64_t range"),
                return ge::GRAPH_FAILED);

    // 总元素数乘积需在 int64 内可表示：溢出会产生负 totalElements 写入 tiling，kernel 转uint64后成为巨大循环上界
    totalElements = 0;
    OP_CHECK_IF(
        !CheckedMul(m, c, totalElements) || !CheckedMul(totalElements, nsample, totalElements),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "features and indices",
            ("m=" + std::to_string(m) + ", c=" + std::to_string(c) + ", nsample=" + std::to_string(nsample)).c_str(),
            "total elements m * c * nsample overflows int64_t range"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}
// ========== ValidateInputs 调度器 ==========
static ge::graphStatus ValidateInputs(gert::TilingContext* context)
{
    // SE §2.2 + §2.6
    OP_CHECK_IF(ValidateDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateDtype failed"),
                return ge::GRAPH_FAILED);
    // SE §2.5
    OP_CHECK_IF(ValidateShape(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateShape failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ========== GetPlatformInfo ==========
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ========== FillTilingData ==========
static void FillTilingData(StackGroupPointsTilingData* tiling, int64_t m, int64_t c, int64_t nsample, int64_t b,
                           int64_t n, int64_t totalElements, int64_t needCoreNum)
{
    tiling->m = m;
    tiling->c = c;
    tiling->nsample = nsample;
    tiling->b = b;
    tiling->n = n;
    tiling->totalElements = totalElements;
    tiling->needCoreNum = needCoreNum;
}

// ========== DumpTilingData (DFX) ==========
static void DumpTilingData(gert::TilingContext* context, const StackGroupPointsTilingData* tiling)
{
    OP_LOGD(context,
            "StackGroupPointsTilingData: m=%ld, c=%ld, nsample=%ld, b=%ld, n=%ld, totalElements=%ld, needCoreNum=%ld",
            tiling->m, tiling->c, tiling->nsample, tiling->b, tiling->n, tiling->totalElements, tiling->needCoreNum);
}

// ========== GetWorkspaceSize ==========
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    // 仅分配系统 workspace（无用户 workspace）
    currentWorkspace[0] = WORKSPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

// ========== StackGroupPointsTilingFunc 主函数 ==========
static ge::graphStatus StackGroupPointsTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    // 2. 校验输入（SE §2.2 + §2.5 + §2.6）
    OP_CHECK_IF(ValidateInputs(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateInputs failed"),
                return ge::GRAPH_FAILED);

    // 3. 提取 shape 维度
    auto featuresShape = context->GetInputShape(kFeaturesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresShape);
    auto indicesShape = context->GetInputShape(kIndicesIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    auto ibcShape = context->GetInputShape(kIndicesBatchCntIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, ibcShape);

    int64_t n = featuresShape->GetStorageShape().GetDim(0);      // N
    int64_t c = featuresShape->GetStorageShape().GetDim(1);      // C
    int64_t m = indicesShape->GetStorageShape().GetDim(0);       // M
    int64_t nsample = indicesShape->GetStorageShape().GetDim(1); // nsample
    int64_t b = ibcShape->GetStorageShape().GetDim(0);           // B

    // 输出描述符一致性校验：输出 buffer 按调用方声明的输出 desc 分配，
    // kernel 写入量按输入推导——dtype 域/同型 + rank/元素数下溢交叉核验
    OP_CHECK_IF(ValidateOutputDesc(context, m, c, nsample) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateOutputDesc failed"), return ge::GRAPH_FAILED);

    // 元素数乘积溢出校验 + 总元素数推导（读侧 n*c 判据 + 写侧 m*c*nsample 写入量，totalElements 经出参回传）
    int64_t totalElements = 0;
    OP_CHECK_IF(ValidateElementOverflow(context, n, c, m, nsample, totalElements) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateElementOverflow failed"), return ge::GRAPH_FAILED);
    // 向上取整采用商 + 余数，避免 totalElements 接近 INT64_MAX 时 + coreNum - 1 的边界加法溢出
    int64_t perCoreElements = totalElements / coreNum + ((totalElements % coreNum != 0) ? 1 : 0);
    if (perCoreElements < PER_CORE_MIN) {
        perCoreElements = PER_CORE_MIN;
    }
    int64_t needCoreNum = totalElements / perCoreElements + ((totalElements % perCoreElements != 0) ? 1 : 0);
    if (needCoreNum > coreNum) {
        needCoreNum = coreNum;
    }
    if (needCoreNum <= 0) {
        needCoreNum = 1;
    }

    auto tiling = context->GetTilingData<StackGroupPointsTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(StackGroupPointsTilingData), 0, sizeof(StackGroupPointsTilingData)) != EOK,
                OP_LOGE(context, "memset_s failed"), return ge::GRAPH_FAILED);
    FillTilingData(tiling, m, c, nsample, b, n, totalElements, needCoreNum);

    // 6. DFX 日志
    DumpTilingData(context, tiling);

    // 7. 设置 workspace（无条件调用 GetWorkspaceSizes(1)）
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "GetWorkspaceSize failed"),
                return ge::GRAPH_FAILED);

    // 8. 设置 BlockDim、LocalMemory
    context->SetBlockDim(static_cast<int32_t>(needCoreNum));
    OP_CHECK_IF((ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE),
                OP_LOGE(context, "ubSize %lu <= DCACHE_SIZE + STATIC_UB_ESTIMATE", ubSize), return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS),
                OP_LOGE(context, "SetLocalMemorySize failed, ubSize=%lu, DCACHE_SIZE=%u, STATIC_UB_ESTIMATE=%u", ubSize,
                        DCACHE_SIZE, STATIC_UB_ESTIMATE),
                return ge::GRAPH_FAILED);

    // 9. 设置 TilingKey（FP32=0, FP16=1，与 tiling_key.h 定义一致）
    auto dtypeDesc = context->GetInputDesc(kFeaturesIdx);
    auto dtype = dtypeDesc->GetDataType();
    if (dtype == ge::DT_FLOAT16) {
        context->SetTilingKey(kTilingKeyFp16);
    } else if (dtype == ge::DT_FLOAT) {
        context->SetTilingKey(kTilingKeyFp32);
    }

    return ge::GRAPH_SUCCESS;
}

// ========== TilingParse 回调 ==========
static ge::graphStatus TilingParseForStackGroupPoints([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// ========== 算子注册 ==========
IMPL_OP_OPTILING(StackGroupPoints)
    .Tiling(StackGroupPointsTilingFunc)
    .TilingParse<StackGroupPointsCompileInfo>(TilingParseForStackGroupPoints);

} // namespace optiling
