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
 * \file gaussian_filter_3_d_tiling.cpp
 * \brief GaussianFilter3D tiling实现：
 *  1. 校验输入 shape（rank 3/4/5）与属性（sigma、mode）；
 *  2. host侧用fp64生成一维高斯权重（2r+1个，三轴共用）；
 *  3. 按UB容量自适应选取沿W的tile长度，行*tile 线性编号均匀摊核；
 *  4. workspace 放 ping/pong 两块 fp32 中间缓冲。
 */
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "gaussian_filter_3_d_tiling.h"

namespace optiling {

namespace {
constexpr uint32_t ATTR_SIGMA_INDEX = 0;
constexpr uint32_t ATTR_MODE_INDEX = 1;
// truncate 固定 4.0（对齐 scipy.ndimage.gaussian_filter 默认值，不开放为API）
constexpr double GAUSSIAN_TRUNCATE = 4.0;
constexpr uint32_t VEC_ALIGN_NUM = 8; // fp32 32B 对应的元素数
constexpr uint32_t MIN_TILE_LEN = 8;
// UB 预算系数：为流水线/标量栈预留
constexpr double UB_BUDGET_RATIO = 0.8;

constexpr int32_t X_INDEX = 0;
constexpr int32_t Y_INDEX = 0;

uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    if (y == 0U) {
        return x;
    }
    return (x + y - 1U) / y;
}

bool ParseModeStr(const char* modeStr, uint32_t& mode)
{
    std::string s(modeStr == nullptr ? "" : modeStr);
    for (auto& c : s) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    if (s == "REFLECT") {
        mode = GAUSSIAN_MODE_REFLECT;
    } else if (s == "CONSTANT") {
        mode = GAUSSIAN_MODE_CONSTANT;
    } else if (s == "NEAREST") {
        mode = GAUSSIAN_MODE_NEAREST;
    } else if (s == "MIRROR") {
        mode = GAUSSIAN_MODE_MIRROR;
    } else if (s == "WRAP") {
        mode = GAUSSIAN_MODE_WRAP;
    } else {
        return false;
    }
    return true;
}

// r = int(truncate * sigma + 0.5)，round-half-up，注意不是 ceil
uint32_t CalcRadius(double sigma) { return static_cast<uint32_t>(GAUSSIAN_TRUNCATE * sigma + 0.5); }

// host侧fp64生成归一化一维高斯权重 w[i] = exp(-0.5*i^2/sigma^2) / sum，i in [-r, r]
void GenGaussianWeights(double sigma, uint32_t radius, uint32_t* weightBits)
{
    // sigma=0 退化为恒等直通（r=0，唯一 tap 的权重必须显式为 1.0）。
    // 不能直接套公式：i=0、sigma=0 时 exp(-0.5*0*0/(0*0)) = exp(NaN) = NaN，
    // 且 NaN 与 0 比较恒为 false，sum<=0 的兜底也拦不住，最终会把 NaN 权重下发到 kernel。
    if (!(sigma > 0.0)) {
        const float one = 1.0f;
        std::memcpy(&weightBits[0], &one, sizeof(float));
        return;
    }
    const int64_t r = static_cast<int64_t>(radius);
    const int64_t taps = 2 * r + 1;
    double sum = 0.0;
    for (int64_t k = 0; k < taps; k++) {
        int64_t i = k - r;
        double w = std::exp(-0.5 * static_cast<double>(i) * static_cast<double>(i) / (sigma * sigma));
        weightBits[k] = 0U;
        float wf = static_cast<float>(w);
        std::memcpy(&weightBits[k], &wf, sizeof(float));
        sum += w;
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        // 理论上 sigma>0 时不会走到这里；兜底把中心权重置 1.0，宁可直通也不要输出 NaN
        std::memset(weightBits, 0, sizeof(uint32_t) * static_cast<size_t>(taps));
        const float one = 1.0f;
        std::memcpy(&weightBits[radius], &one, sizeof(float));
        return;
    }
    for (int64_t k = 0; k < taps; k++) {
        float wf = 0.0f;
        std::memcpy(&wf, &weightBits[k], sizeof(float));
        wf = static_cast<float>(static_cast<double>(wf) / sum);
        std::memcpy(&weightBits[k], &wf, sizeof(float));
    }
}
} // namespace

static ge::graphStatus Tiling4GaussianFilter3D(gert::TilingContext* context)
{
    GaussianFilter3DTilingData tiling;
    // ---------- shape 解析：rank 3/4/5，前导维合并为 batch，尾部三维为 D/H/W ----------
    const gert::StorageShape* xShape = context->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape& shape = xShape->GetStorageShape();
    size_t rank = shape.GetDimNum();
    OP_CHECK_IF(rank < 3 || rank > 5, OP_LOGE(context, "Input x rank should be 3/4/5, but got %zu.", rank),
                return ge::GRAPH_FAILED);
    uint64_t w = static_cast<uint64_t>(shape.GetDim(rank - 1));
    uint64_t h = static_cast<uint64_t>(shape.GetDim(rank - 2));
    uint64_t d = static_cast<uint64_t>(shape.GetDim(rank - 3));
    uint64_t batch = 1;
    for (size_t i = 0; i + 3 < rank; i++) {
        batch *= static_cast<uint64_t>(shape.GetDim(i));
    }
    OP_CHECK_IF(w == 0 || h == 0 || d == 0 || batch == 0, OP_LOGE(context, "Input x dim should not be 0."),
                return ge::GRAPH_FAILED);
    uint64_t elemCnt = batch * d * h * w;
    OP_CHECK_IF(elemCnt > static_cast<uint64_t>(INT32_MAX),
                OP_LOGE(context, "Total element count %lu exceeds INT32_MAX.", static_cast<unsigned long>(elemCnt)),
                return ge::GRAPH_FAILED);

    // ---------- dtype ----------
    auto dataType = context->GetInputDesc(X_INDEX)->GetDataType();
    uint32_t dtypeKey;
    if (dataType == ge::DT_FLOAT) {
        dtypeKey = 0U;
    } else if (dataType == ge::DT_FLOAT16) {
        dtypeKey = 1U;
    } else if (dataType == ge::DT_BF16) {
        dtypeKey = 2U;
    } else {
        OP_LOGE(context, "Input x dtype only support float32/float16/bfloat16.");
        return ge::GRAPH_FAILED;
    }

    // ---------- 属性 ----------
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    double sigma = static_cast<double>(*attrs->GetFloat(ATTR_SIGMA_INDEX));
    OP_CHECK_IF(sigma < 0.0 || sigma > 100.0, OP_LOGE(context, "sigma should be in [0, 100], but got %f.", sigma),
                return ge::GRAPH_FAILED);
    const auto* modePtr = attrs->GetAttrPointer<char>(ATTR_MODE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, modePtr);
    uint32_t mode = 0U;
    OP_CHECK_IF(!ParseModeStr(modePtr, mode),
                OP_LOGE(context, "mode only support REFLECT/CONSTANT/NEAREST/MIRROR/WRAP, but got %s.", modePtr),
                return ge::GRAPH_FAILED);

    uint32_t radius = CalcRadius(sigma);
    OP_CHECK_IF(radius > GAUSSIAN_FILTER_3D_MAX_RADIUS,
                OP_LOGE(context, "Kernel radius %u exceeds max %u.", radius, GAUSSIAN_FILTER_3D_MAX_RADIUS),
                return ge::GRAPH_FAILED);

    // ---------- UB容量自适应 tile 长度（仅与半径/dtype相关，三种pass共用） ----------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0U;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint64_t budget = static_cast<uint64_t>(static_cast<double>(ubSize) * UB_BUDGET_RATIO);
    // fp32 对齐中段路径：acc + TQue(depth2) 共约 3×tileLen×4，保守按 20B/elem 预留（已 NPU 验证）；
    // fp16/bf16 保留 tapQue 紧凑路径（24B/elem，无 fp16 torch 基线，性能 ratio 不验收）
    uint64_t bytesPerElem = (dtypeKey == 0U) ? 20ULL : 24ULL;
    uint64_t maxLenByUb = budget / bytesPerElem;
    if (maxLenByUb < MIN_TILE_LEN) {
        maxLenByUb = MIN_TILE_LEN;
    }
    uint32_t tileLen = static_cast<uint32_t>(w < maxLenByUb ? w : maxLenByUb);
    tileLen = tileLen / VEC_ALIGN_NUM * VEC_ALIGN_NUM;
    if (tileLen < MIN_TILE_LEN) {
        tileLen = w < MIN_TILE_LEN ? static_cast<uint32_t>(w) : MIN_TILE_LEN;
    }
    OP_CHECK_IF(tileLen == 0U, OP_LOGE(context, "tileLen should not be 0."), return ge::GRAPH_FAILED);
    uint32_t chunkNum = CeilDiv(static_cast<uint32_t>(w), tileLen);
    uint32_t lineNum = static_cast<uint32_t>(batch * d * h);

    // ---------- 多核切分：task = line * chunk 线性编号 ----------
    auto maxCoreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    uint64_t taskNum = static_cast<uint64_t>(lineNum) * chunkNum;
    // kernel内pass间使用SyncAll跨核同步，必须启动全部AIV（空闲核空转后同样到达屏障点）
    uint32_t coreNum = maxCoreNum;
    OP_CHECK_IF(coreNum == 0U, OP_LOGE(context, "coreNum should not be 0."), return ge::GRAPH_FAILED);
    uint32_t tasksPerCore = CeilDiv(static_cast<uint32_t>(taskNum), coreNum);

    // ---------- workspace：ping/pong 两块 fp32 中间缓冲 + 16MB 系统区 ----------
    // 对齐ssim等ops-cv算子惯例：系统区由算子自行声明，kernel内用裸指针访问用户区
    constexpr size_t SYS_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;
    constexpr uint64_t BUF_ALIGN = 256ULL;
    // 单块 fp32 缓冲按 256B 向上对齐（用 uint64 计算，避免大 shape 下 uint32 乘法溢出）
    uint64_t bufBytes = (elemCnt * 4ULL + BUF_ALIGN - 1ULL) / BUF_ALIGN * BUF_ALIGN;
    // pongOffset 为 uint32 字段，单块缓冲不得超过其可表示范围（约 4GB），否则明确报错
    OP_CHECK_IF(bufBytes > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()),
                OP_LOGE(context, "Single fp32 buffer %lu bytes exceeds the uint32 pongOffset range.",
                        static_cast<unsigned long>(bufBytes)),
                return ge::GRAPH_FAILED);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = static_cast<size_t>(bufBytes) * 2U + SYS_WORKSPACE_SIZE;

    // ---------- 填 tiling ----------
    tiling.set_batch(static_cast<uint32_t>(batch));
    tiling.set_depth(static_cast<uint32_t>(d));
    tiling.set_height(static_cast<uint32_t>(h));
    tiling.set_width(static_cast<uint32_t>(w));
    tiling.set_radius(radius);
    tiling.set_mode(mode);
    tiling.set_tileLen(tileLen);
    tiling.set_chunkNum(chunkNum);
    tiling.set_lineNum(lineNum);
    tiling.set_coreNum(coreNum);
    tiling.set_tasksPerCore(tasksPerCore);
    tiling.set_pongOffset(static_cast<uint32_t>(bufBytes));

    // 高斯权重：sigma=0 时 r=0，权重为 [1.0]，恒等直通
    uint32_t weightBits[GAUSSIAN_FILTER_3D_MAX_TAPS];
    std::memset(weightBits, 0, sizeof(weightBits));
    GenGaussianWeights(sigma, radius, weightBits);
    tiling.set_weightBits(weightBits);

    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // tiling key：低2位dtype（0 fp32 / 1 fp16 / 2 bf16）
    context->SetTilingKey(dtypeKey);
    OP_LOGD(context,
            "GaussianFilter3D tiling: elemCnt=%lu sigma=%f r=%u mode=%u tileLen=%u chunkNum=%u lineNum=%u "
            "coreNum=%u tasksPerCore=%u wsSize=%zu",
            static_cast<unsigned long>(elemCnt), sigma, radius, mode, tileLen, chunkNum, lineNum, coreNum, tasksPerCore,
            workspaces[0]);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GaussianFilter3D(gert::TilingParseContext* context)
{
    OP_LOGD(context, "Enter TilingPrepare4GaussianFilter3D");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GaussianFilter3D)
    .Tiling(Tiling4GaussianFilter3D)
    .TilingParse<GaussianFilter3DCompileInfo>(TilingPrepare4GaussianFilter3D);
} // namespace optiling
