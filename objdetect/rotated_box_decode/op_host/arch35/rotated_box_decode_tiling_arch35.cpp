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
 * \file rotated_box_decode_tiling_arch35.cpp
 * \brief Tiling implementation for rotated_box_decode operator on arch35
 */
#include "rotated_box_decode_tiling_arch35.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "exe_graph/runtime/runtime_attrs.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/extended_kernel_context.h"

namespace optiling {

static constexpr int64_t ELEMS_PER_BOX = 5;
static constexpr int64_t NUM_IO_BUFS = 3;
static constexpr int64_t NUM_CALC_BUFS = 0;
static constexpr int64_t P = 2;
static constexpr int64_t ELEM_ALIGN_FACTOR = 512;
static constexpr int64_t MIN_TILING_BITS = 32768;

using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

enum class ValidationCode : int32_t {
    kOk = 0,
    kDtype = 1,
    kFormat = 2,
    kRankOrChannel = 3,
    kWeightLength = 4,
    kShapeInconsistent = 5,
    kShapeValue = 6,
    kShapeOverflow = 7,
};

struct ValidationInputs {
    ge::DataType anchorDtype;
    ge::DataType deltasDtype;
    ge::Format anchorFormat;
    ge::Format deltasFormat;
    ge::Format outFormat;
    int64_t rank;
    int64_t channel;
    int64_t deltasRank;
    int64_t deltasChannel;
    int64_t anchorB;
    int64_t anchorN;
    int64_t deltasB;
    int64_t deltasN;
    int64_t weightLen;
};

static ValidationCode ValidateInputs(const ValidationInputs& in)
{
    auto dtypeSupported = [](ge::DataType dt) { return dt == ge::DT_FLOAT16 || dt == ge::DT_FLOAT; };
    if (!dtypeSupported(in.anchorDtype) || !dtypeSupported(in.deltasDtype) || in.anchorDtype != in.deltasDtype) {
        return ValidationCode::kDtype;
    }
    if (in.anchorFormat != ge::FORMAT_ND || in.deltasFormat != ge::FORMAT_ND || in.outFormat != ge::FORMAT_ND) {
        return ValidationCode::kFormat;
    }
    // README 约束三输入 shape 完全相同（3D、shape[1]=5）；deltas 的 rank/channel 需独立校验，
    // 否则 [B,6,N] 或 rank!=3 的 deltas 仅凭 B/N 匹配即可直达 kernel，按固定 5 通道布局
    // (b*5+c)*N+n 寻址读取而通道错位、静默产出错误数值。
    if (in.rank != 3 || in.channel != 5 || in.deltasRank != 3 || in.deltasChannel != 5) {
        return ValidationCode::kRankOrChannel;
    }
    if (in.weightLen != 5) {
        return ValidationCode::kWeightLength;
    }
    if (in.anchorB != in.deltasB || in.anchorN != in.deltasN) {
        return ValidationCode::kShapeInconsistent;
    }
    // -1（动态 shape 未具体化的未知维）与负值均 < 0，统一拒绝：
    // 非法 B/N 若进入 totalCount=B*N 与 SelectUbAxis 分块计算会产生负值与未定义结果
    if (in.anchorB < 0 || in.anchorN < 0 || in.deltasB < 0 || in.deltasN < 0) {
        return ValidationCode::kShapeValue;
    }
    // B、N 非负但无上界：B*N 及派生乘积（SelectUbAxis 的 totalBoxesN*minDtypeBits、
    // B*(N*ELEMS_PER_BOX*sizeofT*8)，kernel Init 的 totalElems=(B*5)*N 等）均为 int64
    // 有符号乘法，溢出即 UB，会使 totalCount/核切分/缓冲区长度取到垃圾值。
    // 阈值 INT64_MAX/160（160 = fp32 的 5 通道 bit 宽度，覆盖两种 dtype）远超可物化
    // 规模（FP32 每 box 20B，阈值对应 ~1.15EB 张量），合法输入不可能触达。
    // 除法判别避免判别式自身溢出：N <= INT64_MAX/B 保证 B*N 可安全求值。
    constexpr int64_t kMaxBoxBits = ELEMS_PER_BOX * static_cast<int64_t>(sizeof(float)) * 8; // 160
    if ((in.anchorB != 0 && in.anchorN > INT64_MAX / in.anchorB) ||
        (in.anchorB * in.anchorN > INT64_MAX / kMaxBoxBits)) {
        return ValidationCode::kShapeOverflow;
    }
    return ValidationCode::kOk;
}

struct SelectUbAxisResult {
    int64_t ubAxis;
    int64_t coreNum;
    int64_t blockFormer;
};

static void SelectUbAxis(int64_t B, int64_t N, int64_t availableCoreNum, int64_t sizeofT, SelectUbAxisResult& out)
{
    int64_t minDtypeBits = ELEMS_PER_BOX * sizeofT * 8;
    int64_t totalBoxesN = B * N;
    int64_t coreNumN = CeilDiv(totalBoxesN * minDtypeBits, MIN_TILING_BITS);
    coreNumN = std::min(coreNumN, availableCoreNum);

    bool nAxisSaturated = (coreNumN >= 1) && (totalBoxesN >= coreNumN * ELEM_ALIGN_FACTOR);

    if (nAxisSaturated) {
        out.ubAxis = static_cast<int64_t>(RBD_UB_AXIS_N);
        out.coreNum = coreNumN;
        out.blockFormer = CeilAlign(CeilDiv(totalBoxesN, coreNumN), ELEM_ALIGN_FACTOR);
        return;
    }

    int64_t minDtypeBitsBatch = N * ELEMS_PER_BOX * sizeofT * 8;
    int64_t coreNumB = CeilDiv(B * minDtypeBitsBatch, MIN_TILING_BITS);
    coreNumB = std::min(coreNumB, availableCoreNum);

    if (B >= 2 && coreNumB >= 1) {
        out.ubAxis = static_cast<int64_t>(RBD_UB_AXIS_B);
        out.coreNum = std::max(coreNumB, static_cast<int64_t>(1));
        out.blockFormer = CeilAlign(CeilDiv(B, coreNumB), ELEM_ALIGN_FACTOR);
        return;
    }

    out.ubAxis = static_cast<int64_t>(RBD_UB_AXIS_N);
    out.coreNum = 1;
    out.blockFormer = CeilAlign(totalBoxesN, ELEM_ALIGN_FACTOR);
}

static ge::graphStatus RotatedBoxDecodeTilingFunc(gert::TilingContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do RotatedBoxDecodeTilingFunc");

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);

    const gert::StorageShape* anchorShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, anchorShape);
    const gert::StorageShape* deltasShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, deltasShape);

    const auto& anchorSs = anchorShape->GetStorageShape();
    const auto& deltasSs = deltasShape->GetStorageShape();
    int64_t rank = static_cast<int64_t>(anchorSs.GetDimNum());
    int64_t channel = (rank >= 2) ? anchorSs.GetDim(1) : 0;
    int64_t B = (rank >= 1) ? anchorSs.GetDim(0) : 0;
    int64_t N = (rank >= 3) ? anchorSs.GetDim(2) : 0;
    int64_t deltasRank = static_cast<int64_t>(deltasSs.GetDimNum());
    int64_t deltasB = (deltasRank >= 1) ? deltasSs.GetDim(0) : 0;
    int64_t deltasN = (deltasRank >= 3) ? deltasSs.GetDim(2) : 0;
    int64_t deltasChannel = (deltasRank >= 2) ? deltasSs.GetDim(1) : 0;

    const gert::Tensor* anchorTensor = context->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, anchorTensor);
    const gert::Tensor* deltasTensor = context->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, deltasTensor);
    ge::DataType anchorDtype = anchorTensor->GetDataType();
    ge::DataType deltasDtype = deltasTensor->GetDataType();
    ge::Format anchorFormat = anchorTensor->GetFormat().GetStorageFormat();
    ge::Format deltasFormat = deltasTensor->GetFormat().GetStorageFormat();

    const gert::CompileTimeTensorDesc* outDesc = context->GetOutputDesc(0);
    ge::Format outFormat = (outDesc != nullptr) ? outDesc->GetFormat().GetStorageFormat() : ge::FORMAT_ND;

    const float* weightPtr = nullptr;
    int64_t weightLen = 0;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const gert::TypedContinuousVector<float>* wv = attrs->GetListFloat(0);
        if (wv != nullptr) {
            weightLen = static_cast<int64_t>(wv->GetSize());
            if (weightLen == 5) {
                weightPtr = wv->GetData();
            }
        }
    }

    ValidationInputs vin{};
    vin.anchorDtype = anchorDtype;
    vin.deltasDtype = deltasDtype;
    vin.anchorFormat = anchorFormat;
    vin.deltasFormat = deltasFormat;
    vin.outFormat = outFormat;
    vin.rank = rank;
    vin.channel = channel;
    vin.deltasRank = deltasRank;
    vin.deltasChannel = deltasChannel;
    vin.anchorB = B;
    vin.anchorN = N;
    vin.deltasB = deltasB;
    vin.deltasN = deltasN;
    vin.weightLen = weightLen;

    ValidationCode vcode = ValidateInputs(vin);
    if (vcode != ValidationCode::kOk) {
        OP_LOGE(context->GetNodeName(),
                "ValidateInputs failed: code=%d (1=dtype 2=format 3=rank/chan 4=weight 5=shape-consistency "
                "6=negative-or-unconcretized dim 7=product-overflow), rank=%ld channel=%ld deltasRank=%ld "
                "deltasChannel=%ld B=%ld N=%ld deltasB=%ld deltasN=%ld",
                static_cast<int32_t>(vcode), rank, channel, deltasRank, deltasChannel, B, N, deltasB, deltasN);
        return ge::GRAPH_FAILED;
    }

    int64_t sizeofT = static_cast<int64_t>(GetSizeByDataType(anchorDtype));
    uint32_t ubBlockSize = Ops::Base::GetUbBlockSize(context);
    uint32_t cacheLineSize = Ops::Base::GetCacheLineSize(context);

    RotatedBoxDecodeTilingData* td = context->GetTilingData<RotatedBoxDecodeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);

    td->rank = RBD_RANK;
    td->inShape[0] = B;
    td->inShape[1] = RBD_CHANNELS;
    td->inShape[2] = N;
    td->outShape[0] = B;
    td->outShape[1] = RBD_CHANNELS;
    td->outShape[2] = N;
    td->totalCount = B * N;
    td->B = B;
    td->N = N;
    td->channelStride = RBD_CHANNELS * sizeofT;
    td->copyMode = RBD_COPY_MODE_NDDMA;

    for (int64_t i = 0; i < RBD_WEIGHT_LEN; i++) {
        td->weight[i] = (weightPtr != nullptr) ? weightPtr[i] : 1.0f;
    }

    if (B == 0 || N == 0) {
        td->perCoreCount = 0;
        td->ubFactor = 0;
        td->ubAxis = RBD_UB_AXIS_N;
        td->bufferSize = 0;
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(ROTATED_BOX_DECODE_COPY_MODE_NDDMA, ROTATED_BOX_DECODE_UB_AXIS_SEL_N));
        size_t* workspaces = context->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
        workspaces[0] = 0;
        OP_LOGI(context->GetNodeName(), "[rotated_box_decode] empty tensor: B=%ld N=%ld", B, N);
        return ge::GRAPH_SUCCESS;
    }

    // N==1 SIMT branch (key=512): every N==1 shape (both dtypes, any B>0)
    // routes to the per-box scalar SIMT pipeline in rotated_box_decode_simt.h.
    // Rationale: N==1 gives 1-box tiles on the VF path (1/8 lane utilization +
    // fp16 conversion tax + 3 tiny MTE descriptors + 6 flag ops per box) —
    // PRC-SIMT-001's scattered-small-packet regime (10B/20B contiguous per
    // box). The VF path (kernel.h) is untouched; all N>1 shapes keep
    // byte-identical scheduling (keys 0/256).
    // Core count: threads-per-core is the compile-time RBD_SIMT_THREAD_NUM;
    // scale AIV count with the work, floor 1, cap at the platform core count.
    if (N == 1) {
        td->perCoreCount = 0; // unused by the SIMT branch
        td->ubAxis = RBD_UB_AXIS_I;
        td->ubFactor = 0;   // no UB tiles on this path
        td->bufferSize = 0; // no TBuf allocation
        int64_t coreNumSimt = CeilDiv(B, RBD_SIMT_THREAD_NUM);
        coreNumSimt = std::max(coreNumSimt, static_cast<int64_t>(1));
        coreNumSimt = std::min(coreNumSimt, static_cast<int64_t>(coreNum));
        context->SetBlockDim(static_cast<uint32_t>(coreNumSimt));
        context->SetTilingKey(GET_TPL_TILING_KEY(ROTATED_BOX_DECODE_COPY_MODE_NDDMA, ROTATED_BOX_DECODE_UB_AXIS_SEL_I));

        size_t* workspaces = context->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
        workspaces[0] = 0;

        OP_LOGI(context->GetNodeName(),
                "rotated_box_decode tiling (simt): shape=(%ld,5,%ld) totalCount=%ld ubAxis=I coreNum=%ld "
                "threadNum=%ld tilingKey=%lu",
                B, N, td->totalCount, coreNumSimt, RBD_SIMT_THREAD_NUM,
                GET_TPL_TILING_KEY(ROTATED_BOX_DECODE_COPY_MODE_NDDMA, ROTATED_BOX_DECODE_UB_AXIS_SEL_I));
        return ge::GRAPH_SUCCESS;
    }

    SelectUbAxisResult axisResult;
    SelectUbAxis(B, N, static_cast<int64_t>(coreNum), sizeofT, axisResult);

    int64_t ubAxis = axisResult.ubAxis;
    int64_t realCoreNum = axisResult.coreNum;
    int64_t blockFormer = axisResult.blockFormer;

    int64_t perBufBytes = FloorAlign(static_cast<int64_t>(ubSize) / P, static_cast<int64_t>(ubBlockSize));
    int64_t perBoxBytes = NUM_IO_BUFS * ELEMS_PER_BOX * sizeofT + NUM_CALC_BUFS * ELEMS_PER_BOX * sizeof(float);
    int64_t maxBoxNum = perBufBytes / perBoxBytes;
    int64_t alignFactor = static_cast<int64_t>(cacheLineSize) / sizeofT;
    int64_t ubFormer = (maxBoxNum / alignFactor) * alignFactor;
    ubFormer = std::max(ubFormer, alignFactor);

    td->perCoreCount = blockFormer;
    td->ubAxis = ubAxis;
    td->ubFactor = ubFormer;
    td->bufferSize = perBufBytes * P;

    uint64_t tilingKey = (ubAxis == RBD_UB_AXIS_N) ?
                             GET_TPL_TILING_KEY(ROTATED_BOX_DECODE_COPY_MODE_NDDMA, ROTATED_BOX_DECODE_UB_AXIS_SEL_N) :
                             GET_TPL_TILING_KEY(ROTATED_BOX_DECODE_COPY_MODE_NDDMA, ROTATED_BOX_DECODE_UB_AXIS_SEL_B);

    context->SetBlockDim(static_cast<uint32_t>(realCoreNum));
    context->SetTilingKey(tilingKey);

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = 0;

    OP_LOGI(context->GetNodeName(),
            "rotated_box_decode tiling: shape=(%ld,5,%ld) totalCount=%ld ubAxis=%ld coreNum=%ld "
            "blockFormer=%ld ubFormer=%ld bufferSize=%ld tilingKey=%lu",
            B, N, td->totalCount, td->ubAxis, realCoreNum, td->perCoreCount, td->ubFactor, td->bufferSize, tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForRotatedBoxDecode(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<RotatedBoxDecodeCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(RotatedBoxDecode)
    .Tiling(RotatedBoxDecodeTilingFunc)
    .TilingParse<RotatedBoxDecodeCompileInfo>(TilingPrepareForRotatedBoxDecode);

} // namespace optiling
