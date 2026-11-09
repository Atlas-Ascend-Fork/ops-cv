/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

// log/log.h — OP_LOGE_FOR_INVALID_* 规范错误码日志 + Ops::Base::ToString（DataType/Format/Shape 序列化）。
#include "log/log.h"
// op_api/error_report_util.h — error_report::ReportTilingStageError（内部空指针类诊断）。
#include "op_api/error_report_util.h"
// BoundingBoxDecodeTilingData struct (§7, 13 fields) — shared host/kernel.
#include "../../op_kernel/arch35/bounding_box_decode_tiling_struct.h"
// gert::TilingContext — GetInputShape / GetInputDesc / GetAttrs / GetTilingData /
//   SetBlockDim / SetTilingKey / GetWorkspaceSizes / GetPlatformInfo.
#include "exe_graph/runtime/tiling_context.h"
// gert::Tensor — GetDataType / GetStorageFormat (input tensor metadata).
#include "exe_graph/runtime/tensor.h"
// ge::DataType / ge::Format — DT_FLOAT16, DT_FLOAT, FORMAT_ND, etc.
#include "graph/types.h"
// platform_ascendc::PlatformAscendC — GetCoreNumAiv / GetCoreMemSize(UB).
#include "tiling/platform/platform_ascendc.h"
// op_def_registry.h — IMPL_OP_OPTILING macro for tiling registration.
#include "register/op_def_registry.h"

namespace optiling {

// =========================================================================
// Constants (DESIGN §9.3 / §9.4 / §9.5 / §9.7 / §6)
// =========================================================================

// §6 TilingKey values — TPL key = T_value (datatype only, no BOOL param)
//   kIsEmpty TPL parameter removed (Task 41 fix): the BOOL TPL parameter
//   caused the framework's NnopbaseExecutorDoTiling to fail allocating the
//   tiling data buffer in aclnn e2e mode (chicken-and-egg: tilingKey encodes
//   the BOOL value but is set by BoundingBoxDecodeTilingFunc, which needs the buffer first).
//   Now the kernel handles empty tensors via runtime check (td.dim0 == 0).
// (T=FP32=0) → 0   (fp32, handles both normal and empty)
// (T=FP16=1) → 1   (fp16, handles both normal and empty)
constexpr uint64_t BOUNDING_BOX_DECODE_FP32 = 0;
constexpr uint64_t BOUNDING_BOX_DECODE_FP16 = 1;

// §9.3 validation constants
constexpr int64_t MAX_RANK = 2;      // spec.yaml rank_range:[2,2], fixed rank=2
constexpr int64_t ELEMS_PER_BOX = 4; // C=4 fixed (kElemsPerBox)

// §9.5 multi-core split constants
constexpr int64_t MIN_TILING_BITS = 32768; // 4 KB, unit: bits
constexpr int64_t ELEM_ALIGN_FACTOR = 512; // multi-core box alignment factor

// §9.4 UB split constants
constexpr int64_t ALIGN_256 = 256;           // UB alignment, unit: bytes
constexpr int64_t RESERVED_BYTES = 8 * 1024; // UB reserved for sync/TBuf
constexpr int64_t NUM_IO_BUFS = 3;           // B_anchor, B_deltas, B_boxes (§9.1)

// =========================================================================
// Inline math helpers (mirror DESIGN §9.4 / §9.5 formulas)
// =========================================================================
static inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
static inline int64_t AlignUp(int64_t v, int64_t f) { return CeilDiv(v, f) * f; }
static inline int64_t AlignDown(int64_t v, int64_t f) { return (v / f) * f; }

// =========================================================================
// §9.5 ComputeMultiCoreSplit — by box count, 512-box alignment
//
// Fills: dim0, coreNum, blockFormer, blockNum.
//   coreNum     = min(CeilDiv(dim0 × minDtypeBits, MIN_TILING_BITS), availableCoreNum); ≥1
//   blockFormer = AlignUp(CeilDiv(dim0, coreNum), ELEM_ALIGN_FACTOR)
//   blockNum    = CeilDiv(dim0, blockFormer)
// =========================================================================
static void ComputeMultiCoreSplit(int64_t dim0, int64_t sizeofT, int64_t availableCoreNum,
                                  BoundingBoxDecodeTilingData* td)
{
    int64_t minDtypeBits = ELEMS_PER_BOX * sizeofT * 8; // bits per box
    int64_t coreNum = CeilDiv(dim0 * minDtypeBits, MIN_TILING_BITS);
    coreNum = std::min(coreNum, availableCoreNum);
    coreNum = std::max(coreNum, static_cast<int64_t>(1)); // at least 1 core

    int64_t blockFormer = AlignUp(CeilDiv(dim0, coreNum), ELEM_ALIGN_FACTOR);
    int64_t blockNum = CeilDiv(dim0, blockFormer);
    // 512-box alignment can enlarge blockFormer so that blockNum < coreNum;
    // cap coreNum to blockNum to avoid launching idle cores (SetBlockDim).
    coreNum = std::min(coreNum, blockNum);

    td->dim0 = dim0;
    td->coreNum = static_cast<int32_t>(coreNum);
    td->blockFormer = blockFormer;
    td->blockNum = blockNum;
}

// =========================================================================
// §9.4 ComputeUbSplit — by box count, 256B alignment
//
// Fills: ubFormer, ubLoopOfFormerBlock, ubTailOfFormerBlock,
//        ubLoopOfTailBlock, ubTailOfTailBlock.
//   perBoxBytes  = NUM_IO_BUFS×4×sizeof(T) + K×4×sizeof(float)   (§9.1)
//   alignFactor  = ALIGN_256 / (4 × sizeof(T))    [box count]
//   ubFormer     = max(AlignDown((ubSize-RESERVED)/perBoxBytes, alignFactor), alignFactor)
//   blockTail    = dim0 - (blockNum-1)×blockFormer
// =========================================================================
static void ComputeUbSplit(int64_t ubSize, int64_t sizeofT, int64_t K, int64_t blockFormer, int64_t dim0,
                           int64_t blockNum, BoundingBoxDecodeTilingData* td)
{
    int64_t perBoxBytes = NUM_IO_BUFS * ELEMS_PER_BOX * sizeofT + K * ELEMS_PER_BOX * sizeof(float);
    // alignFactor 同时满足两个约束：
    //   1) 256B UB 对齐：ALIGN_256 / (4 × sizeof(T)) box（fp16=32 / fp32=16）；
    //   2) VF 物理访问粒度：Reg::LoadAlign/StoreAlign 每 repeat 固定按 VL_F32=64 元素取放，
    //      ubFormer 为 64 box 的倍数时 ceil(count/64)×64 ≤ ubFormer 恒成立，
    //      消除满 tile/尾 tile 下最高通道越过 TBuf 分配末端的物理读写（SAFE.01/1、SAFE.02/1）。
    // 64 box（fp16=512B / fp32=1024B）天然满足 256B 对齐，取两者较大值。
    constexpr int64_t VF_GRANULARITY_BOXES = 64;
    int64_t alignFactor = std::max(ALIGN_256 / (ELEMS_PER_BOX * sizeofT), VF_GRANULARITY_BOXES);
    int64_t maxBoxNum = (ubSize - RESERVED_BYTES) / perBoxBytes;
    int64_t ubFormer = AlignDown(maxBoxNum, alignFactor);
    ubFormer = std::max(ubFormer, alignFactor); // floor at 1 alignment block

    int64_t ubLoopOfFormerBlock = CeilDiv(blockFormer, ubFormer);
    int64_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;

    int64_t blockTail = dim0 - (blockNum - 1) * blockFormer;
    int64_t ubLoopOfTailBlock = CeilDiv(blockTail, ubFormer);
    int64_t ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * ubFormer;

    td->ubFormer = ubFormer;
    td->ubLoopOfFormerBlock = ubLoopOfFormerBlock;
    td->ubTailOfFormerBlock = ubTailOfFormerBlock;
    td->ubLoopOfTailBlock = ubLoopOfTailBlock;
    td->ubTailOfTailBlock = ubTailOfTailBlock;
}

// =========================================================================
// §9.3 BoundingBoxDecodePreCheck — 异常值校验
//
// Fixed order: null → dtype → format → rank(rank==2) → attr → shape.
// On success returns true and fills N (box count) + isEmpty (N==0).
// On any failure returns false (caller returns GRAPH_FAILED, no kernel launch).
//
// Attr layout (OpDef order, indices match AppendAttr order):
//   index 0: means        (ListFloat→ GetListFloat(0))
//   index 1: stds         (ListFloat→ GetListFloat(1))
//   index 2: max_shape    (ListInt  → GetListInt(2))
//   index 3: wh_ratio_clip(Float    → GetFloat(3))
// =========================================================================
static bool BoundingBoxDecodePreCheck(gert::TilingContext* ctx, int64_t& N, bool& isEmpty)
{
    // 1. null check — GetInputShape bounds-checks (returns nullptr for missing input).
    const gert::StorageShape* aShape = ctx->GetInputShape(0);
    const gert::StorageShape* dShape = ctx->GetInputShape(1);
    if (aShape == nullptr || dShape == nullptr) {
        error_report::ReportInvalidArgument(ctx->GetNodeName(), "rois or deltas",
                                            "the input shape of required input rois/deltas is missing in the tiling "
                                            "context; check that the graph node provides all required inputs");
        return false;
    }

    // 2. dtype check: anchor_box/deltas ∈ {fp16,fp32} and equal.
    const gert::CompileTimeTensorDesc* aDesc = ctx->GetInputDesc(0);
    const gert::CompileTimeTensorDesc* dDesc = ctx->GetInputDesc(1);
    if (aDesc == nullptr || dDesc == nullptr) {
        error_report::ReportInvalidArgument(ctx->GetNodeName(), "rois or deltas",
                                            "the input desc of required input rois/deltas is missing in the tiling "
                                            "context; check that the graph node provides all required inputs");
        return false;
    }
    ge::DataType aDt = aDesc->GetDataType();
    ge::DataType dDt = dDesc->GetDataType();
    if (aDt != ge::DT_FLOAT16 && aDt != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(ctx->GetNodeName(), "rois", Ops::Base::ToString(aDt).c_str(), "float16 or float32");
        return false;
    }
    if (aDt != dDt) {
        const std::string dtypeMsg = Ops::Base::ToString(aDt) + " and " + Ops::Base::ToString(dDt);
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(ctx->GetNodeName(), "rois and deltas", dtypeMsg.c_str(),
                                               "rois and deltas must have the same dtype");
        return false;
    }

    // 3. format check: inputs must be ND.
    ge::Format aFmt = aDesc->GetStorageFormat();
    ge::Format dFmt = dDesc->GetStorageFormat();
    if (aFmt != ge::FORMAT_ND || dFmt != ge::FORMAT_ND) {
        const std::string formatMsg = Ops::Base::ToString(aFmt) + " and " + Ops::Base::ToString(dFmt);
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(ctx->GetNodeName(), "rois and deltas", formatMsg.c_str(),
                                                "only ND format is supported");
        return false;
    }

    // 4. rank check: both inputs rank == 2 (MAX_RANK).
    const gert::Shape& aShp = aShape->GetStorageShape();
    const gert::Shape& dShp = dShape->GetStorageShape();
    if (aShp.GetDimNum() != static_cast<size_t>(MAX_RANK) || dShp.GetDimNum() != static_cast<size_t>(MAX_RANK)) {
        const std::string rankMsg = std::to_string(aShp.GetDimNum()) + " and " + std::to_string(dShp.GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            ctx->GetNodeName(), "rois and deltas", rankMsg.c_str(),
            ("the rank of rois and deltas must be " + std::to_string(MAX_RANK)).c_str());
        return false;
    }

    // 5. attr value-range check.
    const gert::RuntimeAttrs* attrs = ctx->GetAttrs();
    if (attrs == nullptr) {
        error_report::ReportTilingStageError(
            ctx->GetNodeName(), "runtime attrs is null in the tiling context; check the TilingContext construction");
        return false;
    }
    const auto* meansVec = attrs->GetListFloat(0);  // means   ListFloat index 0
    const auto* stdsVec = attrs->GetListFloat(1);   // stds    ListFloat index 1
    const auto* maxShapeVec = attrs->GetListInt(2); // max_shape ListInt index 2
    const float* clipPtr = attrs->GetFloat(3);      // wh_ratio_clip Float index 3
    if (maxShapeVec == nullptr || meansVec == nullptr || stdsVec == nullptr || clipPtr == nullptr) {
        error_report::ReportTilingStageError(
            ctx->GetNodeName(),
            "required attr (means/stds/max_shape/wh_ratio_clip) is missing; check the graph construction");
        return false;
    }
    // means/stds length == 4
    if (meansVec->GetSize() != 4 || stdsVec->GetSize() != 4) {
        const std::string sizeMsg = std::to_string(meansVec->GetSize()) + " and " + std::to_string(stdsVec->GetSize());
        OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON(ctx->GetNodeName(), "means and stds", sizeMsg.c_str(),
                                                 "the size of means and stds must be 4");
        return false;
    }
    const float* stds = stdsVec->GetData();
    // stds elements must be non-zero
    if (stds[0] == 0.0f || stds[1] == 0.0f || stds[2] == 0.0f || stds[3] == 0.0f) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx->GetNodeName(), "stds", "0", "stds must not contain zero elements");
        return false;
    }
    // wh_ratio_clip > 0
    if (*clipPtr <= 0.0f) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ctx->GetNodeName(), "wh_ratio_clip", std::to_string(*clipPtr).c_str(),
                                              "wh_ratio_clip must be positive");
        return false;
    }
    // max_shape length == 2 (H, W)
    if (maxShapeVec->GetSize() != 2) {
        OP_LOGE_FOR_INVALID_LISTSIZE_WITH_REASON(ctx->GetNodeName(), "max_shape",
                                                 std::to_string(maxShapeVec->GetSize()).c_str(),
                                                 "the size of max_shape must be 2");
        return false;
    }

    // 6. shape check: last dim == 4 and N (dim 0) matches across inputs.
    if (aShp.GetDim(1) != ELEMS_PER_BOX || dShp.GetDim(1) != ELEMS_PER_BOX) {
        const std::string lastDimMsg = std::to_string(aShp.GetDim(1)) + " and " + std::to_string(dShp.GetDim(1));
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            ctx->GetNodeName(), "rois and deltas", lastDimMsg.c_str(),
            ("the last dimension of rois and deltas must be " + std::to_string(ELEMS_PER_BOX)).c_str());
        return false;
    }
    if (aShp.GetDim(0) != dShp.GetDim(0)) {
        const std::string nMsg = std::to_string(aShp.GetDim(0)) + " and " + std::to_string(dShp.GetDim(0));
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(ctx->GetNodeName(), "rois and deltas", nMsg.c_str(),
                                                  "the first dimension (N) of rois and deltas must be the same");
        return false;
    }

    N = aShp.GetDim(0);
    // 7. N 值域校验（A2 基线 para_check.check_shape 拒绝非法 shape；spec 声明 N∈[0, INT64_MAX]）：
    //   - N < 0（含未解析动态维 -1）：blockFormer 经 AlignUp 归 0，blockNum 计算除零；
    //   - N > maxSafeN：dim0 × ELEMS_PER_BOX × sizeofT × 8 溢出 int64，AlignUp 回绕为负、
    //     coreNum 归 0 并 SetBlockDim(0)。
    const int64_t sizeofT = (aDt == ge::DT_FLOAT) ? 4 : 2;
    const int64_t maxSafeN = INT64_MAX / (ELEMS_PER_BOX * sizeofT * 8);
    if (N < 0 || N > maxSafeN) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            ctx->GetNodeName(), "rois", std::to_string(N).c_str(),
            ("N must be within the range [0, " + std::to_string(maxSafeN) + "]").c_str());
        return false;
    }
    isEmpty = (N == 0);
    return true;
}

// =========================================================================
// §9.7 FillTilingData — empty/normal fork + attr passthrough + SetTilingKey/SetBlockDim
//
// empty branch:  coreNum=1, compute fields=0, SetTilingKey(dtype), SetBlockDim(1).
// normal branch: ComputeMultiCoreSplit → ComputeUbSplit,
//                SetTilingKey(dtype), SetBlockDim(coreNum).
// tilingKey is now dtype-only (no BOOL component) — see struct.h comment.
// Attr scalars (means/stds/maxShapeH/maxShapeW) are passed through in both branches.
// =========================================================================
static void FillTilingData(gert::TilingContext* ctx, BoundingBoxDecodeTilingData* td, int64_t N, bool isEmpty,
                           int64_t sizeofT, int64_t ubSize, int64_t availableCoreNum)
{
    td->dim0 = N;

    // Attr passthrough (§7 field table; supplied via GE graph attrs — this op is
    // aclnn_exclude, kernel Compute consumes).
    const gert::RuntimeAttrs* attrs = ctx->GetAttrs();
    const auto* meansVec = attrs->GetListFloat(0);
    const auto* stdsVec = attrs->GetListFloat(1);
    const auto* maxShapeVec = attrs->GetListInt(2);
    const float* means = meansVec->GetData();
    const float* stds = stdsVec->GetData();
    const int64_t* ms = maxShapeVec->GetData();
    for (int i = 0; i < 4; i++) {
        td->means[i] = means[i];
        td->stds[i] = stds[i];
    }
    td->maxShapeH = ms[0];
    td->maxShapeW = ms[1];
    // wh_ratio_clip 对齐 A2 基线（canndev tbe clamp_denorm_detal:458-522，主链无条件调用）：
    // Host 预计算 whMaxRatio = |ln(wh_ratio_clip)|，kernel 在反标准化后将 dw/dh 裁剪到
    // ±whMaxRatio 再进入 exp（默认 0.016 → ±4.1352）。PreCheck 已保证 clipPtr 非空且 > 0。
    const float* clipPtr = attrs->GetFloat(3);
    td->whMaxRatio = fabsf(logf(*clipPtr));

    if (isEmpty) {
        // empty branch: coreNum=1, compute fields=0 (kernel short-circuits via td.dim0==0).
        td->coreNum = 1;
        td->blockFormer = 0;
        td->blockNum = 1; // 1 (not 0) to avoid downstream div-by-zero
        td->ubFormer = 0;
        td->ubLoopOfFormerBlock = 0;
        td->ubTailOfFormerBlock = 0;
        td->ubLoopOfTailBlock = 0;
        td->ubTailOfTailBlock = 0;
        ctx->SetBlockDim(1);
        ctx->SetTilingKey((sizeofT == 2) ? BOUNDING_BOX_DECODE_FP16 : BOUNDING_BOX_DECODE_FP32);
    } else {
        // normal branch: K by dtype — fp32→0, fp16→2 (§9.1 NUM_CALC_BUFS).
        int64_t K = (sizeofT == static_cast<int64_t>(sizeof(float))) ? 0 : 2;
        ComputeMultiCoreSplit(N, sizeofT, availableCoreNum, td);
        ComputeUbSplit(ubSize, sizeofT, K, td->blockFormer, td->dim0, td->blockNum, td);
        ctx->SetBlockDim(static_cast<uint32_t>(td->coreNum));
        ctx->SetTilingKey((sizeofT == 2) ? BOUNDING_BOX_DECODE_FP16 : BOUNDING_BOX_DECODE_FP32);
    }
}

// =========================================================================
// §9.9 BoundingBoxDecodeTilingFunc — entry point
//
// Workflow:
//   1. Get TilingData buffer (GetTilingData<T> sets data size = sizeof(T)).
//   2. Query platform info (ubSize / availableCoreNum) via PlatformAscendC.
//   3. BoundingBoxDecodePreCheck (§9.3) — fail → GRAPH_FAILED, no kernel.
//   4. FillTilingData (§9.7) — empty/normal fork + attr passthrough.
//   5. Set workspace = 0 (no cross-core partial merge, §9.6).
// =========================================================================
static ge::graphStatus BoundingBoxDecodeTilingFunc(gert::TilingContext* context)
{
    // 1. TilingData buffer
    BoundingBoxDecodeTilingData* td = context->GetTilingData<BoundingBoxDecodeTilingData>();
    if (td == nullptr) {
        return ge::GRAPH_FAILED;
    }
    std::memset(td, 0, sizeof(BoundingBoxDecodeTilingData));

    // 2. Platform info — ubSize / availableCoreNum (never hard-coded)
    int64_t ubSize = 0;
    int64_t availableCoreNum = 0;
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        platform_ascendc::PlatformAscendC ascendcPlatform(platformInfo);
        uint64_t ub = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        ubSize = static_cast<int64_t>(ub);
        availableCoreNum = static_cast<int64_t>(ascendcPlatform.GetCoreNumAiv());
    }
    if (ubSize <= 0) {
        ubSize = 196608; // 192 KB fallback (ascend950 default)
    }
    if (availableCoreNum <= 0) {
        availableCoreNum = 1;
    }

    // 3. PreCheck (§9.3): null → dtype → format → rank → attr → shape
    int64_t N = 0;
    bool isEmpty = false;
    if (!BoundingBoxDecodePreCheck(context, N, isEmpty)) {
        return ge::GRAPH_FAILED; // validation failure → no kernel launch
    }

    // 4. FillTilingData (§9.7): empty/normal fork + attr passthrough
    ge::DataType aDt = context->GetInputDesc(0)->GetDataType();
    int64_t sizeofT = (aDt == ge::DT_FLOAT) ? 4 : 2;
    FillTilingData(context, td, N, isEmpty, sizeofT, ubSize, availableCoreNum);

    // 5. Workspace = 0 (Elementwise, no cross-core partial merge, §9.6)
    size_t* ws = context->GetWorkspaceSizes(1);
    if (ws != nullptr) {
        ws[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBoundingBoxDecode([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct BoundingBoxDecodeCompileInfo {};

IMPL_OP_OPTILING(BoundingBoxDecode)
    .Tiling(BoundingBoxDecodeTilingFunc)
    .TilingParse<BoundingBoxDecodeCompileInfo>(TilingParseForBoundingBoxDecode);

} // namespace optiling
