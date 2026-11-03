/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "scharr_tiling_data.h"
#include "scharr_tiling_key.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr uint64_t UB_RESERVED_BYTES = 16U * 1024U;
constexpr int32_t VECTOR_ALIGN_ELEMENTS = 64;
constexpr int32_t MAX_BLOCK_W = 4096;
constexpr uint64_t UB_ALIGN_BYTES = 32U;
constexpr uint64_t DATA_COPY_MAX_BLOCK_COUNT = 4095U;
// count 模式 Vector API 单次调用的元素上限：255 repeat × 64 元素（FP32）。
// K 行扁平化补偿链的字面量计数为 K×compPitch，必须不超过该上限。
constexpr uint64_t VECTOR_MAX_COUNT_PER_CALL = 16320U;
// Separate simultaneously accessed Vector scratch regions by eight UB rows.
// This changes addresses only; it does not change any arithmetic operation.
// A 256-byte displacement is large enough to break the common periodic bank
// mapping produced by equally sized, back-to-back FP32 vectors on 910B.
constexpr uint64_t UB_BANK_PAD_BYTES = 256U;
// A Scharr output pixel reads input and writes FP32 output. Requiring roughly
// 8 KiB effective work per core avoids paying multi-core launch/halo overhead
// for tiny tensors while still filling all AIVs for genuinely large tensors.
constexpr uint64_t MIN_WORK_BYTES_PER_CORE = 8U * 1024U;
// 列分块 slab 的最小适用宽度：全宽 slab 装不下时，按 bandW 列切带处理。
constexpr uint64_t MIN_BAND_W = 512U;
// A row-split core needs at most two halo rows. Four output rows keep that
// duplicated halo traffic bounded and also preserve useful multirow batches.
constexpr uint64_t MIN_OUTPUT_ROWS_PER_CORE = 4U;
// Narrow images are dominated by per-row queue/Gather overhead rather than
// arithmetic.  uint8 rows are cheap (no software compensation), so about 512
// output pixels per AIV keeps the previous behavior.  fp16/fp32 rows carry a
// serial software-compensated chain (~20 Vector API calls per row), so a much
// smaller per-core pixel floor lets the whole-image plane split drain rows in
// parallel.  W<3 always uses the per-row fallback path, which is even more
// call-bound, so it gets the smallest floor.
constexpr uint64_t SMALL_PIXELS_PER_CORE = 512U;
constexpr uint64_t SMALL_PIXELS_PER_CORE_FP = 32U;
constexpr uint64_t SMALL_PIXELS_PER_CORE_NARROW = 8U;
constexpr uint64_t SMALL_MAX_CORE_NUM = 8U;
constexpr int64_t SMALL_SLAB_MAX_W = 63;

inline uint64_t AlignUp(uint64_t value, uint64_t alignment) { return (value + alignment - 1U) / alignment * alignment; }

struct ScratchLayout {
    uint32_t slots = 0;
    uint32_t work0 = 0;
    uint32_t work1 = 0;
    uint32_t comp0 = 0;
    uint32_t comp1 = 0;
    uint32_t comp2 = 0;
    uint32_t fallback = 0;
    uint32_t mask = 0;
    uint32_t castHalf = 0;
    uint32_t offset0 = 0;
    uint32_t offset1 = 0;
    uint32_t bandTable = 0;   // 列分块 Gather 表 B/C（各含左右两张连续表），A 表复用 offset0/offset1
    uint32_t tailBatch = 0;   // 64行末列的左/中/右快照与跨行 Gather 索引，共1024字节
    uint32_t coefficient = 0; // 常驻补偿系数向量的字节偏移
    uint32_t bytes = 0;
};

inline uint32_t Reserve(uint64_t& cursor, uint64_t bytes, uint64_t alignment = UB_ALIGN_BYTES)
{
    cursor = AlignUp(cursor, alignment);
    const uint32_t offset = static_cast<uint32_t>(cursor);
    cursor += bytes;
    return offset;
}

inline void AddBankPad(uint64_t& cursor) { cursor = AlignUp(cursor + UB_BANK_PAD_BYTES, UB_ALIGN_BYTES); }

// 规划计算暂存区：偏移均以字节计。
// compRows==0：旧布局（fallback 路径），双行补偿布局与 Optimise4 完全一致。
// compRows>=2：slab 路径的 K 行扁平化布局，补偿相关区间按 K 行 × scratchPitch
// 预留；K=2 且 scratchPitch==vectorPitch 时与旧布局兼容（区间只会更大）。
inline ScratchLayout MakeScratchLayout(uint64_t vectorPitch, bool needSoftwareFma, bool needUint8Cast,
                                       bool needTailBatch = false, uint32_t compRows = 0U, uint64_t batchPitch = 0U,
                                       bool xDirection = false, uint64_t bandTableBytes = 0U)
{
    ScratchLayout layout{};
    uint64_t cursor = 0;
    const uint64_t vectorBytes = vectorPitch * sizeof(float);
    const uint64_t compensationBytes = 2U * vectorBytes;
    // Bank separation is enabled only for uint8, which has neither software
    // compensation nor NaN/Inf fallback and was bit-exact in all regular cases
    // after the vector-chain optimization.
    const bool useBankPadding = !needSoftwareFma && needUint8Cast;
    const bool batchLayout = needSoftwareFma && compRows >= 2U;
    const uint64_t scratchPitch = (batchLayout && batchPitch > vectorPitch) ? batchPitch : vectorPitch;
    const uint64_t batchBytes = batchLayout ? static_cast<uint64_t>(compRows) * scratchPitch * sizeof(float) : 0U;
    const uint64_t batchElems = batchLayout ? static_cast<uint64_t>(compRows) * scratchPitch : 0U;
    // X 方向：middleWeighted 放在 slots 内（(1+K) 行），work0 保持 1 行；
    // Y 方向：左右收集块占 2K 行，side 放 work0 需 K 行。
    // K=2 时 X 布局与 Optimise4 完全等容，避免大宽度下 tileH 缩水。
    const uint64_t slotsBytes = !batchLayout ? 3U * vectorBytes :
                                               (xDirection ? std::max(3U * vectorBytes, (1U + compRows) * vectorBytes) :
                                                             std::max(3U * vectorBytes, 2U * batchBytes));
    layout.slots = Reserve(cursor, slotsBytes);
    if (useBankPadding) {
        AddBankPad(cursor);
    }
    const uint64_t work0Bytes = (batchLayout && !xDirection) ? std::max(vectorBytes, batchBytes) : vectorBytes;
    layout.work0 = Reserve(cursor, work0Bytes);
    if (useBankPadding) {
        AddBankPad(cursor);
    }
    layout.work1 = Reserve(cursor, batchLayout ? std::max(compensationBytes, batchBytes) :
                                                 (needSoftwareFma ? compensationBytes : vectorBytes));
    if (useBankPadding) {
        AddBankPad(cursor);
    }
    if (needSoftwareFma) {
        // K 行扁平化补偿链（X finish 与 Y 横向平滑）共用的暂存区。
        // 每块至少覆盖 max(2 行, K*scratchPitch) 个元素。
        const uint64_t compBytes = batchLayout ? std::max(compensationBytes, batchBytes) : compensationBytes;
        layout.comp0 = Reserve(cursor, compBytes);
        layout.comp1 = Reserve(cursor, compBytes);
        layout.comp2 = Reserve(cursor, compBytes);
        layout.fallback = Reserve(cursor, compBytes);
        const uint64_t maskElems = batchLayout ? (batchElems + 64U) : 2U * vectorPitch;
        layout.mask = Reserve(cursor, AlignUp((maskElems + 7U) / 8U, UB_ALIGN_BYTES));
    }
    if (needUint8Cast) {
        // Multirow uint8 conversion includes the W-th zero sentinel used by
        // CONSTANT border. Reserve one extra half element even when W itself
        // is exactly vector aligned.
        layout.castHalf = Reserve(cursor, (vectorPitch + 1U) * sizeof(uint16_t));
    }
    // offset0 is the ordinary [0,4,8,...] table on the fallback path and
    // the left-neighbour table on the full-width path. offset1 is used only
    // by the full-width right-neighbour Gather. No center table is needed.
    layout.offset0 = Reserve(cursor, vectorPitch * sizeof(uint32_t));
    layout.offset1 = Reserve(cursor, vectorPitch * sizeof(uint32_t));
    if (bandTableBytes > 0U) {
        // 列分块模式：B（内部带）/ C（末带）两张左右表；A（首带）复用 offset0/offset1。
        layout.bandTable = Reserve(cursor, bandTableBytes);
    }
    // 优化：常用系数每核只生成一次；覆盖扁平化批处理所需的最长向量。
    // 必须在候选 blockW/tileH 的布局计算中分配，不能挤占预留 UB 或与临时量复用。
    if (needSoftwareFma) {
        layout.coefficient = Reserve(cursor, batchLayout ? std::max(compensationBytes, batchBytes) : compensationBytes);
    }
    // 优化：只为奇数宽 Y 的末列微批预留 1 KiB（空间与图像高度无关）。
    if (needTailBatch) {
        layout.tailBatch = Reserve(cursor, 4U * VECTOR_ALIGN_ELEMENTS * sizeof(float));
    }
    layout.bytes = static_cast<uint32_t>(AlignUp(cursor, UB_ALIGN_BYTES));
    return layout;
}

} // namespace

namespace optiling {

// Host 只根据形状、类型、属性规划核数和 UB，不读取输入数据。
// 所有切分仍采用商余数均分；新增系数区参与每个候选布局的容量检查。
struct ScharrCompileInfo {};

// tiling 分发入口
static ge::graphStatus ScharrTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto* inputTensor = context->GetInputTensor(0);
    const auto* attrs = context->GetAttrs();
    if (inputTensor == nullptr || attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto shapeX = inputTensor->GetOriginShape();
    const int32_t rank = shapeX.GetDimNum();
    if (rank < 2 || rank > 8) {
        return ge::GRAPH_FAILED;
    }

    const int64_t h64 = shapeX.GetDim(rank - 2);
    const int64_t w64 = shapeX.GetDim(rank - 1);
    if (h64 <= 0 || w64 <= 0 || h64 > std::numeric_limits<int32_t>::max() ||
        w64 > std::numeric_limits<int32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    int64_t batchCount64 = 1;
    for (int32_t i = 0; i < rank - 2; ++i) {
        const int64_t dim = shapeX.GetDim(i);
        if (dim <= 0 || batchCount64 > std::numeric_limits<int64_t>::max() / dim) {
            return ge::GRAPH_FAILED;
        }
        batchCount64 *= dim;
    }

    const int64_t plane64 = h64 * w64;
    if (plane64 <= 0 || batchCount64 > std::numeric_limits<int64_t>::max() / plane64) {
        return ge::GRAPH_FAILED;
    }
    const int64_t expectedLength64 = batchCount64 * plane64;
    const int64_t totalLength64 = static_cast<int64_t>(inputTensor->GetShapeSize());
    if (expectedLength64 != totalLength64 || totalLength64 > std::numeric_limits<int32_t>::max() ||
        batchCount64 > std::numeric_limits<int32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    // def declaration order: attr 0 = border_type, attr 1 = direction.
    const char* direction = attrs->GetAttrPointer<char>(1);
    const char* borderType = attrs->GetAttrPointer<char>(0);
    if (direction == nullptr || borderType == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint64_t tilingKey = 0;
    if (std::strcmp(direction, "X") == 0) {
        if (std::strcmp(borderType, "REFLECT_101") == 0) {
            tilingKey = 1;
        } else if (std::strcmp(borderType, "REPLICATE") == 0) {
            tilingKey = 2;
        } else if (std::strcmp(borderType, "CONSTANT") == 0) {
            tilingKey = 3;
        }
    } else if (std::strcmp(direction, "Y") == 0) {
        if (std::strcmp(borderType, "REFLECT_101") == 0) {
            tilingKey = 4;
        } else if (std::strcmp(borderType, "REPLICATE") == 0) {
            tilingKey = 5;
        } else if (std::strcmp(borderType, "CONSTANT") == 0) {
            tilingKey = 6;
        }
    }
    if (tilingKey == 0) {
        return ge::GRAPH_FAILED;
    }
    const int32_t dirIdx = (tilingKey >= 4U) ? 1 : 0;
    const int32_t borderIdx = static_cast<int32_t>((tilingKey - 1U) % 3U);
    const bool isXDirection = std::strcmp(direction, "X") == 0;
    const bool isConstantBorder = std::strcmp(borderType, "CONSTANT") == 0;

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const auto inputType = inputTensor->GetDataType();
    uint32_t inputBytes = 0;
    if (inputType == ge::DT_UINT8) {
        inputBytes = 1U;
    } else if (inputType == ge::DT_FLOAT16) {
        inputBytes = 2U;
    } else if (inputType == ge::DT_FLOAT) {
        inputBytes = 4U;
    } else {
        return ge::GRAPH_FAILED;
    }
    const bool needSoftwareFma = inputType == ge::DT_FLOAT || inputType == ge::DT_FLOAT16;
    const bool needUint8Cast = inputType == ge::DT_UINT8;

    // Pure Vector kernel: query the AIV count from the actual 910B platform.
    // Select fewer cores for small inputs, then split the flattened output-row
    // space by quotient/remainder. The first extraRowCoreCount cores receive
    // one additional row, so the inter-core load difference is at most 1 row.
    uint32_t availableCoreNum = platform.GetCoreNumAiv();
    if (availableCoreNum == 0U) {
        availableCoreNum = platform.GetCoreNum();
    }
    if (availableCoreNum == 0U) {
        availableCoreNum = 1U;
    }
    // 核数选择：综合总读写字节数、行数和切片数量，沿用已验证的小图阈值。
    // 本版不调整核数模型，使性能对比主要反映核内指令开销的变化。
    const uint64_t totalRowTasks = static_cast<uint64_t>(batchCount64) * static_cast<uint64_t>(h64);
    const uint64_t workBytesPerRow = static_cast<uint64_t>(w64) * (static_cast<uint64_t>(inputBytes) + sizeof(float));
    const uint64_t totalWorkBytes = totalRowTasks * workBytesPerRow;
    const uint64_t usefulByWork = std::max<uint64_t>(
        1U, (totalWorkBytes + MIN_WORK_BYTES_PER_CORE - 1U) / MIN_WORK_BYTES_PER_CORE);
    const uint64_t usefulByRows = std::max<uint64_t>(
        1U, (totalRowTasks + MIN_OUTPUT_ROWS_PER_CORE - 1U) / MIN_OUTPUT_ROWS_PER_CORE);
    const uint64_t workCoreNum = std::min<uint64_t>(
        availableCoreNum, std::min<uint64_t>(static_cast<uint64_t>(totalLength64), usefulByWork));
    const uint64_t rowCoreNum = std::min<uint64_t>(
        availableCoreNum, std::min<uint64_t>(totalRowTasks, std::min<uint64_t>(usefulByWork, usefulByRows)));
    // Small full-width images are split only at complete HxW plane boundaries.
    // This avoids duplicated row halos and, more importantly, never lets a
    // core accidentally use a neighbouring image as a vertical border.  This
    // path now also covers H<4 slab strips and W<3 fallback images: their
    // per-row cost is dominated by serial compensation chains, so spreading
    // whole planes over several AIVs is a win even for tiny pixel counts.
    const bool smallSlabShape = h64 >= 1 && w64 >= 1 && w64 <= SMALL_SLAB_MAX_W;
    const uint64_t smallPixelFloor = (w64 < 3) ? SMALL_PIXELS_PER_CORE_NARROW :
                                                 (needSoftwareFma ? SMALL_PIXELS_PER_CORE_FP : SMALL_PIXELS_PER_CORE);
    const uint64_t usefulSmallCores = std::max<uint64_t>(
        1U, (static_cast<uint64_t>(totalLength64) + smallPixelFloor - 1U) / smallPixelFloor);
    const uint64_t planeCoreNum = std::min<uint64_t>(
        availableCoreNum, std::min<uint64_t>(SMALL_MAX_CORE_NUM, std::min<uint64_t>(static_cast<uint64_t>(batchCount64),
                                                                                    usefulSmallCores)));
    // Row splitting preserves the optimized slab path. Only switch to exact
    // pixel balancing when rows cannot expose enough parallelism and each row
    // is wide enough for useful vector work. Mode 2 is reserved for batched
    // narrow planes so every core owns complete images.
    const int32_t splitMode = (smallSlabShape && planeCoreNum > 1U) ?
                                  2 :
                                  ((workCoreNum > rowCoreNum && w64 >= VECTOR_ALIGN_ELEMENTS) ? 1 : 0);
    const uint64_t usedCoreNum64 = splitMode == 2 ? planeCoreNum : (splitMode == 1 ? workCoreNum : rowCoreNum);
    const int32_t usedCoreNum = static_cast<int32_t>(usedCoreNum64);
    const int32_t baseRowsPerCore = splitMode == 0 ? static_cast<int32_t>(totalRowTasks / usedCoreNum64) : 0;
    const int32_t extraRowCoreCount = splitMode == 0 ? static_cast<int32_t>(totalRowTasks % usedCoreNum64) : 0;
    const int32_t basePixelsPerCore = splitMode == 1 ?
                                          static_cast<int32_t>(static_cast<uint64_t>(totalLength64) / usedCoreNum64) :
                                          0;
    const int32_t extraPixelCoreCount = splitMode == 1 ?
                                            static_cast<int32_t>(static_cast<uint64_t>(totalLength64) % usedCoreNum64) :
                                            0;
    const int32_t basePlanesPerCore = splitMode == 2 ?
                                          static_cast<int32_t>(static_cast<uint64_t>(batchCount64) / usedCoreNum64) :
                                          0;
    const int32_t extraPlaneCoreCount = splitMode == 2 ?
                                            static_cast<int32_t>(static_cast<uint64_t>(batchCount64) % usedCoreNum64) :
                                            0;

    const uint64_t usableUb = (ubSize > UB_RESERVED_BYTES) ? (ubSize - UB_RESERVED_BYTES) : ubSize;
    const uint64_t inputAlignElements = 32U / inputBytes;

    int32_t blockW = VECTOR_ALIGN_ELEMENTS;
    uint64_t rawPitch = 0;
    uint64_t vectorPitch = 0;
    ScratchLayout scratch{};
    uint64_t inputBufferBytes = 0;
    uint64_t outputBufferBytes = 0;
    uint64_t floatSlabBytes = 0;
    // UB 规划第一步：从大到小搜索 fallback 列块。
    // MakeScratchLayout 已计入常驻系数，保证不会因新增优化导致 UB 超额。
    const int32_t candidateMax = (w64 >= MAX_BLOCK_W) ?
                                     MAX_BLOCK_W :
                                     static_cast<int32_t>(AlignUp(static_cast<uint64_t>(w64), VECTOR_ALIGN_ELEMENTS));

    for (int32_t candidate = candidateMax; candidate >= VECTOR_ALIGN_ELEMENTS; candidate -= VECTOR_ALIGN_ELEMENTS) {
        const uint64_t candidateRawPitch = AlignUp(static_cast<uint64_t>(candidate) + 2U, inputAlignElements);
        const uint64_t candidateVectorPitch = AlignUp(static_cast<uint64_t>(candidate) + 2U, VECTOR_ALIGN_ELEMENTS);
        const ScratchLayout candidateScratch = MakeScratchLayout(candidateVectorPitch, needSoftwareFma, needUint8Cast);
        const uint64_t candidateInputBytes = 3U * candidateRawPitch * inputBytes;
        const uint64_t candidateOutputBytes = candidateVectorPitch * sizeof(float);
        const uint64_t required = candidateInputBytes + candidateOutputBytes + candidateScratch.bytes;
        if (required <= usableUb) {
            blockW = candidate;
            rawPitch = candidateRawPitch;
            vectorPitch = candidateVectorPitch;
            scratch = candidateScratch;
            inputBufferBytes = candidateInputBytes;
            outputBufferBytes = candidateOutputBytes;
            break;
        }
    }
    if (scratch.bytes == 0U) {
        return ge::GRAPH_FAILED;
    }

    // UB 规划第二步：能容纳完整行时尝试多行 slab，并计算最大 tileH。
    // 输入、转换、输出的物理 pitch 分开管理；不能用 W 代替实际行跨度。
    int32_t tileH = 0;
    int32_t compRows = 0;   // 扁平化补偿链的行批大小 K（0 = fallback 布局）
    uint64_t compPitch = 0; // 补偿暂存区行跨度
    // W>=3 is sufficient for the left/center/right Gather tables. H<4 also
    // takes this path now: the strip degenerates to one full-height batch and
    // avoids the per-column RunTile fallback entirely.
    // splitMode==1 (pixel-range Gather path) requires the identity offset table, which is only
    // built when tileH==0; disable the full-width slab planner for that mode.
    const bool eligible = h64 >= 1 && w64 >= 3 && w64 <= blockW && splitMode != 1;
    uint64_t inputSlabPitch = AlignUp(static_cast<uint64_t>(w64) + 1U, inputAlignElements);
    uint64_t floatSlabPitch = AlignUp(static_cast<uint64_t>(w64) + 1U, VECTOR_ALIGN_ELEMENTS);
    uint64_t outputSlabPitch = AlignUp(static_cast<uint64_t>(w64), 8U);
    if (eligible) {
        uint64_t stripRawPitch = AlignUp(static_cast<uint64_t>(w64) + 1U, inputAlignElements);
        uint64_t stripFloatPitch = AlignUp(static_cast<uint64_t>(w64) + 1U, VECTOR_ALIGN_ELEMENTS);
        uint64_t stripOutputPitch = AlignUp(static_cast<uint64_t>(w64), 8U);
        // By default scratch vectors and Gather results contain exactly W
        // elements; the CONSTANT sentinel belongs only to the source slab.
        uint64_t stripVectorPitch = AlignUp(static_cast<uint64_t>(w64), VECTOR_ALIGN_ELEMENTS);
        // FP16/FP32 X uses an expensive software-compensated 3/10/3 sum to
        // reproduce OpenCV's rounding. FP16 batches through its separate FP32
        // converted slab. FP32 keeps the compact GM->UB slab and instead
        // aligns the staged-output pitch with the cached-row pitch, which
        // enables the same flattened compensation chain without exceeding the
        // DataCopyPad right-padding range. CONSTANT keeps one zero sentinel at
        // column W; the other borders fold offsets into W.
        const bool enableXCompensationBatch = isXDirection && needSoftwareFma;
        if (enableXCompensationBatch) {
            const uint64_t sentinelElements = isConstantBorder ? 1U : 0U;
            const uint64_t batchPitch = AlignUp(static_cast<uint64_t>(w64) + sentinelElements, VECTOR_ALIGN_ELEMENTS);
            stripVectorPitch = batchPitch;
            if (inputType == ge::DT_FLOAT) {
                stripOutputPitch = stripRawPitch;
            } else {
                stripFloatPitch = batchPitch;
                stripOutputPitch = batchPitch;
            }
        }
        // 奇数宽 Y 的末列统一走跨行快照微批：批大小由 compRows 决定，与图像
        // 行数/小图开关解耦。
        const bool enableYTailBatch = !isXDirection && needSoftwareFma && (w64 & 1) != 0 && w64 >= 3;
        const uint64_t batchedCachedPitch = (inputType == ge::DT_FLOAT) ? stripRawPitch : stripFloatPitch;
        const uint64_t rawRowBytes = stripRawPitch * inputBytes;
        const uint64_t convertedRowBytes = inputType == ge::DT_FLOAT ? 0U : stripFloatPitch * sizeof(float);
        const uint64_t residentRowBytes = rawRowBytes + convertedRowBytes;
        const uint64_t outputRowBytes = stripOutputPitch * sizeof(float);
        // CopyIn transfers selectedRows plus two halo rows. DataCopyPad
        // blockCount is limited to 4095 even though its field is uint16_t.
        const uint64_t maxBlockCountRows = DATA_COPY_MAX_BLOCK_COUNT - 2U;
        // 在可行候选中选择 K×tileH 最大者：K 越大单 strip 的向量 API 越少，
        // 但布局占用的 UB 会让 tileH 变小、strip 固定开销占比升高；K×tileH
        // 近似衡量“每 strip 摊薄后的批处理量”，可避免 K 增大却把 tileH 压垮
        // （例如 512 宽 fp16 X 曾出现 K=8/tileH=4，反而比 K=2/tileH=19 慢）。
        // 平局保留更大的 K（按 32、16、8、4、2 顺序，utility 相同则跳过后者）。
        // K×compPitch 不得超过单次 count 模式 Vector API 的元素上限。
        const uint64_t requiredRows = std::min<uint64_t>(static_cast<uint64_t>(h64), 4U);
        uint64_t bestUtility = 0;
        for (int32_t candidateRows : {32, 16, 8, 4, 2}) {
            const uint64_t candidateCompPitch = std::max<uint64_t>(batchedCachedPitch, stripVectorPitch);
            if (needSoftwareFma &&
                static_cast<uint64_t>(candidateRows) * candidateCompPitch > VECTOR_MAX_COUNT_PER_CALL) {
                continue;
            }
            const uint32_t layoutRows = needSoftwareFma ? static_cast<uint32_t>(candidateRows) : 0U;
            const ScratchLayout candidateScratch = MakeScratchLayout(stripVectorPitch, needSoftwareFma, needUint8Cast,
                                                                     enableYTailBatch, layoutRows, batchedCachedPitch,
                                                                     isXDirection);
            const uint64_t fixedBytes = candidateScratch.bytes + 2U * residentRowBytes;
            if (usableUb <= fixedBytes) {
                continue;
            }
            const uint64_t capacityRows = (usableUb - fixedBytes) / (residentRowBytes + outputRowBytes);
            const uint64_t selectedRows = std::min<uint64_t>(static_cast<uint64_t>(h64),
                                                             std::min<uint64_t>(capacityRows, maxBlockCountRows));
            if (selectedRows < requiredRows) {
                continue;
            }
            const uint64_t utility = needSoftwareFma ? static_cast<uint64_t>(candidateRows) * selectedRows : 1U;
            if (utility <= bestUtility) {
                continue;
            }
            bestUtility = utility;
            tileH = static_cast<int32_t>(selectedRows);
            compRows = needSoftwareFma ? candidateRows : 0;
            compPitch = needSoftwareFma ? candidateCompPitch : stripVectorPitch;
            const uint64_t physicalRows = selectedRows + 2U;
            rawPitch = AlignUp(static_cast<uint64_t>(blockW) + 2U, inputAlignElements);
            vectorPitch = stripVectorPitch;
            inputSlabPitch = stripRawPitch;
            floatSlabPitch = stripFloatPitch;
            outputSlabPitch = stripOutputPitch;
            scratch = candidateScratch;
            inputBufferBytes = physicalRows * rawRowBytes;
            outputBufferBytes = selectedRows * outputRowBytes;
            floatSlabBytes = physicalRows * convertedRowBytes;
        }
    }
    // UB 规划第三步：全宽 slab 装不下（tileH=0）且图像较宽时，尝试列分块 slab。
    // 每个带带 ±1 列 halo，内部带用同一套 Gather 表，首/末带各用一套带边界折叠的表。
    int32_t bandW = 0;
    int32_t bandPitch = 0;
    int32_t bandRawPitch = 0;
    int32_t bandFloatPitch = 0;
    int32_t bandOutPitch = 0;
    uint32_t bandTableOffset = 0;
    // The banded planner also raises tileH (>0), so splitMode==1 must skip it as well to keep the
    // identity table that GatherSides relies on.
    if (tileH == 0 && splitMode != 1 && w64 >= static_cast<int64_t>(MIN_BAND_W)) {
        const uint64_t maxBlockCountRows = DATA_COPY_MAX_BLOCK_COUNT - 2U;
        const uint64_t requiredRows = std::min<uint64_t>(static_cast<uint64_t>(h64), 4U);
        // 至少 2 个带：单带几何与全宽 slab 等价（且缺少跨带边界语义），无意义。
        const int32_t bandCandMax = static_cast<int32_t>(std::min<uint64_t>(
            std::min<uint64_t>(static_cast<uint64_t>(w64), MAX_BLOCK_W), static_cast<uint64_t>((w64 + 1) / 2)));
        const int32_t bandCandStart = bandCandMax / VECTOR_ALIGN_ELEMENTS * VECTOR_ALIGN_ELEMENTS;
        for (int32_t candidate = bandCandStart; candidate >= static_cast<int32_t>(MIN_BAND_W);
             candidate -= VECTOR_ALIGN_ELEMENTS) {
            // 仅允许整除 W 的带宽：保证所有带等宽（无短末带），DataCopyPad 的行内
            // padding 恒为常数且远小于 255，行距与表几何完全对齐。
            if (w64 % candidate != 0) {
                continue;
            }
            // Banded 2-D DataCopyPad candidates keep per-row blockLen <= 4096B. The full-width
            // strip path instead uses blockCount <= 4094 (DATA_COPY_MAX_BLOCK_COUNT - 2) and
            // blockLen = W * sizeof(TYPE_X) (<= blockW * sizeof(TYPE_X)), which the uint32
            // blockLen field supports without this 4096B bound.
            // 输入按最大带宽 bw+9、输出按 bw*sizeof(float) 检查。
            if ((static_cast<uint64_t>(candidate) + 9U) * inputBytes > 4096U ||
                static_cast<uint64_t>(candidate) * sizeof(float) > 4096U) {
                continue;
            }
            // 左 halo 固定 8 列（Vector 操作数偏移 32B 对齐）；pitch 取 bw+16 级，
            // 配合首带向右多拷 8 列，使所有 DataCopyPad 单侧 padding ≤32 字节。
            const uint64_t bandShift = 8U;
            const uint64_t bRaw = AlignUp(static_cast<uint64_t>(candidate) + bandShift + 2U, inputAlignElements);
            const uint64_t bFloat = AlignUp(static_cast<uint64_t>(candidate) + bandShift + 2U, VECTOR_ALIGN_ELEMENTS);
            const uint64_t bOut = AlignUp(static_cast<uint64_t>(candidate) + bandShift, 8U);
            const uint64_t bPitch = bFloat;
            const uint64_t bCached = (inputType == ge::DT_FLOAT) ? bRaw : bFloat;
            const uint64_t bandTableBytes = 4U * bPitch * sizeof(uint32_t);
            const bool enableBandTail = !isXDirection && needSoftwareFma && (w64 & 1) != 0;
            const uint64_t bandResident = bRaw * inputBytes +
                                          ((inputType == ge::DT_FLOAT) ? 0U : bFloat * sizeof(float));
            const uint64_t bandOutBytes = bOut * sizeof(float);
            uint64_t bestBandUtility = 0;
            int32_t bestBandTileH = 0;
            int32_t bestBandK = 0;
            uint64_t bestBandCompPitch = 0;
            ScratchLayout bestBandScratch{};
            for (int32_t candidateRows : {32, 16, 8, 4, 2}) {
                const uint64_t candidateCompPitch = std::max<uint64_t>(bCached, bPitch);
                if (needSoftwareFma &&
                    static_cast<uint64_t>(candidateRows) * candidateCompPitch > VECTOR_MAX_COUNT_PER_CALL) {
                    continue;
                }
                const uint32_t layoutRows = needSoftwareFma ? static_cast<uint32_t>(candidateRows) : 0U;
                const ScratchLayout candScratch = MakeScratchLayout(bPitch, needSoftwareFma, needUint8Cast,
                                                                    enableBandTail, layoutRows, bCached, isXDirection,
                                                                    bandTableBytes);
                const uint64_t fixedBytes = candScratch.bytes + 2U * bandResident;
                if (usableUb <= fixedBytes) {
                    continue;
                }
                const uint64_t capacityRows = (usableUb - fixedBytes) / (bandResident + bandOutBytes);
                const uint64_t selectedRows = std::min<uint64_t>(static_cast<uint64_t>(h64),
                                                                 std::min<uint64_t>(capacityRows, maxBlockCountRows));
                if (selectedRows < requiredRows) {
                    continue;
                }
                // strided 2D DataCopyPad（srcStride>0）的 blockCount 取保守上限：
                // blockCount = tileH + 2 ≤ 14。截断必须在 utility 比较之前完成，
                // 否则会高估大批次候选的实际收益。
                const uint64_t cappedRows = std::min<uint64_t>(selectedRows, 12U);
                const uint64_t utility = needSoftwareFma ? static_cast<uint64_t>(candidateRows) * cappedRows : 1U;
                if (utility <= bestBandUtility) {
                    continue;
                }
                bestBandUtility = utility;
                bestBandTileH = static_cast<int32_t>(cappedRows);
                bestBandK = candidateRows;
                bestBandCompPitch = candidateCompPitch;
                bestBandScratch = candScratch;
            }
            if (bestBandTileH > 0) {
                bandW = candidate;
                bandPitch = static_cast<int32_t>(bPitch);
                bandRawPitch = static_cast<int32_t>(bRaw);
                bandFloatPitch = static_cast<int32_t>(bFloat);
                bandOutPitch = static_cast<int32_t>(bOut);
                tileH = bestBandTileH;
                compRows = needSoftwareFma ? bestBandK : 0;
                compPitch = needSoftwareFma ? bestBandCompPitch : bPitch;
                vectorPitch = bPitch;
                inputSlabPitch = bRaw;
                floatSlabPitch = bFloat;
                outputSlabPitch = bOut;
                scratch = bestBandScratch;
                inputBufferBytes = (static_cast<uint64_t>(bestBandTileH) + 2U) * bRaw * inputBytes;
                outputBufferBytes = static_cast<uint64_t>(bestBandTileH) * bOut * sizeof(float);
                floatSlabBytes = (static_cast<uint64_t>(bestBandTileH) + 2U) *
                                 ((inputType == ge::DT_FLOAT) ? 0U : bFloat * sizeof(float));
                bandTableOffset = bestBandScratch.bandTable;
                break;
            }
        }
    }
    if (tileH == 0) {
        compPitch = vectorPitch;
    }

    if (bandW == 0 && ((inputSlabPitch - static_cast<uint64_t>(w64)) * inputBytes > UB_ALIGN_BYTES ||
                       (tileH > 0 && static_cast<uint64_t>(tileH) + 2U > DATA_COPY_MAX_BLOCK_COUNT))) {
        return ge::GRAPH_FAILED;
    }

    // 将最终选中的布局与任务划分序列化；host 和 kernel 必须同时更新。
    ScharrTilingData* tiling = context->GetTilingData<ScharrTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (memset_s(tiling, sizeof(ScharrTilingData), 0, sizeof(ScharrTilingData)) != EOK) {
        return ge::GRAPH_FAILED;
    }
    tiling->totalLength = static_cast<int32_t>(totalLength64);
    tiling->H = static_cast<int32_t>(h64);
    tiling->W = static_cast<int32_t>(w64);
    tiling->batchCount = static_cast<int32_t>(batchCount64);
    tiling->usedCoreNum = usedCoreNum;
    tiling->splitMode = splitMode;
    tiling->baseRowsPerCore = baseRowsPerCore;
    tiling->extraRowCoreCount = extraRowCoreCount;
    tiling->basePixelsPerCore = basePixelsPerCore;
    tiling->extraPixelCoreCount = extraPixelCoreCount;
    tiling->basePlanesPerCore = basePlanesPerCore;
    tiling->extraPlaneCoreCount = extraPlaneCoreCount;
    tiling->blockW = blockW;
    tiling->tileH = tileH;
    tiling->compRows = compRows;
    tiling->compPitch = static_cast<int32_t>(compPitch);
    tiling->bandW = bandW;
    tiling->rawPitch = static_cast<int32_t>(rawPitch);
    tiling->vectorPitch = static_cast<int32_t>(vectorPitch);
    tiling->inputSlabPitch = static_cast<int32_t>(inputSlabPitch);
    tiling->floatSlabPitch = static_cast<int32_t>(floatSlabPitch);
    tiling->outputSlabPitch = static_cast<int32_t>(outputSlabPitch);
    tiling->inputBufferBytes = static_cast<uint32_t>(inputBufferBytes);
    tiling->outputBufferBytes = static_cast<uint32_t>(outputBufferBytes);
    tiling->floatSlabBytes = static_cast<uint32_t>(floatSlabBytes);
    tiling->scratchBytes = scratch.bytes;
    tiling->slotsOffset = scratch.slots;
    tiling->work0Offset = scratch.work0;
    tiling->work1Offset = scratch.work1;
    tiling->comp0Offset = scratch.comp0;
    tiling->comp1Offset = scratch.comp1;
    tiling->comp2Offset = scratch.comp2;
    tiling->fallbackOffset = scratch.fallback;
    tiling->maskOffset = scratch.mask;
    tiling->castHalfOffset = scratch.castHalf;
    tiling->offset0Offset = scratch.offset0;
    tiling->offset1Offset = scratch.offset1;
    tiling->coefficientOffset = scratch.coefficient;
    tiling->tailBatchOffset = scratch.tailBatch;
    tiling->bandTableOffset = bandTableOffset;
    tiling->direction = dirIdx;
    tiling->border = borderIdx;

    context->SetTilingKey(GET_TPL_TILING_KEY(SCHARR_SCH_MODE_DEFAULT));
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // The vector kernel uses Ascend C library APIs. Reserve their system
    // workspace instead of inheriting the scalar implementation's value 0;
    // otherwise the failure may only surface asynchronously at stream sync.
    workspaceSizes[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForScharr([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// tiling娉ㄥ唽鍏ュ彛
IMPL_OP_OPTILING(Scharr).Tiling(ScharrTilingFunc).TilingParse<ScharrCompileInfo>(TilingParseForScharr);
} // namespace optiling
