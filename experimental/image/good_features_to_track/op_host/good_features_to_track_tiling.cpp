/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include "log/log.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "experimental/image/good_features_to_track/op_kernel/good_features_to_track_tiling_data.h"

namespace optiling {
namespace {
constexpr uint32_t kImageRank = 3;
constexpr uint32_t kMaxCornersAttr = 0;
constexpr uint32_t kQualityLevelAttr = 1;
constexpr uint32_t kMinDistanceAttr = 2;
constexpr uint32_t kBlockSizeAttr = 3;
constexpr uint32_t kUseHarrisAttr = 4;
constexpr uint32_t kHarrisKAttr = 5;
constexpr int64_t kDefaultMaxCorners = 1000;
constexpr float kDefaultQualityLevel = 0.01F;
constexpr float kDefaultMinDistance = 1.0F;
constexpr int64_t kDefaultBlockSize = 3;
constexpr float kDefaultHarrisK = 0.04F;

struct GoodFeaturesToTrackCompileInfo {};

static ge::graphStatus GoodFeaturesToTrackTilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape& imageShape = inputShape->GetStorageShape();
    OP_CHECK_IF(imageShape.GetDimNum() != kImageRank, OP_LOGE(context, "image must be [N,H,W]"),
                return ge::GRAPH_FAILED);
    const int64_t batch = imageShape.GetDim(0);
    const int64_t height = imageShape.GetDim(1);
    const int64_t width = imageShape.GetDim(2);
    OP_CHECK_IF(batch <= 0 || height <= 0 || width <= 0, OP_LOGE(context, "invalid image shape"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(height > UINT32_MAX || width > UINT32_MAX || height * width > UINT32_MAX,
                OP_LOGE(context, "image shape exceeds uint32"), return ge::GRAPH_FAILED);
    // 行搬运用 DataCopyParams.blockLen（单位：字节，上限 65535）。response 缓冲恒为 float，
    // 单行占 4*W 字节：W >= 16384 时 blockLen 会被 uint16 截断（W=16384 恰为 0，整行不搬）；
    // 且该宽度域 fullRowBytes > 65535 会使 useUb=false 走回退路径，那里没有其他宽度防护。
    constexpr int64_t kMaxBlockLenBytes = 65535;
    OP_CHECK_IF(width * static_cast<int64_t>(sizeof(float)) > kMaxBlockLenBytes,
                OP_LOGE(context, "width %ld exceeds row-copy limit: 4*W must be <= %ld", width, kMaxBlockLenBytes),
                return ge::GRAPH_FAILED);
    // 阶段 2b 的键直方图用 fp32 累加计数：fp32 只能精确表示 <= 2^24 的整数，超出后
    // 单桶计数与跨核求和会舍入，b*/b** 边界桶选取偏移 ⇒ 保留集偏离精确 Top-K。
    constexpr int64_t kMaxHistExactCount = 1LL << 24;
    OP_CHECK_IF(
        height * width > kMaxHistExactCount,
        OP_LOGE(context, "H*W = %ld exceeds histogram exact-count limit %ld", height * width, kMaxHistExactCount),
        return ge::GRAPH_FAILED);

    // 属性解析（对齐 OpenCV goodFeaturesToTrack）
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t maxCorners = attrs->GetInt(kMaxCornersAttr) == nullptr ? kDefaultMaxCorners :
                                                                           *attrs->GetInt(kMaxCornersAttr);
    const float quality = attrs->GetFloat(kQualityLevelAttr) == nullptr ? kDefaultQualityLevel :
                                                                          *attrs->GetFloat(kQualityLevelAttr);
    const float minDistance = attrs->GetFloat(kMinDistanceAttr) == nullptr ? kDefaultMinDistance :
                                                                             *attrs->GetFloat(kMinDistanceAttr);
    const int64_t blockSize = attrs->GetInt(kBlockSizeAttr) == nullptr ? kDefaultBlockSize :
                                                                         *attrs->GetInt(kBlockSizeAttr);
    const bool useHarris = attrs->GetBool(kUseHarrisAttr) != nullptr && *attrs->GetBool(kUseHarrisAttr);
    const float harrisK = attrs->GetFloat(kHarrisKAttr) == nullptr ? kDefaultHarrisK : *attrs->GetFloat(kHarrisKAttr);
    OP_CHECK_IF(quality <= 0.0F || quality > 1.0F || minDistance < 0.0F || blockSize <= 0 || (blockSize & 1) == 0 ||
                    harrisK < 0.0F,
                OP_LOGE(context, "invalid attributes"), return ge::GRAPH_FAILED);

    // 输入 dtype 校验：FLOAT16 / FLOAT32
    const auto imageDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, imageDesc);
    OP_CHECK_IF(imageDesc->GetDataType() != ge::DT_FLOAT16 && imageDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context, "image must be FLOAT16 or FLOAT32"), return ge::GRAPH_FAILED);

    // 可选掩码校验：与 image 同 shape，UINT8
    const gert::StorageShape* maskShape = context->GetInputShape(1);
    const bool hasMask = maskShape != nullptr;
    if (hasMask) {
        const gert::Shape& maskStorage = maskShape->GetStorageShape();
        OP_CHECK_IF(maskStorage.GetDimNum() != kImageRank || maskStorage.GetDim(0) != batch ||
                        maskStorage.GetDim(1) != height || maskStorage.GetDim(2) != width,
                    OP_LOGE(context, "mask must match image shape"), return ge::GRAPH_FAILED);
        const auto maskDesc = context->GetInputDesc(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, maskDesc);
        OP_CHECK_IF(maskDesc->GetDataType() != ge::DT_UINT8, OP_LOGE(context, "mask must be UINT8"),
                    return ge::GRAPH_FAILED);
    }

    // 多核：响应计算按像素均匀分发
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t availableCores = platform.GetCoreNumAiv();
    const uint64_t totalPixels = static_cast<uint64_t>(batch) * static_cast<uint64_t>(height) *
                                 static_cast<uint64_t>(width);
    const uint32_t usedCores = static_cast<uint32_t>(totalPixels < availableCores ? totalPixels : availableCores);
    OP_CHECK_IF(usedCores == 0, OP_LOGE(context, "no available AIV core"), return ge::GRAPH_FAILED);

    GoodFeaturesToTrackTilingData* tiling = context->GetTilingData<GoodFeaturesToTrackTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(*tiling), 0, sizeof(*tiling)) != EOK, OP_LOGE(context, "clear tiling failed"),
                return ge::GRAPH_FAILED);

    tiling->batch = static_cast<uint32_t>(batch);
    tiling->coreNum = usedCores;
    tiling->height = static_cast<uint32_t>(height);
    tiling->width = static_cast<uint32_t>(width);
    tiling->imageElements = static_cast<uint32_t>(height * width);
    tiling->outputCapacity = static_cast<uint32_t>(maxCorners > 0 ? maxCorners : height * width);
    tiling->blockSize = static_cast<int32_t>(blockSize);
    tiling->qualityLevel = quality;
    tiling->minDistance = minDistance;
    tiling->harrisK = harrisK;
    tiling->useHarris = useHarris ? 1U : 0U;
    tiling->hasMask = hasMask ? 1U : 0U;

    // workspace 布局：
    //   [0, libSize)                        框架预留段（GetLibApiWorkSpaceSize 要求，本算子不读写；
    //                                       本算子用无参硬件 SyncAll，不需要同步 workspace）
    //   [libSize, +batch*H*W*4)             response 缓冲（每像素 float）
    //   [+batch*H*W*4, +batch*coreNum*8*4)  每核「每图」统计槽（[0]=该图局部 max、[1]=该核在该图的候选数）
    //   [+batch*coreNum*8*4, +coreNum*256*4) 每核候选键高 8 位直方图（并行 TopK）
    //   [+coreNum*256*4, +(2*coreNum+8)*4)  并行 TopK 元数据：[0]=b*、[1]=b**、
    //                                       [2+c]=核 c 「小于组」偏移、[2+coreNum+c]=「等于组」偏移、
    //                                       [2+2*coreNum]=小于组总数、[2+2*coreNum+1]=等于组保留数、
    //                                       [2+2*coreNum+2]=保留总数
    //   [+(2*coreNum+8)*4, +coreNum*256*4)  每核二级直方图（键 bit16-23，仅 bucket==b* 的候选）
    //   [+coreNum*256*4, +batch*H*W*4)      候选 idx 主缓冲（uint32，各核写自己分区）
    //   [+batch*H*W*4, +batch*H*W*4)        候选 val 主缓冲（float）
    //   [+batch*H*W*4, +batch*H*W*4)        候选 idx 保留集缓冲（并行过滤的输出 / 基数排序乒乓）
    //   [+batch*H*W*4, +batch*H*W*4)        候选 val 保留集缓冲
    // 注：保留集单独用一对缓冲，避免「各核并行过滤」时输出区间与别的核的输入分区重叠（数据竞争）。
    // 注：原 flag 缓冲（1B/像素）为纯写不读的死代码，已移除。
    const size_t respBytes = static_cast<size_t>(batch) * height * width * sizeof(float);
    // 统计量按图存放：[batch][core] × 8 float（kernel 端 StatSlot() 与此同源；逐图阈值用）
    const size_t maxPerCoreBytes = static_cast<size_t>(batch) * usedCores * 8U * sizeof(float);
    const size_t keyHistBytes = static_cast<size_t>(usedCores) * 256U * sizeof(int32_t);
    // 2*coreNum+8 个 int32（8*coreNum+32 字节，天然 32B 对齐）；必须与 kernel 端完全一致
    const size_t topkMetaBytes = (2U * static_cast<size_t>(usedCores) + 8U) * sizeof(int32_t);
    const size_t subHistBytes = static_cast<size_t>(usedCores) * 256U * sizeof(int32_t);
    const size_t candBytes = static_cast<size_t>(batch) * height * width * sizeof(uint32_t);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = static_cast<size_t>(platform.GetLibApiWorkSpaceSize()) + respBytes + maxPerCoreBytes +
                    keyHistBytes + topkMetaBytes + subHistBytes + candBytes * 4U;

    context->SetBlockDim(usedCores);
    context->SetScheduleMode(1);
    return ge::GRAPH_SUCCESS;
}
} // namespace

static ge::graphStatus TilingParseForGoodFeaturesToTrack([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GoodFeaturesToTrack)
    .Tiling(GoodFeaturesToTrackTilingFunc)
    .TilingParse<GoodFeaturesToTrackCompileInfo>(TilingParseForGoodFeaturesToTrack);
} // namespace optiling
