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
 * \file crop_and_resize_tiling_arch35.cpp
 * \brief Tiling implementation for crop_and_resize operator
 */

#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "../crop_and_resize_constraints.h"
#include "image/crop_and_resize/op_kernel/arch35/crop_and_resize_tiling_data.h"
#include "image/crop_and_resize/op_kernel/arch35/crop_and_resize_tiling_key.h"

#include <limits>
#include <string>

namespace optiling {

// tiling 专属常量（非约束阈值，不与 def/infershape 共享）
// warp 级并行：并行域为每 warp 1 个输出位置（kernel 内 lane 分摊通道维，d ≡ laneId mod 32），
// 每核 warp 位置数下限 32：uint32 路径恰好占满 1 个 block
// （1024 线程 = 32 warp，与 kernel 侧 WARP_LANES=32 对应）
constexpr int64_t WARP_PER_CORE_MIN = 32;
constexpr uint32_t DCACHE_SIZE = 128 * 1024;
constexpr uint32_t STATIC_UB_ESTIMATE = 0;

// 结构/语义常量来自 op_host/crop_and_resize_constraints.h（与 def.cpp/infershape.cpp 共享）

struct CropAndResizeCompileInfo {};

// 输入信息结构体，用于在约束检查和 tiling 计算间传递
struct CropAndResizeInputInfo {
    int64_t batch = 0;
    int64_t imageHeight = 0;
    int64_t imageWidth = 0;
    int64_t depth = 0;
    int64_t numBoxes = 0;
    int64_t cropHeight = 0;
    int64_t cropWidth = 0;
    float extrapolationValue = 0.0f;
    ge::DataType xDtype = ge::DT_UNDEFINED;
    bool isNchw = false; // x format 是否为 NCHW：决定 dims→H/W/C 解析来源与 TilingKey 分流
    bool isNearest = false;
};

// 非负 int64 乘法溢出检查
static bool MulOverflowInt64(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0) {
        return true;
    }
    if (lhs == 0 || rhs == 0) {
        result = 0;
        return false;
    }
    if (lhs > std::numeric_limits<int64_t>::max() / rhs) {
        return true;
    }
    result = lhs * rhs;
    return false;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum,
                                       uint64_t& sysWorkspaceSize)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

// 约束 7：dtype 检查（x/boxes/box_index）
static ge::graphStatus CheckInputDtypes(gert::TilingContext* context, const CropAndResizeInputInfo& info)
{
    if (info.xDtype != ge::DT_FLOAT && info.xDtype != ge::DT_FLOAT16 && info.xDtype != ge::DT_UINT8 &&
        info.xDtype != ge::DT_UINT16 && info.xDtype != ge::DT_INT8 && info.xDtype != ge::DT_INT16 &&
        info.xDtype != ge::DT_INT32 && info.xDtype != ge::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(info.xDtype).c_str(),
                                  "FLOAT/FLOAT16/UINT8/UINT16/INT8/INT16/INT32/INT64");
        return ge::GRAPH_FAILED;
    }
    ge::DataType boxesDtype = context->GetInputDesc(IDX_BOXES)->GetDataType();
    if (boxesDtype != ge::DT_FLOAT && boxesDtype != ge::DT_FLOAT16) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "boxes", Ops::Base::ToString(boxesDtype).c_str(),
                                  "FLOAT16/FLOAT");
        return ge::GRAPH_FAILED;
    }
    ge::DataType boxIndexDtype = context->GetInputDesc(IDX_BOX_INDEX)->GetDataType();
    if (boxIndexDtype != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "box_index", Ops::Base::ToString(boxIndexDtype).c_str(),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 语义/技术约束：保留正维度、结构一致性
static ge::graphStatus CheckShapeConstraints(gert::TilingContext* context, const CropAndResizeInputInfo& info)
{
    if (info.batch <= 0 || info.imageHeight <= 0 || info.imageWidth <= 0 || info.depth <= 0) {
        std::string valMsg = "[" + std::to_string(info.batch) + ", " + std::to_string(info.imageHeight) + ", " +
                             std::to_string(info.imageWidth) + ", " + std::to_string(info.depth) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", valMsg.c_str(), "all dims must be positive");
        return ge::GRAPH_FAILED;
    }
    if (info.cropHeight <= 0 || info.cropWidth <= 0) {
        std::string valMsg = "[" + std::to_string(info.cropHeight) + ", " + std::to_string(info.cropWidth) + "]";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "crop_size", valMsg.c_str(),
                                              "crop_height and crop_width must be positive");
        return ge::GRAPH_FAILED;
    }
    if (info.numBoxes < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "boxes", std::to_string(info.numBoxes).c_str(),
                                              "num_boxes must be non-negative");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(CheckInputDtypes(context, info) != ge::GRAPH_SUCCESS, OP_LOGE(context, "dtype check failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckSafetyConstraints(gert::TilingContext* context)
{
    auto boxesShapePtr = context->GetInputShape(IDX_BOXES);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesShapePtr);
    auto boxIndexShapePtr = context->GetInputShape(IDX_BOX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxIndexShapePtr);
    auto boxesShape = boxesShapePtr->GetStorageShape();
    auto boxIndexShape = boxIndexShapePtr->GetStorageShape();
    if (boxIndexShape.GetDimNum() != BOX_INDEX_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "box_index",
                                     (std::to_string(boxIndexShape.GetDimNum()) + "D").c_str(), "1D");
        return ge::GRAPH_FAILED;
    }
    if (boxesShape.GetDim(0) != boxIndexShape.GetDim(0)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "boxes and box_index",
                                               std::to_string(boxesShape.GetDim(0)).c_str(),
                                               "boxes.shape[0] must equal box_index.shape[0]");
        return ge::GRAPH_FAILED; // 约束10
    }
    if (boxesShape.GetDim(1) != BOX_COORDS) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context->GetNodeName(), "boxes", std::to_string(boxesShape.GetDim(1)).c_str(), "boxes.shape[1] must be 4");
        return ge::GRAPH_FAILED; // 约束11
    }

    // 约束13: 非有限 boxes 处理已移至 kernel 运行时（逐位置判断，填 extrapolationValue）
    // 原因：tiling 阶段在二进制/动态/常量编译模式下 boxes 数据不可用（nullptr），导致 OPTILING_FAILURE

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForCropAndResize([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// 提取输入 format + shape + crop_size 值 + dtype
static ge::graphStatus ExtractInputInfo(gert::TilingContext* context, CropAndResizeInputInfo& info)
{
    auto xShapePtr = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    auto boxesShapePtr = context->GetInputShape(IDX_BOXES);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesShapePtr);
    auto xShape = xShapePtr->GetStorageShape();
    auto boxesShape = boxesShapePtr->GetStorageShape();

    // 读 x format，非 ND/NHWC/NCHW 拒绝（防御性兜底，与 infershape 拦截集合一致）
    auto xDescPtr = context->GetInputDesc(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDescPtr);
    ge::Format xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDescPtr->GetStorageFormat()));
    OP_CHECK_IF(xFormat != ge::FORMAT_NCHW && xFormat != ge::FORMAT_NHWC && xFormat != ge::FORMAT_ND,
                OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "x", Ops::Base::ToString(xFormat).c_str(),
                                           "NCHW, NHWC and ND"),
                return ge::GRAPH_FAILED);
    info.isNchw = (xFormat == ge::FORMAT_NCHW);

    if (xShape.GetDimNum() != X_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", (std::to_string(xShape.GetDimNum()) + "D").c_str(),
                                     "4D");
        return ge::GRAPH_FAILED;
    }
    info.batch = xShape.GetDim(0);
    if (info.isNchw) {
        // NCHW: x = (N, C, H, W)
        info.depth = xShape.GetDim(1);       // C
        info.imageHeight = xShape.GetDim(2); // H
        info.imageWidth = xShape.GetDim(3);  // W
    } else {
        // ND/NHWC: x = (N, H, W, C)（现状解析，零回归）
        info.imageHeight = xShape.GetDim(1); // H
        info.imageWidth = xShape.GetDim(2);  // W
        info.depth = xShape.GetDim(3);       // C
    }
    if (boxesShape.GetDimNum() != BOXES_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "boxes",
                                     (std::to_string(boxesShape.GetDimNum()) + "D").c_str(), "2D");
        return ge::GRAPH_FAILED;
    }
    info.numBoxes = boxesShape.GetDim(0);
    info.xDtype = context->GetInputDesc(IDX_X)->GetDataType();

    auto cropSizeShapePtr = context->GetInputShape(IDX_CROP_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, cropSizeShapePtr);
    auto cropSizeShape = cropSizeShapePtr->GetStorageShape();
    if (cropSizeShape.GetDimNum() != CROP_SIZE_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "crop_size",
                                     (std::to_string(cropSizeShape.GetDimNum()) + "D").c_str(), "1D");
        return ge::GRAPH_FAILED;
    }
    if (cropSizeShape.GetDim(0) != CROP_SIZE_LEN) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "crop_size",
                                                  std::to_string(cropSizeShape.GetDim(0)).c_str(),
                                                  "crop_size.shape[0] must be 2");
        return ge::GRAPH_FAILED;
    }

    ge::DataType cropSizeDtype = context->GetInputDesc(IDX_CROP_SIZE)->GetDataType();
    if (cropSizeDtype != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "crop_size", Ops::Base::ToString(cropSizeDtype).c_str(),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }

    const gert::Tensor* cropSizeTensor = context->GetInputTensor(IDX_CROP_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, cropSizeTensor);
    const int32_t* cropSizeData = cropSizeTensor->GetData<int32_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, cropSizeData);
    info.cropHeight = cropSizeData[0];
    info.cropWidth = cropSizeData[1];
    return ge::GRAPH_SUCCESS;
}

// 提取属性 extrapolation_value + method 校验
static ge::graphStatus ExtractAttrs(gert::TilingContext* context, CropAndResizeInputInfo& info)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const float* extrapolationValuePtr = attrs->GetFloat(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, extrapolationValuePtr);
    info.extrapolationValue = *extrapolationValuePtr;
    const char* methodStr = attrs->GetStr(1);
    OP_CHECK_IF(methodStr == nullptr, OP_LOGE(context, "method attr is null"), return ge::GRAPH_FAILED);
    std::string method(methodStr);
    if (method != "bilinear" && method != "nearest") {
        OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), "method", methodStr, "bilinear/nearest");
        return ge::GRAPH_FAILED;
    }
    info.isNearest = (method == "nearest");
    return ge::GRAPH_SUCCESS;
}

// 核数切分 + 设置 TilingData + 设置 BlockDim/Workspace
static ge::graphStatus ComputeAndSetTiling(gert::TilingContext* context, const CropAndResizeInputInfo& info,
                                           int64_t coreNum, uint64_t ubSize, uint64_t sysWorkspaceSize)
{
    // warp 级并行域换算：totalPositions 即 warp 位置总数（每 warp 处理一个输出位置），
    // 两步法：先按物理核均分得到每核 warp 位置数（下限 WARP_PER_CORE_MIN），再反推实际核数
    int64_t cropHW = 0;
    int64_t totalPositions = 0;
    if (MulOverflowInt64(info.cropHeight, info.cropWidth, cropHW) ||
        MulOverflowInt64(info.numBoxes, cropHW, totalPositions)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "shape", "total positions overflow",
                                              "num_boxes * crop_h * crop_w must fit int64");
        return ge::GRAPH_FAILED;
    }
    int64_t hw = 0;
    int64_t imgHWC = 0;
    int64_t cropHWC = 0;
    if (MulOverflowInt64(info.imageHeight, info.imageWidth, hw) || MulOverflowInt64(hw, info.depth, imgHWC) ||
        MulOverflowInt64(cropHW, info.depth, cropHWC)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "shape", "addressing product overflow",
                                              "H*W*C and crop_h*crop_w*C must fit int64");
        return ge::GRAPH_FAILED;
    }
    (void)imgHWC;
    (void)cropHWC;

    int64_t perCoreWarps = Ops::Base::CeilDiv(totalPositions, coreNum);
    if (perCoreWarps < WARP_PER_CORE_MIN) {
        perCoreWarps = WARP_PER_CORE_MIN;
    }
    int64_t needCoreNum = Ops::Base::CeilDiv(totalPositions, perCoreWarps);
    // 空 boxes: totalPositions=0 时仍需启动 1 个 block，kernel 入口直接返回
    if (needCoreNum < 1) {
        needCoreNum = 1;
    }

    CropAndResizeTilingData* tiling = context->GetTilingData<CropAndResizeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(CropAndResizeTilingData), 0, sizeof(CropAndResizeTilingData)) != EOK,
                OP_LOGE(context, "memset_s tiling data error"), return ge::GRAPH_FAILED);
    tiling->totalPositions = totalPositions;
    tiling->batch = info.batch;
    tiling->imageHeight = info.imageHeight;
    tiling->imageWidth = info.imageWidth;
    tiling->depth = info.depth;
    tiling->cropHeight = info.cropHeight;
    tiling->cropWidth = info.cropWidth;
    tiling->numBoxes = info.numBoxes;
    tiling->extrapolationValue = info.extrapolationValue;

    context->SetBlockDim(static_cast<uint32_t>(needCoreNum));
    OP_CHECK_IF(ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE, OP_LOGE(context, "ubSize too small"),
                return ge::GRAPH_FAILED);
    context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    // layout 经 TilingKey 编码为 kernel 模板参数（NHWC 值 0 与原单值模式二进制兼容，ND/NHWC 路径零回归）
    // 三目结果显式 uint64_t 承接，避免花括号初始化列表 narrowing 报错
    uint64_t schMode = 0;
    if (info.isNearest) {
        schMode = info.isNchw ? static_cast<uint64_t>(CROP_AND_RESIZE_MODE_NEAREST_NCHW) :
                                static_cast<uint64_t>(CROP_AND_RESIZE_MODE_NEAREST_NHWC);
    } else {
        schMode = info.isNchw ? static_cast<uint64_t>(CROP_AND_RESIZE_MODE_BILINEAR_NCHW) :
                                static_cast<uint64_t>(CROP_AND_RESIZE_MODE_BILINEAR_NHWC);
    }
    context->SetTilingKey(GET_TPL_TILING_KEY(schMode));
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    int64_t userWorkspaceSize = 0;
    currentWorkspace[0] = static_cast<size_t>(userWorkspaceSize + static_cast<int64_t>(sysWorkspaceSize));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CropAndResizeTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    uint64_t sysWorkspaceSize = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum, sysWorkspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    CropAndResizeInputInfo info;
    OP_CHECK_IF(ExtractInputInfo(context, info) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ExtractInputInfo failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ExtractAttrs(context, info) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ExtractAttrs failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckShapeConstraints(context, info) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "shape constraints check failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckSafetyConstraints(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "safety constraints check failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeAndSetTiling(context, info, coreNum, ubSize, sysWorkspaceSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ComputeAndSetTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// Tiling 注册：三段式 + 值依赖（crop_size index=3 shape推导；非有限 boxes 处理已移至 kernel 运行时）
IMPL_OP_OPTILING(CropAndResize)
    .Tiling(CropAndResizeTilingFunc)
    .TilingParse<CropAndResizeCompileInfo>(TilingParseForCropAndResize)
    .TilingInputsDataDependency({IDX_CROP_SIZE});
} // namespace optiling
