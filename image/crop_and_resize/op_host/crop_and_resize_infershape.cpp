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
 * \file crop_and_resize_infershape.cpp
 * \brief InferShape implementation for crop_and_resize operator
 *
 * Output shape:
 *   ND/NHWC x: [num_boxes, crop_height, crop_width, depth], depth = x.shape[3]
 *   NCHW x:    [num_boxes, depth, crop_height, crop_width], depth = x.shape[1]
 * Output dtype = boxes dtype (handled by def.cpp DataType config)
 *
 * Unknown rank inputs are normalized to expected-rank all-UNKNOWN_DIM shapes
 * at entry (v1 WithRank paradigm), so validation/derivation only sees -1.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "crop_and_resize_constraints.h"

#include <string>

using namespace ge;

namespace ops {

// 约束阈值常量和维度/索引常量来自 crop_and_resize_constraints.h

// unknown rank 归一化（对齐 v1 WithRank 语义）：按预期 rank 展开为全 UNKNOWN_DIM，
// 后续校验与推导统一面对 -1（未知维度），不再区分 -1/-2 两种未知表示
static gert::Shape NormalizeUnknownRank(const gert::Shape& shape, size_t expectRank)
{
    if (!Ops::Base::IsUnknownRank(shape)) {
        return shape;
    }
    gert::Shape normalized;
    normalized.SetDimNum(expectRank);
    for (size_t i = 0; i < expectRank; i++) {
        normalized.SetDim(i, ge::UNKNOWN_DIM);
    }
    return normalized;
}

// 校验输入 shape（入口已归一化 unknown rank 为全 -1，本函数统一按 UNKNOWN_DIM 豁免）：
// 维度数、x 逐维非 0、boxes 形状与 box_index 一致性、crop_size 形状
static ge::graphStatus ValidateInputShapes(gert::InferShapeContext* context, const gert::Shape& xShape,
                                           const gert::Shape& boxesShape, const gert::Shape& boxIndexShape,
                                           const gert::Shape& cropSizeShape)
{
    if (xShape.GetDimNum() != X_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", (std::to_string(xShape.GetDimNum()) + "D").c_str(),
                                     "4D");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        if (xShape.GetDim(i) == 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x", std::to_string(xShape.GetDim(i)).c_str(),
                                                  ("x.shape[" + std::to_string(i) + "] must not be zero").c_str());
            return ge::GRAPH_FAILED;
        }
    }
    if (boxesShape.GetDimNum() != BOXES_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "boxes",
                                     (std::to_string(boxesShape.GetDimNum()) + "D").c_str(), "2D");
        return ge::GRAPH_FAILED;
    }
    if (boxesShape.GetDim(1) != ge::UNKNOWN_DIM && boxesShape.GetDim(1) != BOX_COORDS) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            context->GetNodeName(), "boxes", std::to_string(boxesShape.GetDim(1)).c_str(), "boxes.shape[1] must be 4");
        return ge::GRAPH_FAILED;
    }
    // box_index rank 检查先行：0D 时 GetDim(0) 读脏值、2D 时 dim0 巧合相等可绕过一致性判定
    // （unknown rank 已归一化为 [-1]，rank=1 自然通过）
    if (boxIndexShape.GetDimNum() != BOX_INDEX_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "box_index",
                                     (std::to_string(boxIndexShape.GetDimNum()) + "D").c_str(), "1D");
        return ge::GRAPH_FAILED;
    }
    // boxes.shape[0]==0 允许：合法空输出
    // 约束10：boxes.shape[0] 必须等于 box_index.shape[0]（双方 UNKNOWN_DIM 豁免；
    // box_index unknown rank 已入口归一化为 -1，同样豁免，运行时校验，与 tiling 层双保险）
    if (boxesShape.GetDim(0) != ge::UNKNOWN_DIM && boxIndexShape.GetDim(0) != ge::UNKNOWN_DIM &&
        boxesShape.GetDim(0) != boxIndexShape.GetDim(0)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "boxes and box_index",
                                               std::to_string(boxIndexShape.GetDim(0)).c_str(),
                                               "boxes.shape[0] must equal box_index.shape[0]");
        return ge::GRAPH_FAILED;
    }
    if (cropSizeShape.GetDimNum() != CROP_SIZE_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "crop_size",
                                     (std::to_string(cropSizeShape.GetDimNum()) + "D").c_str(), "1D");
        return ge::GRAPH_FAILED;
    }
    // crop_size.shape[0]==UNKNOWN_DIM 允许（unknown rank 已归一化为 -1，一并覆盖）：
    // 运行时校验实际 shape，H/W 置 UNKNOWN_DIM
    if (cropSizeShape.GetDim(0) != CROP_SIZE_LEN && cropSizeShape.GetDim(0) != ge::UNKNOWN_DIM) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "crop_size",
                                                  std::to_string(cropSizeShape.GetDim(0)).c_str(),
                                                  "crop_size.shape[0] must be 2");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// 读取 crop_size 值并校验 dtype 和范围
// shape 未确定时跳过读值（防按固定长度误读），H/W 保持 UNKNOWN_DIM
static ge::graphStatus ReadCropSizeValue(gert::InferShapeContext* context, const gert::Shape* cropSizeShape,
                                         int64_t& cropHeight, int64_t& cropWidth)
{
    const gert::Tensor* cropSizeTensor = context->GetInputTensor(IDX_CROP_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, cropSizeTensor);

    ge::DataType cropSizeDtype = context->GetInputDesc(IDX_CROP_SIZE)->GetDataType();
    if (cropSizeDtype != ge::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "crop_size", Ops::Base::ToString(cropSizeDtype).c_str(),
                                  "INT32");
        return ge::GRAPH_FAILED;
    }

    cropHeight = ge::UNKNOWN_DIM;
    cropWidth = ge::UNKNOWN_DIM;

    if (cropSizeShape->GetDim(0) == ge::UNKNOWN_DIM) {
        OP_LOGD(context->GetNodeName(),
                "crop_size.shape[0] is UNKNOWN_DIM, skip reading value, keep output dims as UNKNOWN_DIM");
        return ge::GRAPH_SUCCESS;
    }

    if (cropSizeTensor->GetAddr() != nullptr) {
        const int32_t* cropSizeData = cropSizeTensor->GetData<int32_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, cropSizeData);
        cropHeight = static_cast<int64_t>(cropSizeData[0]);
        cropWidth = static_cast<int64_t>(cropSizeData[1]);

        if (cropHeight <= 0 || cropWidth <= 0) {
            std::string valMsg = "[" + std::to_string(cropHeight) + ", " + std::to_string(cropWidth) + "]";
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "crop_size", valMsg.c_str(),
                                                  "crop_height and crop_width must be positive");
            return ge::GRAPH_FAILED;
        }

    } else {
        OP_LOGD(context->GetNodeName(), "crop_size is non-const tensor, set output dims to UNKNOWN_DIM");
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeCropAndResize(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeCropAndResize");

    const gert::Shape* xShapePtr = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape* boxesShapePtr = context->GetInputShape(IDX_BOXES);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxesShapePtr);
    const gert::Shape* boxIndexShapePtr = context->GetInputShape(IDX_BOX_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, boxIndexShapePtr);
    const gert::Shape* cropSizeShapePtr = context->GetInputShape(IDX_CROP_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, cropSizeShapePtr);

    // 入口归一化：unknown rank 按预期 rank 展开为全 -1，后续统一处理
    const gert::Shape xShape = NormalizeUnknownRank(*xShapePtr, X_DIM);
    const gert::Shape boxesShape = NormalizeUnknownRank(*boxesShapePtr, BOXES_DIM);
    const gert::Shape boxIndexShape = NormalizeUnknownRank(*boxIndexShapePtr, BOX_INDEX_DIM);
    const gert::Shape cropSizeShape = NormalizeUnknownRank(*cropSizeShapePtr, CROP_SIZE_DIM);

    // 读 x format 并校验（ND/NHWC/NCHW 三分支持，其余拒绝）。调用链对齐 resize_bicubic_v2_infershape.cpp L40-44
    auto xDesc = context->GetInputDesc(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    ge::Format xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetFormat().GetStorageFormat()));
    OP_CHECK_IF(
        xFormat != ge::FORMAT_ND && xFormat != ge::FORMAT_NHWC && xFormat != ge::FORMAT_NCHW,
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "x", Ops::Base::ToString(xFormat).c_str(), "ND/NHWC/NCHW"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateInputShapes(context, xShape, boxesShape, boxIndexShape, cropSizeShape) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ValidateInputShapes failed"), return ge::GRAPH_FAILED);

    int64_t cropHeight = ge::UNKNOWN_DIM;
    int64_t cropWidth = ge::UNKNOWN_DIM;
    OP_CHECK_IF(ReadCropSizeValue(context, &cropSizeShape, cropHeight, cropWidth) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ReadCropSizeValue failed"), return ge::GRAPH_FAILED);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    yShape->SetDimNum(X_DIM);
    yShape->SetDim(0, boxesShape.GetDim(0)); // num_boxes（两分支相同）
    if (xFormat == ge::FORMAT_NCHW) {
        // NCHW: y = [num_boxes, C, crop_h, crop_w]（对齐 TBE image_ops.cc L253-257，数据不转置）
        yShape->SetDim(1, xShape.GetDim(1)); // C = x.shape[1]
        yShape->SetDim(2, cropHeight);
        yShape->SetDim(3, cropWidth);
    } else {
        // ND/NHWC: y = [num_boxes, crop_h, crop_w, C]（现状排布，零回归）
        yShape->SetDim(1, cropHeight);
        yShape->SetDim(2, cropWidth);
        yShape->SetDim(3, xShape.GetDim(3)); // C = x.shape[3]
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeCropAndResize");
    return GRAPH_SUCCESS;
}

// 注册 infershape + 值依赖（crop_size index=3）
IMPL_OP_INFERSHAPE(CropAndResize).InferShape(InferShapeCropAndResize).InputsDataDependency({IDX_CROP_SIZE});
} // namespace ops
