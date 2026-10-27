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
 * \file col2_im_v2_tiling.cpp
 * \brief Tiling implementation for col2_im_v2 operator
 */

#include <cstdint>
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "../../op_kernel/arch35/col2_im_v2_tiling_data.h"
#include "../../op_kernel/arch35/col2_im_v2_tiling_key.h"

namespace optiling {

constexpr int64_t PER_CORE_MIN = 1024;       // 单核最少处理元素数（32 的倍数，对齐 warp）
constexpr uint32_t DCACHE_SIZE = 128 * 1024; // DCache 预留 128KB
constexpr uint32_t STATIC_UB_ESTIMATE = 0;   // 无静态 UB
constexpr int64_t ATTR_VEC_LEN = 2;          // dilation/padding/stride 长度
constexpr int64_t SIZE_TENSOR_LEN = 2;       // output_size/kernel_size 长度
constexpr size_t X_RANK = 3;                 // x 维度数（x 仅支持 3-D）
constexpr size_t SIZE_TENSOR_RANK = 1;       // output_size/kernel_size 维度数（仅支持 1-D）

// ========== input index constants（与 REG_OP 原型输入/属性顺序一致）==========
static constexpr int32_t kXIdx = 0;
static constexpr int32_t kOutputSizeIdx = 1;
static constexpr int32_t kKernelSizeIdx = 2;
static constexpr int32_t kYIdx = 0;
static constexpr int32_t kDilationAttrIdx = 0;
static constexpr int32_t kPaddingAttrIdx = 1;
static constexpr int32_t kStrideAttrIdx = 2;

struct Col2ImV2CompileInfo {};

// 几何参数（host 侧传递用）
struct Col2ImV2Geom {
    int64_t outH = 0;
    int64_t outW = 0;
    int64_t kernelH = 0;
    int64_t kernelW = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
    int64_t paddingH = 0;
    int64_t paddingW = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t colH = 0;
    int64_t colW = 0;
    int64_t totalLength = 0;
};

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

// ========== ValidateDtype（dtype 单参数约束 + 跨参数 dtype 关系）==========
static ge::graphStatus ValidateDtype(gert::TilingContext* context)
{
    auto xDesc = context->GetInputDesc(kXIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    auto xDtype = xDesc->GetDataType();
    // x dtype 仅支持 float32/float16
    OP_CHECK_IF(
        xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16,
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(xDtype).c_str(), "FLOAT/FLOAT16"),
        return ge::GRAPH_FAILED);
    for (int32_t idx : {kOutputSizeIdx, kKernelSizeIdx}) {
        auto desc = context->GetInputDesc(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        // output_size/kernel_size dtype 固定 int32
        OP_CHECK_IF(
            desc->GetDataType() != ge::DT_INT32,
            OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), (idx == kOutputSizeIdx ? "output_size" : "kernel_size"),
                                      Ops::Base::ToString(desc->GetDataType()).c_str(), "INT32"),
            return ge::GRAPH_FAILED);
    }
    auto yDesc = context->GetOutputDesc(kYIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    // y.dtype 必须与 x.dtype 一致
    OP_CHECK_IF(yDesc->GetDataType() != xDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "x, y",
                    (Ops::Base::ToString(xDtype) + ", " + Ops::Base::ToString(yDesc->GetDataType())).c_str(),
                    "y.dtype must equal x.dtype"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ========== ValidateShape（shape 单参数约束）==========
static ge::graphStatus ValidateShape(gert::TilingContext* context)
{
    auto xInput = context->GetInputShape(kXIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, xInput);
    auto xShape = xInput->GetStorageShape();
    // x 仅支持 3-D
    OP_CHECK_IF(xShape.GetDimNum() != X_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x",
                                             (std::to_string(xShape.GetDimNum()) + "D").c_str(), "3D"),
                return ge::GRAPH_FAILED);
    // x 非 batch 维度必须非零（batch 维 n 允许为 0）
    OP_CHECK_IF(
        xShape.GetDim(1) <= 0 || xShape.GetDim(2) <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "x",
            ("dim(1)=" + std::to_string(xShape.GetDim(1)) + ", dim(2)=" + std::to_string(xShape.GetDim(2))).c_str(),
            "non-batch dims must be positive"),
        return ge::GRAPH_FAILED);
    for (int32_t idx : {kOutputSizeIdx, kKernelSizeIdx}) {
        auto input = context->GetInputShape(idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, input);
        auto shape = input->GetStorageShape();
        // output_size/kernel_size 必须为 1-D 且长度为 2
        OP_CHECK_IF(shape.GetDimNum() != SIZE_TENSOR_RANK || shape.GetDim(0) != SIZE_TENSOR_LEN,
                    OP_LOGE_FOR_INVALID_SHAPESIZE(
                        context->GetNodeName(), (idx == kOutputSizeIdx ? "output_size" : "kernel_size"),
                        (std::to_string(shape.GetDimNum()) + "D").c_str(), "1D with 2 elements"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// ========== 读取并校验单个长度 2 的 ListInt 属性 ==========
static ge::graphStatus ReadAttrVec(gert::TilingContext* context, int32_t attrIdx, const char* attrName,
                                   int64_t minValue, int64_t& elemH, int64_t& elemW)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto vecPtr = attrs->GetAttrPointer<gert::ContinuousVector>(attrIdx);
    OP_CHECK_NULL_WITH_CONTEXT(context, vecPtr);
    // 属性长度必须为 2
    OP_CHECK_IF(vecPtr->GetSize() != ATTR_VEC_LEN,
                OP_LOGE_FOR_INVALID_VALUE(context->GetNodeName(), attrName, std::to_string(vecPtr->GetSize()).c_str(),
                                          "2 elements"),
                return ge::GRAPH_FAILED);
    const int64_t* data = reinterpret_cast<const int64_t*>(vecPtr->GetData());
    OP_CHECK_NULL_WITH_CONTEXT(context, data);
    elemH = data[0];
    elemW = data[1];
    // dilation/stride 元素必须 > 0，padding 元素必须 >= 0
    OP_CHECK_IF(elemH < minValue || elemW < minValue,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), attrName,
                    ("(" + std::to_string(elemH) + ", " + std::to_string(elemW) + ")").c_str(),
                    ("elements must be >= " + std::to_string(minValue)).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ========== ValidateAttr（属性值域校验）==========
static ge::graphStatus ValidateAttr(gert::TilingContext* context, Col2ImV2Geom& geom)
{
    OP_CHECK_IF(
        ReadAttrVec(context, kDilationAttrIdx, "dilation", 1, geom.dilationH, geom.dilationW) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "ReadAttrVec dilation failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadAttrVec(context, kPaddingAttrIdx, "padding", 0, geom.paddingH, geom.paddingW) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ReadAttrVec padding failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadAttrVec(context, kStrideAttrIdx, "stride", 1, geom.strideH, geom.strideW) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ReadAttrVec stride failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ========== ValidateInputs 调度器 ==========
static ge::graphStatus ValidateInputs(gert::TilingContext* context, Col2ImV2Geom& geom)
{
    OP_CHECK_IF(ValidateDtype(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateDtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateShape(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateShape failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateAttr(context, geom) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateAttr failed"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ========== 读取 const tensor 值（值依赖输入：output_size/kernel_size）==========
static ge::graphStatus ReadConstSize(gert::TilingContext* context, int32_t idx, const char* inputName, int64_t& sizeH,
                                     int64_t& sizeW)
{
    auto tensor = context->GetInputTensor(idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensor);
    const int32_t* data = tensor->GetData<int32_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, data);
    sizeH = static_cast<int64_t>(data[0]);
    sizeW = static_cast<int64_t>(data[1]);
    // output_size/kernel_size 元素必须 > 0
    OP_CHECK_IF(sizeH <= 0 || sizeW <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), inputName,
                    ("(" + std::to_string(sizeH) + ", " + std::to_string(sizeW) + ")").c_str(), "elements must be > 0"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ========== 跨参数约束与 32 位值域保护 ==========
static ge::graphStatus ComputeGeometry(gert::TilingContext* context, Col2ImV2Geom& geom)
{
    auto xShape = context->GetInputShape(kXIdx)->GetStorageShape();
    // x.dim(1) 必须能被 kH*kW 整除
    int64_t kernelArea = geom.kernelH * geom.kernelW;
    OP_CHECK_IF(xShape.GetDim(1) % kernelArea != 0,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "x, kernel_size",
                    ("x.dim(1)=" + std::to_string(xShape.GetDim(1)) + ", kH*kW=" + std::to_string(kernelArea)).c_str(),
                    "x.dim(1) must be divisible by kH*kW"),
                return ge::GRAPH_FAILED);
    // ho = (outH + 2*padH - dilH*(kH-1) - 1)/strideH + 1 >= 1，wo 同理
    geom.colH = (geom.outH + 2 * geom.paddingH - geom.dilationH * (geom.kernelH - 1) - 1) / geom.strideH + 1;
    geom.colW = (geom.outW + 2 * geom.paddingW - geom.dilationW * (geom.kernelW - 1) - 1) / geom.strideW + 1;
    OP_CHECK_IF(
        geom.colH < 1 || geom.colW < 1,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "output_size, kernel_size, dilation, padding, stride",
            ("ho=" + std::to_string(geom.colH) + ", wo=" + std::to_string(geom.colW)).c_str(), "ho/wo must be >= 1"),
        return ge::GRAPH_FAILED);
    // x.dim(2) 必须等于 ho*wo
    OP_CHECK_IF(xShape.GetDim(2) != geom.colH * geom.colW,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x",
                                                       ("x.dim(2)=" + std::to_string(xShape.GetDim(2)) +
                                                        ", ho*wo=" + std::to_string(geom.colH * geom.colW))
                                                           .c_str(),
                                                       "x.dim(2) must equal ho*wo"),
                return ge::GRAPH_FAILED);
    // 32 位值域保护（UintDiv<uint32_t> 要求操作数 <= INT32_MAX）
    int64_t outHW = geom.outH * geom.outW;
    OP_CHECK_IF(outHW > INT32_MAX || geom.outW + geom.paddingW > INT32_MAX || geom.outH + geom.paddingH > INT32_MAX,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "output_size, padding",
                                                      ("outH*outW=" + std::to_string(outHW)).c_str(),
                                                      "outH*outW/outW+padW/outH+padH must be <= INT32_MAX"),
                return ge::GRAPH_FAILED);
    int64_t channel = xShape.GetDim(1) / kernelArea;
    const int64_t batch = xShape.GetDim(0);
    // Check the product before multiplying: shape metadata can exceed int64
    // even though each individual dimension is positive and valid.
    const int64_t maxBatchChannel = INT32_MAX / outHW;
    OP_CHECK_IF(
        batch > 0 && channel > maxBatchChannel / batch,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "x, output_size", "totalLength exceeds INT32_MAX",
                                              "totalLength must be <= INT32_MAX"),
        return ge::GRAPH_FAILED);
    geom.totalLength = batch * channel * outHW;
    return ge::GRAPH_SUCCESS;
}

// ========== DumpTilingData (DFX) ==========
static void DumpTilingData(gert::TilingContext* context, const Col2ImV2TilingData* tiling)
{
    OP_LOGD(context,
            "Col2ImV2TilingData: totalLength=%ld, needCoreNum=%ld, outputSizeH=%ld, outputSizeW=%ld, "
            "kernelSizeH=%ld, kernelSizeW=%ld, dilationH=%ld, dilationW=%ld, paddingH=%ld, paddingW=%ld, "
            "strideH=%ld, strideW=%ld, colH=%ld, colW=%ld",
            tiling->totalLength, tiling->needCoreNum, tiling->outputSizeH, tiling->outputSizeW, tiling->kernelSizeH,
            tiling->kernelSizeW, tiling->dilationH, tiling->dilationW, tiling->paddingH, tiling->paddingW,
            tiling->strideH, tiling->strideW, tiling->colH, tiling->colW);
}

// ========== FillTilingData ==========
static void FillTilingData(Col2ImV2TilingData* tiling, const Col2ImV2Geom& geom, int64_t needCoreNum)
{
    tiling->totalLength = geom.totalLength;
    tiling->needCoreNum = needCoreNum;
    tiling->outputSizeH = geom.outH;
    tiling->outputSizeW = geom.outW;
    tiling->kernelSizeH = geom.kernelH;
    tiling->kernelSizeW = geom.kernelW;
    tiling->dilationH = geom.dilationH;
    tiling->dilationW = geom.dilationW;
    tiling->paddingH = geom.paddingH;
    tiling->paddingW = geom.paddingW;
    tiling->strideH = geom.strideH;
    tiling->strideW = geom.strideW;
    tiling->colH = geom.colH;
    tiling->colW = geom.colW;
}

// ========== 核数两步法 + PER_CORE_MIN 抬升 ==========
static int64_t ComputeNeedCoreNum(int64_t totalLength, int64_t coreNum)
{
    if (totalLength == 0) {
        return 1; // 空 tensor（n=0）短路，避免除零
    }
    int64_t perCoreElements = (totalLength + coreNum - 1) / coreNum;
    if (perCoreElements < PER_CORE_MIN) {
        perCoreElements = PER_CORE_MIN;
    }
    return (totalLength + perCoreElements - 1) / perCoreElements;
}

// ========== TilingFunc ==========
static ge::graphStatus Col2ImV2TilingFunc(gert::TilingContext* context)
{
    // 1. validate inputs (dtype/shape/attr)
    Col2ImV2Geom geom;
    OP_CHECK_IF(ValidateInputs(context, geom) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ValidateInputs failed"),
                return ge::GRAPH_FAILED);
    // 2. read const tensor values (值依赖输入)
    OP_CHECK_IF(ReadConstSize(context, kOutputSizeIdx, "output_size", geom.outH, geom.outW) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ReadConstSize output_size failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadConstSize(context, kKernelSizeIdx, "kernel_size", geom.kernelH, geom.kernelW) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "ReadConstSize kernel_size failed"), return ge::GRAPH_FAILED);
    // 3. cross-param checks + geometry
    OP_CHECK_IF(ComputeGeometry(context, geom) != ge::GRAPH_SUCCESS, OP_LOGE(context, "ComputeGeometry failed"),
                return ge::GRAPH_FAILED);
    // 4. platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    // 5. fill tiling data
    Col2ImV2TilingData* tiling = context->GetTilingData<Col2ImV2TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    int64_t needCoreNum = ComputeNeedCoreNum(geom.totalLength, coreNum);
    FillTilingData(tiling, geom, needCoreNum);
    DumpTilingData(context, tiling);
    // 6. workspace（系统 workspace 必须无条件分配）
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    // 7. set block dim / local memory / tiling key
    context->SetBlockDim(needCoreNum);
    OP_CHECK_IF((ubSize <= DCACHE_SIZE + STATIC_UB_ESTIMATE),
                OP_LOGE(context, "ubSize %lu <= DCACHE_SIZE + STATIC_UB_ESTIMATE", ubSize), return ge::GRAPH_FAILED);
    auto res = context->SetLocalMemorySize(static_cast<uint32_t>(ubSize - DCACHE_SIZE - STATIC_UB_ESTIMATE));
    OP_CHECK_IF((res != ge::GRAPH_SUCCESS), OP_LOGE(context, "SetLocalMemorySize failed, ubSize=%lu", ubSize),
                return ge::GRAPH_FAILED);
    context->SetTilingKey(GET_TPL_TILING_KEY(static_cast<uint64_t>(COL2_IM_V2_SCH_MODE_DEFAULT)));
    return ge::GRAPH_SUCCESS;
}

// ========== TilingParse ==========
static ge::graphStatus TilingParseForCol2ImV2([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// ========== Registration（值依赖声明与 def.cpp/infershape 三处同步）==========
IMPL_OP_OPTILING(Col2ImV2)
    .Tiling(Col2ImV2TilingFunc)
    .TilingParse<Col2ImV2CompileInfo>(TilingParseForCol2ImV2)
    .TilingInputsDataDependency({kOutputSizeIdx, kKernelSizeIdx});

} // namespace optiling
