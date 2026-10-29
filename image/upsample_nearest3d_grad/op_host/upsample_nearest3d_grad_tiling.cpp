/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_nearest3d_grad_tiling.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "upsample_nearest3d_grad_tiling.h"

#include <string>
#include "tiling/platform/platform_ascendc.h"
#include "op_host/tiling_util.h"
#include "op_api/error_report_util.h"

namespace optiling {
constexpr int64_t BEST_PERFORMANCE_SIZE_1 = 16;
constexpr int64_t BEST_PERFORMANCE_SIZE_2 = 32;
constexpr int64_t BEST_PERFORMANCE_SIZE_3 = 48;
constexpr int64_t BEST_PERFORMANCE_SIZE_4 = 64;

constexpr float BEST_PERFORMANCE_SCALE_1 = 50.0f;
constexpr float BEST_PERFORMANCE_SCALE_2 = 24.0f;
constexpr float BEST_PERFORMANCE_SCALE_3 = 10.0f;
constexpr float BEST_PERFORMANCE_SCALE_4 = 6.0f;

constexpr float ZERO_FLOAT = 0.0f;
constexpr float ONE_FLOAT = 1.0f;

constexpr uint8_t RESERVED_LENGTH = 5;
constexpr size_t RESERVED_LENGTH_INPUT_SIZE = 5;
constexpr size_t RESERVED_LENGTH_SCALES = 3;

constexpr uint8_t HALF_TYPE = 1;
constexpr uint8_t FLOAT_TYPE = 2;
constexpr uint8_t BFLOAT_TYPE = 3;

constexpr uint8_t BYTE_LEN_4 = 4;
constexpr uint8_t BYTE_SIZE_2 = 2;

constexpr uint8_t BATCH_DIM = 2;
constexpr uint8_t DIM = 3;
constexpr uint8_t D_INDEX = 0;
constexpr uint8_t H_INDEX = 1;
constexpr uint8_t W_INDEX = 2;
constexpr uint8_t SCHEDULE_MODE = 1;

constexpr int64_t WORK_SPACE_SIZE = 16 * 1024 * 1024;

class UpsampleNearest3dGradTiling {
public:
    explicit UpsampleNearest3dGradTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus Init() const;
    ge::graphStatus RunBigKernelTiling();

private:
    bool CheckScales() const;
    void SetScale();
    inline float ComputeScaleValue(int64_t inSize, int64_t outSize, const float scale) const;
    inline bool GetNeedResize(int64_t inSize, int64_t outSize, const float scale) const;
    void GetWorkSpace(int64_t neededCoreNum);
    ge::graphStatus GetShapes();
    void GetSlideSize();
    uint8_t GetDataTypeVal() const;
    uint8_t GetDataTypeSize() const;
    int64_t GetNeedCoreNum(int64_t coreNumPlatform);
    int64_t GetNeedCoreNumW(int64_t coreNumPlatform);
    int64_t GetNeedCoreNumH(int64_t coreNumPlatform);
    int64_t GetNeedCoreNumD(int64_t coreNumPlatform);
    void GetTCubeTilingW();
    void GetTCubeTilingH();
    void GetTCubeTilingD();
    void FillTilingData();

    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const;

    template <typename T1>
    inline int64_t Ceil(T1 x) const;

    template <typename T1>
    inline T1 Max(T1 a, T1 b, T1 c) const;

    template <typename T1>
    inline T1 Min(T1 a, T1 b, T1 c) const;

private:
    ge::DataType dataType = ge::DT_UNDEFINED;
    uint8_t dataTypeSize = 0;

    gert::TilingContext* tilingContext = nullptr;
    gert::Shape gradOutputShape;
    const gert::ContinuousVector* outputSizeAttr = nullptr;
    const gert::ContinuousVector* inputSizeAttr = nullptr;
    const gert::ContinuousVector* scalesAttr = nullptr;
    float scaleD = 0.0f;
    float scaleH = 0.0f;
    float scaleW = 0.0f;

    float realScaleW = 0.0f;
    float realScaleH = 0.0f;
    float realScaleD = 0.0f;
    bool needResizeW = true;
    bool needResizeH = true;
    bool needResizeD = true;

    int64_t batches = 0;
    int64_t gradOutputShapes[3] = {0};
    int64_t gradInputShapes[3] = {0};

    int64_t eachCoreSlideNumArray[3] = {0, 0, 0};
    int64_t remainderArray[3] = {0, 0, 0};
    int64_t tailStartSlideNumArray[3] = {0, 0, 0};
    int64_t groupCoreNums[3] = {0, 0, 0};
    int64_t inputRows[3] = {0, 0, 0};
    int64_t tailAvergingRows[3] = {0, 0, 0};
    int64_t needCoreNums[3] = {0, 0, 0};

    int64_t singleCoreKW = 0;
    int64_t singleCoreKH = 0;
    int64_t singleCoreKD = 0;

    int64_t tensorSizeW = 0;
    int64_t tensorSizeH = 0;
    int64_t tensorSizeD = 0;

    int64_t tensorSizeMappingW = 0;
    int64_t tensorSizeMappingH = 0;
    int64_t tensorSizeMappingD = 0;

    int64_t slideSize = 0;
    UpsampleNearest3dGradTilingData tilingData;
};

inline bool FloatEqual(float p, float q)
{
    float closeTo0 = float(1e-6);
    if (p > q) {
        return p - q < closeTo0;
    } else {
        return q - p < closeTo0;
    }
};

ge::graphStatus UpsampleNearest3dGradTiling::Init() const { return ge::GRAPH_SUCCESS; }

ge::graphStatus UpsampleNearest3dGradTiling::RunBigKernelTiling()
{
    bool regBase = Ops::Cv::OpTiling::IsRegbaseSocVersion(tilingContext);
    std::string opType(tilingContext->GetNodeType());
    if (regBase) {
        OP_LOGI(tilingContext->GetNodeName(), "Enter Tiling4UpsampleNearest3dGradRegbase");
        return Tiling4UpsampleNearest3dGradRegbase(tilingContext);
    }
    // 获取输入矩阵
    auto srcInputTensor = tilingContext->GetInputTensor(0);
    if (srcInputTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取输入的参数
    const gert::RuntimeAttrs* tilingAttrs = tilingContext->GetAttrs();
    if (tilingAttrs == nullptr) {
        return ge::GRAPH_FAILED;
    }

    size_t idx = 0;
    inputSizeAttr = tilingAttrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    OP_CHECK_IF(inputSizeAttr == nullptr,
                error_report::ReportTilingStageError(tilingContext->GetNodeName(),
                                                     "the attr inputSizeAttr is missing in the tiling context; check "
                                                     "the operator registration and the TilingContext construction"),
                return ge::GRAPH_FAILED);
    size_t inputSizeNum = inputSizeAttr->GetSize();
    OP_CHECK_IF(
        inputSizeNum != RESERVED_LENGTH_INPUT_SIZE,
        OP_LOGE_FOR_INVALID_SHAPESIZE(tilingContext->GetNodeName(), "input_size", std::to_string(inputSizeNum).c_str(),
                                      std::to_string(RESERVED_LENGTH_INPUT_SIZE).c_str()),
        return ge::GRAPH_FAILED);
    outputSizeAttr = tilingAttrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    OP_CHECK_IF(outputSizeAttr == nullptr,
                error_report::ReportTilingStageError(
                    tilingContext->GetNodeName(),
                    "outputSizeAttr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    scalesAttr = tilingAttrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    OP_CHECK_IF(scalesAttr == nullptr,
                error_report::ReportTilingStageError(
                    tilingContext->GetNodeName(),
                    "scalesAttr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    size_t scalesNum = scalesAttr->GetSize();
    OP_CHECK_IF(scalesNum != RESERVED_LENGTH_SCALES,
                OP_LOGE_FOR_INVALID_VALUE(tilingContext->GetNodeName(), "scales", std::to_string(scalesNum).c_str(),
                                          std::to_string(RESERVED_LENGTH_SCALES).c_str()),
                return ge::GRAPH_FAILED);
    const float* scalesArray = reinterpret_cast<const float*>(scalesAttr->GetData());
    scaleD = scalesArray[D_INDEX];
    scaleH = scalesArray[H_INDEX];
    scaleW = scalesArray[W_INDEX];

    // 获取数据类型
    auto temp = tilingContext->GetInputDesc(0);
    if (temp == nullptr) {
        return ge::GRAPH_FAILED;
    }
    dataType = tilingContext->GetInputDesc(0)->GetDataType();
    dataTypeSize = GetDataTypeSize();

    // 获取输入的shape
    auto srcShape = tilingContext->GetInputShape(0);
    gradOutputShape = srcShape->GetOriginShape();

    if (GetShapes() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckScales() == false) {
        return ge::GRAPH_FAILED;
    }
    SetScale();
    GetSlideSize();

    // 数据分核
    auto compileInfo = reinterpret_cast<const UpsampleNearest3dGradCompileInfo*>(tilingContext->GetCompileInfo());
    int64_t coreNumPlatform = compileInfo->coreNum;
    int64_t neededCoreNum = GetNeedCoreNum(coreNumPlatform);
    GetWorkSpace(neededCoreNum);
    tilingContext->SetBlockDim(neededCoreNum);
    tilingContext->SetTilingKey(1);

    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleNearest3dGradTiling::GetShapes()
{
    const int64_t* inputSizeArray = reinterpret_cast<const int64_t*>(inputSizeAttr->GetData());

    // grad_output 的 N/C 维须与 input_size 一致
    OP_CHECK_IF(inputSizeArray[0] != gradOutputShape.GetDim(0),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    tilingContext->GetNodeName(), "attr",
                    (std::to_string(inputSizeArray[0]) + ", " + std::to_string(gradOutputShape.GetDim(0))).c_str(),
                    "attr::input_size[0](get ) != input::grad_output N dim(get )"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(inputSizeArray[1] != gradOutputShape.GetDim(1),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    tilingContext->GetNodeName(), "attr",
                    (std::to_string(inputSizeArray[1]) + ", " + std::to_string(gradOutputShape.GetDim(1))).c_str(),
                    "attr::input_size[1](get ) != input::grad_output C dim(get )"),
                return ge::GRAPH_FAILED);

    batches = inputSizeArray[0] * inputSizeArray[1];
    for (int8_t i = 0; i < DIM; i++) {
        gradInputShapes[i] = inputSizeArray[i + BATCH_DIM];
        gradOutputShapes[i] = gradOutputShape[i + BATCH_DIM];
    }
    tilingData.set_batches(batches);
    tilingData.set_gradInputShapes(gradInputShapes);
    tilingData.set_gradOutputShapes(gradOutputShapes);
    return ge::GRAPH_SUCCESS;
}

bool UpsampleNearest3dGradTiling::CheckScales() const
{
    float checkScalesD = ComputeScaleValue(gradInputShapes[D_INDEX], gradOutputShapes[D_INDEX], scaleD);
    float checkScalesH = ComputeScaleValue(gradInputShapes[H_INDEX], gradOutputShapes[H_INDEX], scaleH);
    float checkScalesW = ComputeScaleValue(gradInputShapes[W_INDEX], gradOutputShapes[W_INDEX], scaleW);
    OP_CHECK_IF(
        checkScalesD > BEST_PERFORMANCE_SCALE_1 || checkScalesH > BEST_PERFORMANCE_SCALE_1 ||
            checkScalesW > BEST_PERFORMANCE_SCALE_1,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            tilingContext->GetNodeName(), "scales",
            (std::to_string(checkScalesD) + ", " + std::to_string(checkScalesH) + ", " + std::to_string(checkScalesW))
                .c_str(),
            "scales are too large, scalesD [], scalesH [], scalesW []"),
        return false);
    return true;
}

void UpsampleNearest3dGradTiling::SetScale()
{
    needResizeD = GetNeedResize(gradInputShapes[D_INDEX], gradOutputShapes[D_INDEX], scaleD);
    needResizeH = GetNeedResize(gradInputShapes[H_INDEX], gradOutputShapes[H_INDEX], scaleH);
    needResizeW = GetNeedResize(gradInputShapes[W_INDEX], gradOutputShapes[W_INDEX], scaleW);
    if (!needResizeD && !needResizeH && !needResizeW) {
        needResizeW = true;
    }
    tilingData.set_needResizeD(needResizeD);
    tilingData.set_needResizeH(needResizeH);
    tilingData.set_needResizeW(needResizeW);

    realScaleD = ComputeScaleValue(gradInputShapes[D_INDEX], gradOutputShapes[D_INDEX], scaleD);
    realScaleH = ComputeScaleValue(gradInputShapes[H_INDEX], gradOutputShapes[H_INDEX], scaleH);
    realScaleW = ComputeScaleValue(gradInputShapes[W_INDEX], gradOutputShapes[W_INDEX], scaleW);
    tilingData.set_scaleD(realScaleD);
    tilingData.set_scaleH(realScaleH);
    tilingData.set_scaleW(realScaleW);
}

inline float UpsampleNearest3dGradTiling::ComputeScaleValue(int64_t inSize, int64_t outSize, const float scale) const
{
    if (scale > ZERO_FLOAT) {
        return scale;
    } else {
        return inSize != 0 ? (static_cast<float>(outSize) / inSize) : ZERO_FLOAT;
    }
}

inline bool UpsampleNearest3dGradTiling::GetNeedResize(int64_t inSize, int64_t outSize, const float scale) const
{
    if (scale > ZERO_FLOAT) {
        return !FloatEqual(scale, ONE_FLOAT);
    } else {
        return inSize != outSize;
    }
}

void UpsampleNearest3dGradTiling::GetSlideSize()
{
    auto maxScale = Max(realScaleW, realScaleH, realScaleD);
    if (maxScale <= BEST_PERFORMANCE_SCALE_4) {
        slideSize = BEST_PERFORMANCE_SIZE_4;
    } else if (maxScale <= BEST_PERFORMANCE_SCALE_3) {
        slideSize = BEST_PERFORMANCE_SIZE_3;
    } else if (maxScale <= BEST_PERFORMANCE_SCALE_2) {
        slideSize = BEST_PERFORMANCE_SIZE_2;
    } else {
        slideSize = BEST_PERFORMANCE_SIZE_1;
    }
    tilingData.set_slideSize(slideSize);
}

int64_t UpsampleNearest3dGradTiling::GetNeedCoreNum(int64_t coreNumPlatform)
{
    int64_t needCoreNumW = 0;
    int64_t needCoreNumH = 0;
    int64_t needCoreNumD = 0;
    if (needResizeW) {
        singleCoreKW = Ceil(static_cast<float>(slideSize) * realScaleW) + RESERVED_LENGTH;
        if (singleCoreKW > gradOutputShapes[W_INDEX]) {
            singleCoreKW = gradOutputShapes[W_INDEX];
        }
        needCoreNumW = GetNeedCoreNumW(coreNumPlatform);
        GetTCubeTilingW();
    }

    if (needResizeH) {
        singleCoreKH = Ceil(static_cast<float>(slideSize) * realScaleH) + RESERVED_LENGTH;
        if (singleCoreKH > gradOutputShapes[H_INDEX]) {
            singleCoreKH = gradOutputShapes[H_INDEX];
        }
        needCoreNumH = GetNeedCoreNumH(coreNumPlatform);
        GetTCubeTilingH();
    }

    if (needResizeD) {
        singleCoreKD = Ceil(static_cast<float>(slideSize) * realScaleD) + RESERVED_LENGTH;
        if (singleCoreKD > gradOutputShapes[D_INDEX]) {
            singleCoreKD = gradOutputShapes[D_INDEX];
        }
        needCoreNumD = GetNeedCoreNumD(coreNumPlatform);
        GetTCubeTilingD();
    }

    int64_t tensorSize = Max(tensorSizeW, tensorSizeH, tensorSizeD);
    tilingData.set_tensorSize(tensorSize + slideSize);

    int64_t tensorSizeMapping = Max(tensorSizeMappingW, tensorSizeMappingH, tensorSizeMappingD);
    tilingData.set_tensorSizeMapping(tensorSizeMapping);

    int64_t neededCoreNum = Max(needCoreNumW, needCoreNumH, needCoreNumD);
    neededCoreNum = neededCoreNum < 1 ? 1 : neededCoreNum;
    return neededCoreNum;
}

int64_t UpsampleNearest3dGradTiling::GetNeedCoreNumW(int64_t coreNumPlatform)
{
    int64_t slideNum = CeilA2B(gradInputShapes[W_INDEX], slideSize);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;
    tensorSizeW = slideSize + 1;
    tensorSizeMappingW = Ceil(static_cast<float>(tensorSizeW) * realScaleW) + slideSize;

    inputRows[W_INDEX] = batches * gradOutputShapes[D_INDEX] * gradOutputShapes[H_INDEX];
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRow = slideSize;
    if (remainder > 0) {
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRow = std::max(CeilA2B(inputRows[W_INDEX], groupCoreNum), slideSize);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(inputRows[W_INDEX], tailAvergingRow));
    }

    int64_t neededCoreNum = coreNumPlatform;
    if (eachCoreSlideNum == 0 && remainder > 0) {
        neededCoreNum = remainder * groupCoreNum;
    }

    eachCoreSlideNumArray[W_INDEX] = eachCoreSlideNum;
    remainderArray[W_INDEX] = remainder;
    tailStartSlideNumArray[W_INDEX] = eachCoreSlideNum * coreNumPlatform;
    groupCoreNums[W_INDEX] = groupCoreNum;
    tailAvergingRows[W_INDEX] = tailAvergingRow;
    needCoreNums[W_INDEX] = neededCoreNum;
    return neededCoreNum;
}

int64_t UpsampleNearest3dGradTiling::GetNeedCoreNumH(int64_t coreNumPlatform)
{
    int64_t slideNum = CeilA2B(gradInputShapes[H_INDEX], slideSize);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;
    tensorSizeH = slideSize + 1;
    tensorSizeMappingH = Ceil(static_cast<float>(tensorSizeH) * realScaleH) + slideSize;

    if (batches * gradOutputShapes[D_INDEX] > gradInputShapes[W_INDEX]) {
        inputRows[H_INDEX] = batches * gradOutputShapes[D_INDEX];
    } else {
        inputRows[H_INDEX] = gradInputShapes[W_INDEX];
    }
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRow = slideSize;
    if (remainder > 0) {
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRow = std::max(CeilA2B(inputRows[H_INDEX], groupCoreNum), slideSize);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(inputRows[H_INDEX], tailAvergingRow));
    }

    int64_t neededCoreNum = coreNumPlatform;
    if (eachCoreSlideNum == 0 && remainder > 0) {
        neededCoreNum = remainder * groupCoreNum;
    }

    eachCoreSlideNumArray[H_INDEX] = eachCoreSlideNum;
    remainderArray[H_INDEX] = remainder;
    tailStartSlideNumArray[H_INDEX] = eachCoreSlideNum * coreNumPlatform;
    groupCoreNums[H_INDEX] = groupCoreNum;
    tailAvergingRows[H_INDEX] = tailAvergingRow;
    needCoreNums[H_INDEX] = neededCoreNum;
    return neededCoreNum;
}

int64_t UpsampleNearest3dGradTiling::GetNeedCoreNumD(int64_t coreNumPlatform)
{
    int64_t slideNum = CeilA2B(gradInputShapes[D_INDEX], slideSize);
    int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
    int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;
    tensorSizeD = slideSize + 1;
    tensorSizeMappingD = Ceil(static_cast<float>(tensorSizeD) * realScaleD) + slideSize;

    if (batches > gradInputShapes[H_INDEX] * gradInputShapes[W_INDEX]) {
        inputRows[D_INDEX] = batches;
    } else {
        inputRows[D_INDEX] = gradInputShapes[H_INDEX] * gradInputShapes[W_INDEX];
    }
    int64_t groupCoreNum = coreNumPlatform;
    int64_t tailAvergingRow = slideSize;
    if (remainder > 0) {
        groupCoreNum = coreNumPlatform / remainder;
        tailAvergingRow = std::max(CeilA2B(inputRows[D_INDEX], groupCoreNum), slideSize);
        groupCoreNum = std::min(groupCoreNum, CeilA2B(inputRows[D_INDEX], tailAvergingRow));
    }

    int64_t neededCoreNum = coreNumPlatform;
    if (eachCoreSlideNum == 0 && remainder > 0) {
        neededCoreNum = remainder * groupCoreNum;
    }

    eachCoreSlideNumArray[D_INDEX] = eachCoreSlideNum;
    remainderArray[D_INDEX] = remainder;
    tailStartSlideNumArray[D_INDEX] = eachCoreSlideNum * coreNumPlatform;
    groupCoreNums[D_INDEX] = groupCoreNum;
    tailAvergingRows[D_INDEX] = tailAvergingRow;
    needCoreNums[D_INDEX] = neededCoreNum;
    return neededCoreNum;
}

void UpsampleNearest3dGradTiling::GetWorkSpace(int64_t neededCoreNum)
{
    uint8_t size = 32 / dataTypeSize;
    // 中间矩阵预留GM空间
    int64_t intermediateMatrixSizeW = 0;
    if (needResizeW && (needResizeH || needResizeD)) {
        intermediateMatrixSizeW = batches * gradOutputShapes[D_INDEX] * gradOutputShapes[H_INDEX] *
                                  gradInputShapes[W_INDEX];
        intermediateMatrixSizeW = CeilA2B(intermediateMatrixSizeW, size) * size;
    }
    int64_t intermediateMatrixSizeH = 0;
    if (needResizeH && needResizeD) {
        intermediateMatrixSizeH = batches * gradOutputShapes[D_INDEX] * gradInputShapes[H_INDEX] *
                                  gradInputShapes[W_INDEX];
        intermediateMatrixSizeH = CeilA2B(intermediateMatrixSizeH, size) * size;
    }
    tilingData.set_intermediateMatrixSizeW(intermediateMatrixSizeW);
    tilingData.set_intermediateMatrixSizeH(intermediateMatrixSizeH);

    // 权重矩阵预留GM空间
    int64_t singleCoreK = Max(singleCoreKW, singleCoreKH, singleCoreKD);
    int64_t radioMatrixSize = slideSize * singleCoreK;
    tilingData.set_radioMatrixSize(radioMatrixSize);

    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = (intermediateMatrixSizeW + intermediateMatrixSizeH + radioMatrixSize * neededCoreNum) *
                        dataTypeSize +
                    WORK_SPACE_SIZE;
}

void UpsampleNearest3dGradTiling::GetTCubeTilingW()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTilingW;
    mmTilingW.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingW.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingW.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingW.SetOrgShape(batches * gradOutputShapes[D_INDEX] * gradOutputShapes[H_INDEX], gradInputShapes[W_INDEX],
                          gradOutputShapes[W_INDEX]);
    mmTilingW.SetShape(batches * gradOutputShapes[D_INDEX] * gradOutputShapes[H_INDEX], slideSize, singleCoreKW);
    if (mmTilingW.GetTiling(tilingData.matmulTilingW) == -1) {
        return;
    }
}

void UpsampleNearest3dGradTiling::GetTCubeTilingH()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTilingH;
    mmTilingH.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingH.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingH.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingH.SetOrgShape(gradInputShapes[H_INDEX], gradInputShapes[W_INDEX], gradOutputShapes[H_INDEX]);
    mmTilingH.SetShape(slideSize, gradInputShapes[W_INDEX], singleCoreKH);
    if (mmTilingH.GetTiling(tilingData.matmulTilingH) == -1) {
        return;
    }
}

void UpsampleNearest3dGradTiling::GetTCubeTilingD()
{
    auto mmDataType = static_cast<matmul_tiling::DataType>(dataType);
    matmul_tiling::MatmulApiTiling mmTilingD;
    mmTilingD.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingD.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType, false);
    mmTilingD.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    mmTilingD.SetOrgShape(gradInputShapes[D_INDEX], gradInputShapes[H_INDEX] * gradInputShapes[W_INDEX],
                          gradOutputShapes[D_INDEX]);
    mmTilingD.SetShape(slideSize, gradInputShapes[H_INDEX] * gradInputShapes[W_INDEX], singleCoreKD);
    if (mmTilingD.GetTiling(tilingData.matmulTilingD) == -1) {
        return;
    }
}

void UpsampleNearest3dGradTiling::FillTilingData()
{
    tilingData.set_eachCoreSlideNums(eachCoreSlideNumArray);
    tilingData.set_remainders(remainderArray);
    tilingData.set_tailStartSlideNums(tailStartSlideNumArray);
    tilingData.set_groupCoreNums(groupCoreNums);
    tilingData.set_inputRows(inputRows);
    tilingData.set_tailAvergingRows(tailAvergingRows);
    tilingData.set_needCoreNums(needCoreNums);
    tilingData.set_dataType(GetDataTypeVal());
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

uint8_t UpsampleNearest3dGradTiling::GetDataTypeSize() const
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return BYTE_LEN_4;
        case ge::DT_FLOAT16:
            return BYTE_SIZE_2;
        case ge::DT_BF16:
            return BYTE_SIZE_2;
        default:
            return BYTE_LEN_4;
    }
}

uint8_t UpsampleNearest3dGradTiling::GetDataTypeVal() const
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            return HALF_TYPE;
        case ge::DT_BF16:
            return BFLOAT_TYPE;
        case ge::DT_FLOAT:
            return FLOAT_TYPE;
        default:
            return 0;
    }
}

template <typename T1, typename T2>
inline auto UpsampleNearest3dGradTiling::CeilA2B(T1 a, T2 b) const -> T1
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

template <typename T1>
inline int64_t UpsampleNearest3dGradTiling::Ceil(T1 x) const
{
    int64_t floorX = int64_t(x);
    if (FloatEqual(x, floorX)) {
        return floorX;
    }
    return floorX + 1;
}

template <typename T1>
inline auto UpsampleNearest3dGradTiling::Max(T1 a, T1 b, T1 c) const -> T1
{
    if (a > b) {
        return a > c ? a : c;
    }
    return b > c ? b : c;
}

template <typename T1>
inline auto UpsampleNearest3dGradTiling::Min(T1 a, T1 b, T1 c) const -> T1
{
    if (a < b) {
        return a < c ? a : c;
    }
    return b < c ? b : c;
}

static ge::graphStatus Tiling4UpsampleNearest3dGradTiling(gert::TilingContext* context)
{
    UpsampleNearest3dGradTiling tilingObject(context);
    context->SetScheduleMode(SCHEDULE_MODE);
    return tilingObject.RunBigKernelTiling();
}

static ge::graphStatus TilingPrepareTiling(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<UpsampleNearest3dGradCompileInfo>();
    OP_CHECK_IF(
        compileInfo == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "compileInfo is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(
        platformInfo == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "platformInfo is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(compileInfo->coreNum <= 0,
                error_report::ReportTilingStageError(context->GetNodeName(),
                                                     "failed to get the hardware info from the platform info"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpsampleNearest3dGrad)
    .Tiling(Tiling4UpsampleNearest3dGradTiling)
    .TilingParse<UpsampleNearest3dGradCompileInfo>(TilingPrepareTiling);

IMPL_OP_OPTILING(UpsampleNearestExact3dGrad)
    .Tiling(Tiling4UpsampleNearest3dGradTiling)
    .TilingParse<UpsampleNearest3dGradCompileInfo>(TilingPrepareTiling);

} // namespace optiling
