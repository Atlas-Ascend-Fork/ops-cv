/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grid_sampler3_d_grad_tiling.cpp
 * \brief
 */

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include "grid_sampler3_d_grad_tiling.h"
#include "op_host/tiling_util.h"
#include "op_api/error_report_util.h"

namespace optiling {
constexpr uint32_t RESERVED_UB = static_cast<uint32_t>(8 * 1024);
constexpr uint32_t DIM_LEN = 5;
constexpr uint32_t GRAD_INPUT_INDEX = 0;
constexpr uint32_t X_INPUT_INDEX = 1;
constexpr uint32_t GRID_INPUT_INDEX = 2;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr uint32_t DIM_INDEX2 = 2;
constexpr uint32_t DIM_INDEX3 = 3;
constexpr uint32_t DIM_INDEX4 = 4;
constexpr uint32_t FP32_BLOCK_NUM = 8;
constexpr uint32_t FP32_GROUP_SIZE = 8;
constexpr uint32_t DTYPE_SIZE_FLOAT = 4;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t CHANNEL_256 = 256;
constexpr uint32_t CHANNEL_512 = 512;
constexpr uint32_t CHANNEL_1024 = 1024;
constexpr uint32_t CONST_TWO = 2;
constexpr uint32_t FP32_GROUP_SIZE_LT_256 = 32;
constexpr uint32_t FP32_GROUP_SIZE_GT_256_LT_512 = 16;
constexpr uint32_t FP32_GROUP_SIZE_GT_512_LT_1024 = 8;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t ALIGN_256_BYTES = 256;
constexpr uint32_t BILINEAR_DIVIDE_UB_NUM = 180;
constexpr uint32_t NEAREST_DIVIDE_UB_NUM = 30;
constexpr uint32_t CONST_SEVENTEEN = 17;
constexpr uint32_t WORKSPACE_SIZE = static_cast<uint32_t>(16 * 1024 * 1024);
constexpr uint32_t CHECK_DIM_NUM = 5;
constexpr uint32_t INTERPOLATION_MODE_INDEX = 0;
constexpr uint32_t PADDING_MODE_INDEX = 1;
constexpr uint32_t ALIGN_CORNERS_INDEX = 2;
constexpr uint32_t INTERPOLATION_MODE_BILINEAR = 0;
constexpr uint32_t INTERPOLATION_MODE_NEAREST = 1;
constexpr uint32_t PADDING_MODE_ZEROS = 0;
constexpr uint32_t PADDING_MODE_BORDER = 1;
constexpr uint32_t PADDING_MODE_REFLECTION = 2;
constexpr uint32_t ALIGN_CORNERS_FALSE = 0;
constexpr uint32_t ALIGN_CORNERS_TRUE = 1;
constexpr uint32_t VF_MAX_THREAD_NUM = 256;
constexpr uint32_t GRID_LAST_NUM = 3;
constexpr uint32_t REGBASE_MAX_CHANNEL_NUM = 128;
constexpr uint32_t DETERMINISTIC_BATCH_NUM = 1;
constexpr uint32_t DETERMINISTIC_COMPUTE_NUM = 16;
constexpr uint32_t FLOAT32_TILING_KEY = 1;
constexpr uint32_t FLOAT16_TILING_KEY = 2;
constexpr uint32_t BFLOAT16_TILING_KEY = 3;
constexpr uint32_t DET_FLOAT32_TILING_KEY = 4;
constexpr uint32_t DET_FLOAT16_TILING_KEY = 5;
constexpr uint32_t DET_BFLOAT16_TILING_KEY = 6;
constexpr uint32_t SIMD_FLOAT32_TILING_KEY = 7;
constexpr uint8_t SCHEDULE_MODE = 1;

template <typename TilingData>
class GridSampler3DGradTiling {
public:
    explicit GridSampler3DGradTiling(InputParamsInfo& param, const uint32_t inputCoreNum, const uint32_t inputUbSize,
                                     const uint32_t deterministic)
    {
        this->coreNum = inputCoreNum;
        this->batch = param.batch;
        this->channel = param.channel;
        this->xD = param.xD;
        this->xH = param.xH;
        this->xW = param.xW;
        this->gridD = param.gridD;
        this->gridH = param.gridH;
        this->gridW = param.gridW;
        this->interpolation = param.interpolation;
        this->padding = param.padding;
        this->alignCorners = param.alignCorners;
        this->tilingKey = param.tilingKey;
        this->ubSize = FloorAlign(inputUbSize, BYTE_BLOCK);
        this->isDeterministic = deterministic;
        this->regBase = param.regBase;
        this->regBaseSIMD = param.regBaseSIMD;
        return;
    }

    void GetTiling(TilingData* tilingData);

private:
    void GetUsedCore();
    void SplitUb();
    void FillTilingData(TilingData* tilingData);

    template <typename T1, typename T2>
    inline auto FloorDiv(T1 a, T2 b)
    {
        if (b == 0) {
            throw std::invalid_argument("FloorDiv Division by zero");
        }
        return (a) / (b);
    }
    template <typename T1, typename T2>
    inline auto CeilAlign(T1 a, T2 b)
    {
        if (b == 0) {
            throw std::invalid_argument("CeilAlign Division by zero");
        }
        return (a + b - 1) / b * b;
    }
    template <typename T1, typename T2>
    inline auto FloorAlign(T1 a, T2 b)
    {
        if (b == 0) {
            throw std::invalid_argument("FloorAlign Division by zero");
        }
        return (a) / b * b;
    }
    inline uint32_t Ceil(uint32_t a, uint32_t b)
    {
        if (b == 0) {
            throw std::invalid_argument("Ceil Division by zero");
        }
        uint32_t tmp = a % b;
        if (tmp > 0) {
            return a / b + 1;
        } else {
            return a / b;
        }
    }

private:
    uint32_t batch = 0;
    uint32_t channel = 0;
    uint32_t xD = 0;
    uint32_t xH = 0;
    uint32_t xW = 0;
    uint32_t gridD = 0;
    uint32_t gridH = 0;
    uint32_t gridW = 0;
    uint32_t interpolation = 0;
    uint32_t padding = 0;
    uint32_t alignCorners = 0;
    uint32_t usedCoreNum = 0;
    uint32_t tailPNum = 0;
    uint32_t coreNum = 0;
    uint32_t group = 0;
    uint32_t ubSize = 0;
    uint32_t pNumPerCore = 0;
    uint32_t tilingKey = 1;
    uint32_t divideUbNum = 1;
    uint32_t extraUbSize = 0;
    uint32_t ubFactorElement = 0;
    uint32_t isDeterministic = 0;
    uint32_t tailBNum = 0;
    bool regBase = false;
    bool regBaseSIMD = false;
};

template <typename TilingData>
void GridSampler3DGradTiling<TilingData>::GetUsedCore()
{
    if (regBase && !regBaseSIMD) {
        uint64_t mulNCDHW = static_cast<uint64_t>(batch) * gridD * gridH * gridW * GRID_LAST_NUM;
        uint32_t tmpCoreNum = Ceil(mulNCDHW, VF_MAX_THREAD_NUM);
        usedCoreNum = tmpCoreNum <= coreNum ? tmpCoreNum : coreNum;
        pNumPerCore = 0;
        tailPNum = 0;
        if (isDeterministic == 1) {
            if (batch > coreNum) {
                usedCoreNum = coreNum;
                pNumPerCore = Ceil(batch, usedCoreNum);
                return;
            }
            usedCoreNum = batch;
            pNumPerCore = DETERMINISTIC_BATCH_NUM;
        }
        return;
    } else if (isDeterministic == 0) {
        uint64_t mulNDHW = static_cast<uint64_t>(batch) * gridD * gridH * gridW;
        if (mulNDHW <= coreNum) {
            usedCoreNum = mulNDHW;
            pNumPerCore = static_cast<uint32_t>(1);
            tailPNum = static_cast<uint32_t>(0);
            return;
        }
        usedCoreNum = coreNum;
        pNumPerCore = FloorDiv(mulNDHW, coreNum);
        tailPNum = static_cast<uint32_t>(mulNDHW % usedCoreNum);
    } else {
        if (batch > coreNum) {
            usedCoreNum = coreNum;
            uint32_t bNumPerCore = FloorDiv(batch, usedCoreNum);
            tailBNum = static_cast<uint32_t>(batch % usedCoreNum);
            pNumPerCore = gridD * gridH * gridW * bNumPerCore;
            return;
        }
        usedCoreNum = batch;
        pNumPerCore = gridD * gridH * gridW;
    }
}

template <typename TilingData>
void GridSampler3DGradTiling<TilingData>::SplitUb()
{
    uint32_t alignChannel = 0;
    alignChannel = CeilAlign(channel, FP32_BLOCK_NUM);

    if (static_cast<int32_t>(interpolation) == 0) { // bilinear
        divideUbNum = BILINEAR_DIVIDE_UB_NUM;
        extraUbSize = CONST_SEVENTEEN * alignChannel * DTYPE_SIZE_FLOAT;
        group = static_cast<uint32_t>(1);
    } else if (static_cast<int32_t>(interpolation) == 1) { // nearest
        divideUbNum = NEAREST_DIVIDE_UB_NUM;
        if (channel <= CHANNEL_256) {
            extraUbSize = BUFFER_NUM * (FP32_GROUP_SIZE_LT_256 + 1) * alignChannel * DTYPE_SIZE_FLOAT;
            group = FP32_GROUP_SIZE_LT_256;
        } else if (channel <= CHANNEL_512) {
            extraUbSize = BUFFER_NUM * (FP32_GROUP_SIZE_GT_256_LT_512 + 1) * alignChannel * DTYPE_SIZE_FLOAT;
            group = FP32_GROUP_SIZE_GT_256_LT_512;
        } else if (channel <= CHANNEL_1024) {
            extraUbSize = BUFFER_NUM * (FP32_GROUP_SIZE_GT_512_LT_1024 + 1) * alignChannel * DTYPE_SIZE_FLOAT;
            group = FP32_GROUP_SIZE_GT_512_LT_1024;
        } else {
            extraUbSize = BUFFER_NUM * CONST_TWO * alignChannel * DTYPE_SIZE_FLOAT;
            group = static_cast<uint32_t>(1);
        }
    }
    uint32_t tilingDataSize = CeilAlign(sizeof(TilingData), BYTE_BLOCK);
    uint32_t canUseUbSize = FloorAlign(ubSize - tilingDataSize, BYTE_BLOCK);
    if (canUseUbSize <= extraUbSize) {
        ubFactorElement = static_cast<uint32_t>(0);
        return;
    }
    ubFactorElement = FloorAlign((canUseUbSize - extraUbSize) / divideUbNum, ALIGN_256_BYTES) / DTYPE_SIZE_FLOAT;
}

template <typename TilingData>
void GridSampler3DGradTiling<TilingData>::FillTilingData(TilingData* tilingData)
{
    tilingData->set_batch(batch);
    tilingData->set_channel(channel);
    tilingData->set_xD(xD);
    tilingData->set_xH(xH);
    tilingData->set_xW(xW);
    tilingData->set_gridD(gridD);
    tilingData->set_gridH(gridH);
    tilingData->set_gridW(gridW);
    tilingData->set_interpolation(interpolation);
    tilingData->set_padding(padding);
    tilingData->set_alignCorners(alignCorners);
    tilingData->set_blockNum(usedCoreNum);
    tilingData->set_pNumPerCore(pNumPerCore);
    tilingData->set_tailPNum(tailPNum);
    tilingData->set_ubFactorElement(ubFactorElement);
    tilingData->set_group(group);
    tilingData->set_isDeterministic(isDeterministic);
    tilingData->set_tailBNum(tailBNum);
}

template <typename TilingData>
void GridSampler3DGradTiling<TilingData>::GetTiling(TilingData* tilingData)
{
    GetUsedCore();
    SplitUb();
    FillTilingData(tilingData);
}

template <typename TilingData>
void GetGridSampler3DGradTiling(TilingData* tilingData, InputParamsInfo& params, uint32_t coreNum, uint32_t ubSize,
                                uint32_t deterministic)
{
    class GridSampler3DGradTiling<TilingData> tilingObj(params, coreNum, ubSize, deterministic);
    tilingObj.GetTiling(tilingData);
}

static ge::graphStatus CheckShapes(gert::TilingContext* tilingContext, InputParamsInfo& params,
                                   const gert::StorageShape* gradShape, const gert::StorageShape* xShape,
                                   const gert::StorageShape* gridShape)
{
    if (xShape->GetStorageShape().GetDimNum() != CHECK_DIM_NUM) {
        error_report::ReportInvalidArgument(tilingContext->GetNodeName(), "input",
                                            "input dim is not 5, please check input");
        return ge::GRAPH_FAILED;
    }

    auto input = tilingContext->GetInputTensor(DIM_INDEX0);
    auto format = input->GetFormat().GetStorageFormat();

    uint32_t outD = gradShape->GetStorageShape().GetDim(DIM_INDEX1);
    uint32_t outH = gradShape->GetStorageShape().GetDim(DIM_INDEX2);
    uint32_t outW = gradShape->GetStorageShape().GetDim(DIM_INDEX3);
    if (params.regBase && format == ge::FORMAT_NCDHW) {
        outD = gradShape->GetStorageShape().GetDim(DIM_INDEX2);
        outH = gradShape->GetStorageShape().GetDim(DIM_INDEX3);
        outW = gradShape->GetStorageShape().GetDim(DIM_INDEX4);
    }
    uint32_t deterministic = tilingContext->GetDeterministic();
    if (params.regBase && params.channel >= REGBASE_MAX_CHANNEL_NUM && deterministic == 0) {
        params.regBaseSIMD = true;
    }

    if (outD != params.gridD || outH != params.gridH || outW != params.gridW) {
        error_report::ReportInvalidArgument(tilingContext->GetNodeName(), "check",
                                            "please check grad's dims and grid's dims");
        return ge::GRAPH_FAILED;
    }

    // grad、x、grid 的 N 必须一致
    const uint32_t gradN = gradShape->GetStorageShape().GetDim(DIM_INDEX0);
    const uint32_t xN = xShape->GetStorageShape().GetDim(DIM_INDEX0);
    const uint32_t gridN = gridShape->GetStorageShape().GetDim(DIM_INDEX0);
    if (gradN != xN || gradN != gridN) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            tilingContext->GetNodeName(), "expect",
            (std::to_string(gradN) + ", " + std::to_string(xN) + ", " + std::to_string(gridN)).c_str(),
            "expect grad, x and grid to have same N");
        return ge::GRAPH_FAILED;
    }

    // grad 与 x 的 C 必须一致，与 GetInputInfo 中 x 的 C 维解析条件保持一致
    const uint32_t gradC = gradShape->GetStorageShape().GetDim(
        params.regBase && format == ge::FORMAT_NCDHW ? DIM_INDEX1 : DIM_INDEX4);
    if (gradC != params.channel) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(tilingContext->GetNodeName(), "expect",
                                               (std::to_string(gradC) + ", " + std::to_string(params.channel)).c_str(),
                                               "expect grad and x to have same C");
        return ge::GRAPH_FAILED;
    }

    // grid 最后一维必须为 3，表示输入像素位置信息 (x, y, z)
    const uint32_t gridLastDim = gridShape->GetStorageShape().GetDim(DIM_INDEX4);
    if (gridLastDim != GRID_LAST_NUM) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "grid", std::to_string(gridLastDim).c_str(),
                                              "grid last dim must be 3");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetInputInfo(gert::TilingContext* tilingContext, InputParamsInfo& params)
{
    OP_LOGI(tilingContext->GetNodeName(), "GetInputInfo -> strat to get input dims.");
    const gert::StorageShape* gradShape = tilingContext->GetInputShape(GRAD_INPUT_INDEX);
    const gert::StorageShape* xShape = tilingContext->GetInputShape(X_INPUT_INDEX);
    const gert::StorageShape* gridShape = tilingContext->GetInputShape(GRID_INPUT_INDEX);
    OP_CHECK_IF(gradShape == nullptr,
                error_report::ReportTilingStageError(
                    tilingContext->GetNodeName(),
                    "gradShape is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        xShape == nullptr,
        error_report::ReportTilingStageError(
            tilingContext->GetNodeName(), "xShape is null in the tiling context; check the TilingContext construction"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(gridShape == nullptr,
                error_report::ReportTilingStageError(
                    tilingContext->GetNodeName(),
                    "gridShape is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(gradShape->GetStorageShape().GetDimNum() != CHECK_DIM_NUM,
                error_report::ReportInvalidArgument(tilingContext->GetNodeName(), "grad",
                                                    "grad dim is not 5, please check input"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        xShape->GetStorageShape().GetDimNum() != CHECK_DIM_NUM,
        error_report::ReportInvalidArgument(tilingContext->GetNodeName(), "dim", "x dim is not 5, please check input"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(gridShape->GetStorageShape().GetDimNum() != CHECK_DIM_NUM,
                error_report::ReportInvalidArgument(tilingContext->GetNodeName(), "grid",
                                                    "grid dim is not 5, please check input"),
                return ge::GRAPH_FAILED);

    auto input = tilingContext->GetInputTensor(DIM_INDEX0);
    auto format = input->GetFormat().GetStorageFormat();

    auto compileInfo = reinterpret_cast<const Tiling4GridSampler3DGradCompileInfo*>(tilingContext->GetCompileInfo());
    params.regBase = compileInfo->regBase;
    params.batch = xShape->GetStorageShape().GetDim(DIM_INDEX0);
    if (params.regBase && format == ge::FORMAT_NCDHW) {
        params.channel = xShape->GetStorageShape().GetDim(DIM_INDEX1);
        params.xD = xShape->GetStorageShape().GetDim(DIM_INDEX2);
        params.xH = xShape->GetStorageShape().GetDim(DIM_INDEX3);
        params.xW = xShape->GetStorageShape().GetDim(DIM_INDEX4);
    } else {
        params.xD = xShape->GetStorageShape().GetDim(DIM_INDEX1);
        params.xH = xShape->GetStorageShape().GetDim(DIM_INDEX2);
        params.xW = xShape->GetStorageShape().GetDim(DIM_INDEX3);
        params.channel = xShape->GetStorageShape().GetDim(DIM_INDEX4);
    }
    params.gridD = gridShape->GetStorageShape().GetDim(DIM_INDEX1);
    params.gridH = gridShape->GetStorageShape().GetDim(DIM_INDEX2);
    params.gridW = gridShape->GetStorageShape().GetDim(DIM_INDEX3);

    OP_CHECK_IF(
        CheckShapes(tilingContext, params, gradShape, xShape, gridShape) != ge::GRAPH_SUCCESS,
        OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(),
                               "Failed to check input shapes; propagate the first error reported by the failing stage"),
        return ge::GRAPH_FAILED);

    const gert::RuntimeAttrs* attrs = tilingContext->GetAttrs();
    const char* pInterpolationMode = attrs->GetAttrPointer<char>(static_cast<std::uint32_t>(INTERPOLATION_MODE_INDEX));
    OP_CHECK_IF(pInterpolationMode == nullptr,
                error_report::ReportTilingStageError(
                    tilingContext->GetNodeName(),
                    "pInterpolationMode is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    if (strcmp(pInterpolationMode, "bilinear") == 0) {
        params.interpolation = INTERPOLATION_MODE_BILINEAR;
    } else if (strcmp(pInterpolationMode, "nearest") == 0) {
        params.interpolation = INTERPOLATION_MODE_NEAREST;
    } else {
        error_report::ReportInvalidArgument(tilingContext->GetNodeName(), "interpolation_mode",
                                            "interpolation_mode only supports bilinear or nearest");
        return ge::GRAPH_FAILED;
    }

    const char* pPaddingMode = attrs->GetAttrPointer<char>(static_cast<std::uint32_t>(PADDING_MODE_INDEX));
    OP_CHECK_IF(pPaddingMode == nullptr,
                error_report::ReportTilingStageError(
                    tilingContext->GetNodeName(),
                    "pPaddingMode is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    if (strcmp(pPaddingMode, "zeros") == 0) {
        params.padding = PADDING_MODE_ZEROS;
    } else if (strcmp(pPaddingMode, "border") == 0) {
        params.padding = PADDING_MODE_BORDER;
    } else if (strcmp(pPaddingMode, "reflection") == 0) {
        params.padding = PADDING_MODE_REFLECTION;
    } else {
        error_report::ReportInvalidArgument(tilingContext->GetNodeName(), "padding_mode",
                                            "padding_mode only supports zeros, border or reflection");
        return ge::GRAPH_FAILED;
    }

    const bool* pAlignCorners = attrs->GetAttrPointer<bool>(static_cast<std::uint32_t>(ALIGN_CORNERS_INDEX));
    OP_CHECK_IF(pAlignCorners == nullptr,
                error_report::ReportTilingStageError(
                    tilingContext->GetNodeName(),
                    "pAlignCorners is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    params.alignCorners = ALIGN_CORNERS_FALSE;
    if (*pAlignCorners) {
        params.alignCorners = ALIGN_CORNERS_TRUE;
    }

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    size_t sysWorkspaceSize = WORKSPACE_SIZE;
    if (params.regBase && !params.regBaseSIMD) {
        uint32_t deterministic = tilingContext->GetDeterministic();
        if (deterministic == 1) {
            sysWorkspaceSize = WORKSPACE_SIZE + VF_MAX_THREAD_NUM * sizeof(int32_t) * DETERMINISTIC_COMPUTE_NUM *
                                                    compileInfo->coreNum * params.batch;
        } else {
            sysWorkspaceSize = 0;
        }
    }
    OP_CHECK_IF(
        (currentWorkspace == nullptr),
        OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(),
                               "Get currentWorkspace Failed; propagate the first error reported by the failing stage"),
        return ge::GRAPH_FAILED);
    currentWorkspace[0] = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus tiling4GridSampler3DGradTiling(gert::TilingContext* tilingContext)
{
    OP_LOGI(tilingContext->GetNodeName(), "Tiling4GridSampler3DGradTiling start.");
    uint32_t deterministic = tilingContext->GetDeterministic();
    if (deterministic == 1) {
        OP_LOGI(tilingContext->GetNodeName(), "Deterministic status is open.");
    }
    auto compileInfo = reinterpret_cast<const Tiling4GridSampler3DGradCompileInfo*>(tilingContext->GetCompileInfo());
    uint64_t ubSizePlatForm = compileInfo->ubSizePlatForm;

    uint32_t ubSize = static_cast<uint32_t>(ubSizePlatForm);
    uint32_t availableUb = ubSize - RESERVED_UB;

    ge::DataType inputDatatype = tilingContext->GetInputDesc(GRAD_INPUT_INDEX)->GetDataType();
    uint32_t coreNum = compileInfo->coreNum;
    OP_LOGI(tilingContext->GetNodeName(), "ubSizePlatForm: %lu, coreNum: %u", ubSizePlatForm, coreNum);

    InputParamsInfo params;
    if (GetInputInfo(tilingContext, params) != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "Failed to Parse input params, please check inputs; "
                                                             "propagate the first error reported by the failing stage");
        return ge::GRAPH_FAILED;
    }

    GridSampler3DGradTilingData tilingData;
    GetGridSampler3DGradTiling<GridSampler3DGradTilingData>(&tilingData, params, coreNum, availableUb, deterministic);

    if (inputDatatype == ge::DT_FLOAT) {
        tilingContext->SetTilingKey(FLOAT32_TILING_KEY);
        if (deterministic == 1 && compileInfo->regBase) {
            tilingContext->SetTilingKey(DET_FLOAT32_TILING_KEY);
            tilingContext->SetScheduleMode(SCHEDULE_MODE);
        } else if (params.regBaseSIMD) {
            tilingContext->SetTilingKey(SIMD_FLOAT32_TILING_KEY);
        }
    } else if (inputDatatype == ge::DT_FLOAT16) {
        tilingContext->SetTilingKey(FLOAT16_TILING_KEY);
        if (deterministic == 1) {
            tilingContext->SetTilingKey(DET_FLOAT16_TILING_KEY);
            tilingContext->SetScheduleMode(SCHEDULE_MODE);
        }
    } else if (inputDatatype == ge::DT_BF16) {
        tilingContext->SetTilingKey(BFLOAT16_TILING_KEY);
        if (deterministic == 1) {
            tilingContext->SetTilingKey(DET_BFLOAT16_TILING_KEY);
            tilingContext->SetScheduleMode(SCHEDULE_MODE);
        }
    }

    tilingContext->SetBlockDim(tilingData.get_blockNum());
    tilingContext->SetNeedAtomic(true);
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    OP_LOGI(tilingContext->GetNodeName(), "tiling4GridSampler3DGradTiling end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus tilingPrepare4GridSampler3DGrad(gert::TilingParseContext* context)
{
    OP_LOGI(context->GetNodeName(), "TilingPrepare4GridSampler3DGrad start.");
    auto compileInfo = context->GetCompiledInfo<Tiling4GridSampler3DGradCompileInfo>();
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
    auto ascendCPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendCPlatform.GetCoreNumAiv();
    compileInfo->regBase = Ops::Cv::OpTiling::IsRegbaseSocVersion(context);
    OP_CHECK_IF((compileInfo->coreNum <= 0),
                error_report::ReportTilingStageError(context->GetNodeName(),
                                                     "failed to get the core num from the platform info"),
                return ge::GRAPH_FAILED);
    uint64_t ubSizePlatForm;
    ascendCPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_CHECK_IF((compileInfo->ubSizePlatForm == 0),
                error_report::ReportTilingStageError(context->GetNodeName(),
                                                     "failed to get the ub size from the platform info"),
                return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "TilingPrepare4GridSampler3DGrad end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GridSampler3DGrad)
    .Tiling(tiling4GridSampler3DGradTiling)
    .TilingParse<Tiling4GridSampler3DGradCompileInfo>(tilingPrepare4GridSampler3DGrad);

} // namespace optiling
