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
 * \file upsample_bicubic2d_grad_tiling.cpp
 * \brief
 */
#include <cmath>
#include "tiling/tiling_api.h"
#include "op_host/tiling_base.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/storage_format.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "upsample_bicubic2d_grad_tiling.h"
#include "op_api/error_report_util.h"

namespace optiling {

bool AddWorkspaces(gert::TilingContext* context, const size_t workspace)
{
    size_t* workspace_size = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspace_size == nullptr,
                error_report::ReportTilingStageError(
                    context->GetNodeName(),
                    "workspace_size is null in the tiling context; check the TilingContext construction"),
                return false);
    *workspace_size = workspace;
    return true;
}

inline bool FloatEqual(float a, float b)
{
    float closeToZero = float(1e-6);
    if (a > b) {
        return a - b < closeToZero;
    } else {
        return b - a < closeToZero;
    }
}

bool UpsampleBicubic2dGradTiling::GetPlatformInfo(const gert::TilingContext* context)
{
    auto compileInfoPtr = reinterpret_cast<const UpsampleBicubic2dGradCompileInfo*>(context->GetCompileInfo());
    if (compileInfoPtr == nullptr) {
        error_report::ReportTilingStageError(
            context->GetNodeName(),
            "the compile info of UpsampleBicubic2dGrad is null; check the TilingContext construction");
        return false;
    }
    _Params.CoreNum = compileInfoPtr->aivNum;
    return true;
}

bool UpsampleBicubic2dGradTiling::GetCheckAttr(const gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(
        attrs == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "attrs is null in the tiling context; check the TilingContext construction"),
        return false);

    const bool* align_corners = attrs->GetAttrPointer<bool>(0);
    OP_CHECK_IF(align_corners == nullptr,
                error_report::ReportTilingStageError(
                    context->GetNodeName(),
                    "align_corners is null in the tiling context; check the TilingContext construction"),
                return false);
    const float* scales_h = attrs->GetAttrPointer<float>(1);
    OP_CHECK_IF(
        scales_h == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "scales_h is null in the tiling context; check the TilingContext construction"),
        return false);
    const float* scales_w = attrs->GetAttrPointer<float>(2);
    OP_CHECK_IF(
        scales_w == nullptr,
        error_report::ReportTilingStageError(
            context->GetNodeName(), "scales_w is null in the tiling context; check the TilingContext construction"),
        return false);

    _Params.alignCorners = *align_corners;
    _Params.scalesH = *scales_h;
    _Params.scalesW = *scales_w;
    return true;
}

bool UpsampleBicubic2dGradTiling::CheckInOutShapes(const gert::TilingContext* context)
{
    // input
    auto input_tensor = context->GetInputShape(0);
    OP_CHECK_IF(
        input_tensor == nullptr,
        error_report::ReportInvalidArgument(context->GetNodeName(), "input_tensor",
                                            "the input desc/shape of required input input_tensor is missing in the "
                                            "tiling context; check that the graph node provides all required inputs"),
        return false);
    auto input_shape = input_tensor->GetStorageShape();
    OP_CHECK_IF(input_shape.GetDimNum() != 4,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "get", std::to_string(input_shape.GetDimNum()).c_str(),
                    "upsampleBicubic2dGrad get input shape dim is not 4[NCHW], please check"),
                return false);
    _Params.batch = input_shape.GetDim(0) * input_shape.GetDim(1);
    _Params.inputN = input_shape.GetDim(0);
    _Params.inputC = input_shape.GetDim(1);
    _Params.inputH = input_shape.GetDim(NUM_TWO);
    _Params.inputW = input_shape.GetDim(NUM_THREE);

    auto output_tensor = context->GetOutputShape(0);
    OP_CHECK_IF(output_tensor == nullptr,
                error_report::ReportTilingStageError(
                    context->GetNodeName(),
                    "output_tensor is null in the tiling context; check the TilingContext construction"),
                return false);
    auto output_shape = output_tensor->GetStorageShape();
    OP_CHECK_IF(output_shape.GetDimNum() != 4,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "get", std::to_string(output_shape.GetDimNum()).c_str(),
                    "upsampleBicubic2dGrad get output shape dim is not 4[NCHW], please check"),
                return false);
    OP_CHECK_IF(output_shape.GetDim(0) != input_shape.GetDim(0),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "get",
                    (std::to_string(output_shape.GetDim(0)) + ", " + std::to_string(input_shape.GetDim(0))).c_str(),
                    "upsampleBicubic2dGrad get output shape dim[0] not match input shape dim[0] , please check"),
                return false);
    OP_CHECK_IF(output_shape.GetDim(1) != input_shape.GetDim(1),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "get",
                    (std::to_string(output_shape.GetDim(1)) + ", " + std::to_string(input_shape.GetDim(1))).c_str(),
                    "upsampleBicubic2dGrad get output shape dim[1] not match input shape dim[1] , please check"),
                return false);
    _Params.outputH = output_shape.GetDim(NUM_TWO);
    _Params.outputW = output_shape.GetDim(NUM_THREE);

    return true;
}

void UpsampleBicubic2dGradTiling::InitPlatformInfo(const UpsampleBicubic2dGradCompileInfo* compileInfoPtr,
                                                   matmul_tiling::PlatformInfo& platformInfo) const
{
    platformInfo.socVersion = compileInfoPtr->socVersion;
    platformInfo.l1Size = compileInfoPtr->l1Size;
    platformInfo.l0CSize = compileInfoPtr->l0CSize;
    platformInfo.ubSize = compileInfoPtr->ubSize;
    platformInfo.l0ASize = compileInfoPtr->l0ASize;
    platformInfo.l0BSize = compileInfoPtr->l0BSize;
}

bool UpsampleBicubic2dGradTiling::GetMMTilingData(const gert::TilingContext* context)
{
    auto dataType = static_cast<matmul_tiling::DataType>(_Params.dataType);
    matmul_tiling::PlatformInfo platformInfo;
    auto compileInfoPtr = reinterpret_cast<const UpsampleBicubic2dGradCompileInfo*>(context->GetCompileInfo());
    if (compileInfoPtr == nullptr) {
        error_report::ReportTilingStageError(
            context->GetNodeName(),
            "the compile info of UpsampleBicubic2dGrad is null; check the TilingContext construction");
        return false;
    }
    InitPlatformInfo(compileInfoPtr, platformInfo);
    matmul_tiling::MatmulApiTiling matmul_h(platformInfo);
    auto ret = matmul_h.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    OP_CHECK_IF(
        ret == -1,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "matmul_h SetAType failed; propagate the first error reported by the failing stage"),
        return false);
    ret = matmul_h.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    OP_CHECK_IF(
        ret == -1,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "matmul_h SetBType failed; propagate the first error reported by the failing stage"),
        return false);
    ret = matmul_h.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    OP_CHECK_IF(
        ret == -1,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "matmul_h SetCType failed; propagate the first error reported by the failing stage"),
        return false);
    ret = matmul_h.SetOrgShape(_Params.baseNH, _Params.outputW, NUM_FRACTAL, _Params.inputH);
    OP_CHECK_IF(
        ret == -1,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "matmul_h SetOrgShape failed; propagate the first error reported by the failing stage"),
        return false);
    ret = matmul_h.SetShape(_Params.baseNH, _Params.outputW, NUM_FRACTAL);
    OP_CHECK_IF(ret == -1,
                OP_LOGE_WITHOUT_REPORT(
                    context->GetNodeName(),
                    "matmul_h Set single shape failed; propagate the first error reported by the failing stage"),
                return false);
    ret = matmul_h.SetBufferSpace(-1, -1, -1);
    OP_CHECK_IF(ret == -1,
                OP_LOGE_WITHOUT_REPORT(
                    context->GetNodeName(),
                    "matmul_h SetBufferSpace failed; propagate the first error reported by the failing stage"),
                return false);
    ret = matmul_h.GetTiling(tilingData.MMParamH);
    OP_CHECK_IF(
        ret == -1,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "matmul_h GetTiling failed; propagate the first error reported by the failing stage"),
        return false);
    matmul_tiling::MatmulApiTiling matmul_w(platformInfo);
    ret = matmul_w.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    OP_CHECK_IF(
        ret == -1,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "matmul_w SetAType failed; propagate the first error reported by the failing stage"),
        return false);
    ret = matmul_w.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    OP_CHECK_IF(
        ret == -1,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "matmul_w SetBType failed; propagate the first error reported by the failing stage"),
        return false);
    ret = matmul_w.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    OP_CHECK_IF(
        ret == -1,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "matmul_w SetCType failed; propagate the first error reported by the failing stage"),
        return false);
    int innerBatchW_ = _Params.innerBatchW;
    if (_Params.innerBatchW == 0) {
        innerBatchW_ = 1;
    }
    ret = matmul_w.SetOrgShape(innerBatchW_ * _Params.inputH, _Params.baseNW, _Params.inputW, NUM_FRACTAL);
    OP_CHECK_IF(
        ret == -1,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "matmul_w SetOrgShape failed; propagate the first error reported by the failing stage"),
        return false);
    ret = matmul_w.SetShape(innerBatchW_ * _Params.inputH, _Params.baseNW, NUM_FRACTAL);
    OP_CHECK_IF(ret == -1,
                OP_LOGE_WITHOUT_REPORT(
                    context->GetNodeName(),
                    "matmul_w Set single shape failed; propagate the first error reported by the failing stage"),
                return false);
    ret = matmul_w.SetBufferSpace(-1, -1, -1);
    OP_CHECK_IF(ret == -1,
                OP_LOGE_WITHOUT_REPORT(
                    context->GetNodeName(),
                    "matmul_w SetBufferSpace failed; propagate the first error reported by the failing stage"),
                return false);
    ret = matmul_w.GetTiling(tilingData.MMParamW);
    OP_CHECK_IF(
        ret == -1,
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "matmul_w GetTiling failed; propagate the first error reported by the failing stage"),
        return false);
    return true;
}
uint32_t UpsampleBicubic2dGradTiling::GetNumPerBlock()
{
    if (_Params.dataType == ge::DataType::DT_FLOAT) {
        return NUM_PER_BLOCK_FLOAT32;
    }
    return NUM_PER_BLOCK_FLOAT16;
}

uint32_t UpsampleBicubic2dGradTiling::GetDtypeSize()
{
    if (_Params.dataType == ge::DataType::DT_FLOAT) {
        return sizeof(float);
    }
    return sizeof(short);
}

bool UpsampleBicubic2dGradTiling::GetClearTilingData()
{
    _Params.clearBaseN = UB_CLEAR_SIZE / GetDtypeSize();
    uint64_t total_count = _Params.batch * _Params.inputH * _Params.outputW;
    uint64_t step_count = _Params.CoreNum * _Params.clearBaseN;

    _Params.clearInterLoop = total_count / step_count;
    uint64_t total_count_tail = total_count % step_count;
    uint64_t total_block_tail = (total_count_tail + GetNumPerBlock() - 1UL) / GetNumPerBlock();
    _Params.clearInterTailN = total_block_tail / _Params.CoreNum * GetNumPerBlock();
    _Params.clearInterTailCoreNum = total_block_tail % _Params.CoreNum;

    total_count = _Params.batch * _Params.outputH * _Params.outputW;

    _Params.clearOutLoop = total_count / step_count;
    total_count_tail = total_count % step_count;
    total_block_tail = (total_count_tail + GetNumPerBlock() - 1UL) / GetNumPerBlock();
    _Params.clearOutTailN = total_block_tail / _Params.CoreNum * GetNumPerBlock();
    _Params.clearOutTailCoreNum = total_block_tail % _Params.CoreNum;
    return true;
}

bool UpsampleBicubic2dGradTiling::GetTilingData(const gert::TilingContext* context)
{
    uint64_t loop_H = (_Params.inputH + NUM_FRACTAL - 1) / NUM_FRACTAL;
    _Params.tailH = ((_Params.inputH - 1) % NUM_FRACTAL) + 1;
    _Params.innerCoreNumH = _Params.CoreNum / loop_H;
    _Params.innerCoreNumH = _Params.innerCoreNumH == 0 ? 1 : _Params.innerCoreNumH;
    _Params.CoreNumH = _Params.CoreNum / _Params.innerCoreNumH;
    _Params.loopH = loop_H / _Params.CoreNumH;
    _Params.loopTailCoreH = loop_H % _Params.CoreNumH;
    _Params.innerBatchH = _Params.batch / _Params.innerCoreNumH;
    _Params.innerBatchTailCoreH = _Params.batch % _Params.innerCoreNumH;

    uint64_t loop_W = (_Params.inputW + NUM_FRACTAL - 1) / NUM_FRACTAL;
    _Params.tailW = ((_Params.inputW - 1) % NUM_FRACTAL) + 1;
    _Params.innerCoreNumW = _Params.CoreNum / loop_W;
    _Params.innerCoreNumW = _Params.innerCoreNumW == 0 ? 1 : _Params.innerCoreNumW;
    _Params.CoreNumW = _Params.CoreNum / _Params.innerCoreNumW;
    _Params.loopW = loop_W / _Params.CoreNumW;
    _Params.loopTailCoreW = loop_W % _Params.CoreNumW;
    _Params.innerBatchW = _Params.batch / _Params.innerCoreNumW;
    _Params.innerBatchTailCoreW = _Params.batch % _Params.innerCoreNumW;

    _Params.baseNH = NUM_FRACTAL * static_cast<uint64_t>(ceil(_Params.scalesH + THRESHOLD));
    _Params.baseNW = NUM_FRACTAL * static_cast<uint64_t>(ceil(_Params.scalesW + THRESHOLD));
    OP_CHECK_IF(
        !GetClearTilingData(),
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "get clear tiling data failed; propagate the first error reported by the failing stage"),
        return ge::GRAPH_FAILED);
    return GetMMTilingData(context);
}

bool UpsampleBicubic2dGradTiling::SetTilingData(gert::TilingContext* context)
{
    tilingData.set_dataType(static_cast<uint32_t>(_Params.dataType));
    tilingData.set_CoreNum(_Params.CoreNum);
    tilingData.set_alignCorners(_Params.alignCorners);
    tilingData.set_scalesH(_Params.scalesH);
    tilingData.set_scalesW(_Params.scalesW);
    tilingData.set_baseNH(_Params.baseNH);
    tilingData.set_baseNW(_Params.baseNW);

    tilingData.set_batch(_Params.batch);
    tilingData.set_inputH(_Params.inputH);
    tilingData.set_inputW(_Params.inputW);
    tilingData.set_outputH(_Params.outputH);
    tilingData.set_outputW(_Params.outputW);

    tilingData.set_tailH(_Params.tailH);
    tilingData.set_CoreNumH(_Params.CoreNumH);
    tilingData.set_loopH(_Params.loopH);
    tilingData.set_loopTailCoreH(_Params.loopTailCoreH);
    tilingData.set_innerCoreNumH(_Params.innerCoreNumH);
    tilingData.set_innerBatchH(_Params.innerBatchH);
    tilingData.set_innerBatchTailCoreH(_Params.innerBatchTailCoreH);

    tilingData.set_tailW(_Params.tailW);
    tilingData.set_CoreNumW(_Params.CoreNumW);
    tilingData.set_loopW(_Params.loopW);
    tilingData.set_loopTailCoreW(_Params.loopTailCoreW);
    tilingData.set_innerCoreNumW(_Params.innerCoreNumW);
    tilingData.set_innerBatchW(_Params.innerBatchW);
    tilingData.set_innerBatchTailCoreW(_Params.innerBatchTailCoreW);

    tilingData.set_clearBaseN(_Params.clearBaseN);
    tilingData.set_clearInterLoop(_Params.clearInterLoop);
    tilingData.set_clearInterTailN(_Params.clearInterTailN);
    tilingData.set_clearInterTailCoreNum(_Params.clearInterTailCoreNum);
    tilingData.set_clearOutLoop(_Params.clearOutLoop);
    tilingData.set_clearOutTailN(_Params.clearOutTailN);
    tilingData.set_clearOutTailCoreNum(_Params.clearOutTailCoreNum);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return true;
}

bool UpsampleBicubic2dGradTiling::SetLaunchInfo(gert::TilingContext* context)
{
    context->SetBlockDim(_Params.CoreNum / NUM_TWO);
    context->SetTilingKey(static_cast<int64_t>(UpsampleBicubic2dGradTilingKey::BASE_MODE));

    int64_t workspaceSize = ((_Params.baseNH > _Params.baseNW) ? _Params.baseNH : _Params.baseNW) * NUM_FRACTAL *
                                _Params.CoreNum * GetDtypeSize() +
                            (_Params.batch * _Params.inputH * _Params.outputW + GetNumPerBlock() - 1) /
                                GetNumPerBlock() * BLOCK_SIZE +
                            _Params.CoreNum * BLOCK_SIZE * NUM_TWO + 16 * 1024 * 1024;
    if (AddWorkspaces(context, workspaceSize)) {
        return true;
    } else {
        return false;
    }
}

bool UpsampleBicubic2dGradTiling::IsDeterministicCalc(const gert::TilingContext* context)
{
    _Params.deterministic = context->GetDeterministic();
    if (_Params.scalesW >= MAX_SCALE || _Params.scalesW < (1 / MAX_SCALE) || _Params.scalesH >= MAX_SCALE ||
        _Params.scalesH < (1 / MAX_SCALE)) {
        return true;
    }
    return _Params.deterministic;
}

bool UpsampleBicubic2dGradTiling::GetTilingDataDC(const gert::TilingContext* context)
{
    _Params.slideSize = NUM_FRACTAL;
    _Params.CoreNumW = 0;
    _Params.CoreNumH = 0;
    _Params.CoreNum = _Params.CoreNum / NUM_TWO;
    CalcScales();
    CalcNeedCoreNum();
    CalcSingleCoreK();
    CalcTCubeTiling(context);

    return true;
}

void UpsampleBicubic2dGradTiling::CalcScales()
{
    _Params.needExpandW = _Params.inputW != _Params.outputW;
    _Params.needExpandH = _Params.inputH != _Params.outputH;
}

void UpsampleBicubic2dGradTiling::CalcNeedCoreNum()
{
    if (_Params.needExpandW) {
        CalcNeedCoreNumW();
    }

    if (_Params.needExpandH || !_Params.needExpandW) {
        CalcNeedCoreNumH();
    }
    _Params.CoreNum = (_Params.CoreNumW > _Params.CoreNumH) ? _Params.CoreNumW : _Params.CoreNumH;
    _Params.CoreNum = _Params.CoreNum > 1 ? _Params.CoreNum : 1;
}

class SplitTilingData {
public:
    int64_t coreNum = 0;

    int64_t perCoreSlideNum = 0; // 每个核至少处理的整块

    int64_t tailStartCoreNum = 0;
    int64_t tailEndCoreNum = 0;
    int64_t perCoreTailSlideNum = 0;
    int64_t extraTailSlideCoreNum = 0;

    int64_t needCoreNum = 0;
    SplitTilingData(int64_t coreNumber, int64_t ele, int64_t otherDimEle, int64_t slideSize)
    {
        this->coreNum = coreNumber;

        this->perCoreSlideNum = ele / (slideSize * this->coreNum) * slideSize;

        this->perCoreTailSlideNum = otherDimEle / this->coreNum;
        this->extraTailSlideCoreNum = otherDimEle - this->perCoreTailSlideNum * this->coreNum;

        if (this->perCoreSlideNum > 0 || this->perCoreTailSlideNum > 0) {
            this->needCoreNum = this->coreNum;
        } else {
            this->needCoreNum = this->extraTailSlideCoreNum;
        }
    }
};

void UpsampleBicubic2dGradTiling::CalcNeedCoreNumW()
{
    SplitTilingData splitTilingData(_Params.CoreNum, _Params.outputW, _Params.inputH * _Params.batch,
                                    _Params.slideSize);

    _Params.perCoreSlideNumW = splitTilingData.perCoreSlideNum;
    _Params.perCoreTailSlideNumW = splitTilingData.perCoreTailSlideNum;
    _Params.extraTailSlideCoreNumW = splitTilingData.extraTailSlideCoreNum;
    _Params.CoreNumW = splitTilingData.needCoreNum;
}

void UpsampleBicubic2dGradTiling::CalcNeedCoreNumH()
{
    SplitTilingData splitTilingData(_Params.CoreNum, _Params.outputH, _Params.outputW, _Params.slideSize);

    _Params.perCoreSlideNumH = splitTilingData.perCoreSlideNum;
    _Params.perCoreTailSlideNumH = splitTilingData.perCoreTailSlideNum;
    _Params.extraTailSlideCoreNumH = splitTilingData.extraTailSlideCoreNum;
    _Params.CoreNumH = splitTilingData.needCoreNum;
}

void UpsampleBicubic2dGradTiling::CalcSingleCoreK()
{
    // 计算singleCoreK,处理时增加余量
    if (!FloatEqual(_Params.scalesW, 0.0f)) {
        _Params.singleCoreKW = int64_t((_Params.slideSize + NUM_FOUR) / _Params.scalesW) + 1;
        _Params.singleCoreKW = _Params.singleCoreKW < _Params.inputW ? _Params.singleCoreKW : _Params.inputW;
    } else {
        _Params.singleCoreKW = _Params.inputW;
    }
    if (!FloatEqual(_Params.scalesH, 0.0f)) {
        _Params.singleCoreKH = int64_t((_Params.slideSize + NUM_FOUR) / _Params.scalesH) + 1;
        _Params.singleCoreKH = _Params.singleCoreKH < _Params.inputH ? _Params.singleCoreKH : _Params.inputH;
    } else {
        _Params.singleCoreKH = _Params.inputH;
    }
}

void UpsampleBicubic2dGradTiling::CalcTCubeTiling(const gert::TilingContext* context)
{
    auto dataType = static_cast<matmul_tiling::DataType>(_Params.dataType);
    matmul_tiling::PlatformInfo platformInfo;
    auto compileInfoPtr = reinterpret_cast<const UpsampleBicubic2dGradCompileInfo*>(context->GetCompileInfo());
    if (compileInfoPtr == nullptr) {
        return;
    }
    InitPlatformInfo(compileInfoPtr, platformInfo);
    _Params.UBSize = platformInfo.ubSize;
    _Params.radioMatrixSize = ((_Params.singleCoreKH > _Params.singleCoreKW) ? _Params.singleCoreKH :
                                                                               _Params.singleCoreKW) *
                              _Params.slideSize;
    // 中间tensor
    _Params.intermediateMatrixSize = _Params.inputN * _Params.inputC * _Params.inputH * _Params.outputW;
    matmul_tiling::MatmulApiTiling matmul_h(platformInfo);
    // matmul_tiling::MatmulApiTiling matmul_h;
    matmul_h.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    matmul_h.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    matmul_h.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    matmul_h.SetOrgShape(_Params.outputH, _Params.outputW, _Params.inputW);
    matmul_h.SetShape(_Params.slideSize, _Params.outputW, _Params.singleCoreKH);

    if (matmul_h.GetTiling(tilingData.MMParamH) == -1) {
        return;
    }

    matmul_tiling::MatmulApiTiling matmul_w(platformInfo);
    matmul_w.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    matmul_w.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    matmul_w.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, dataType);
    matmul_w.SetOrgShape(_Params.batch * _Params.inputH, _Params.outputW, _Params.inputW);
    matmul_w.SetShape(_Params.batch * _Params.inputH, _Params.slideSize, _Params.singleCoreKW);
    if (matmul_w.GetTiling(tilingData.MMParamW) == -1) {
        return;
    }
}

bool UpsampleBicubic2dGradTiling::SetTilingDataDC(gert::TilingContext* context)
{
    tilingData.set_dataType(static_cast<uint32_t>(_Params.dataType));
    tilingData.set_CoreNum(_Params.CoreNum);
    tilingData.set_ubSize(_Params.UBSize);
    tilingData.set_CoreNumW(_Params.CoreNumW);
    tilingData.set_CoreNumH(_Params.CoreNumH);
    tilingData.set_alignCorners(_Params.alignCorners);
    tilingData.set_scalesH(_Params.scalesH);
    tilingData.set_scalesW(_Params.scalesW);
    tilingData.set_singleCoreKW(_Params.singleCoreKW);
    tilingData.set_singleCoreKH(_Params.singleCoreKH);
    tilingData.set_needExpandW(_Params.needExpandW);
    tilingData.set_needExpandH(_Params.needExpandH);

    tilingData.set_batch(_Params.batch);
    tilingData.set_inputN(_Params.inputN);
    tilingData.set_inputC(_Params.inputC);
    tilingData.set_inputH(_Params.inputH);
    tilingData.set_inputW(_Params.inputW);
    tilingData.set_outputH(_Params.outputH);
    tilingData.set_outputW(_Params.outputW);

    tilingData.set_slideSize(_Params.slideSize);
    tilingData.set_radioMatrixSize(_Params.radioMatrixSize);
    tilingData.set_intermediateMatrixSize(_Params.intermediateMatrixSize);

    tilingData.set_perCoreSlideNumW(_Params.perCoreSlideNumW);
    tilingData.set_perCoreTailSlideNumW(_Params.perCoreTailSlideNumW);
    tilingData.set_extraTailSlideCoreNumW(_Params.extraTailSlideCoreNumW);

    tilingData.set_perCoreSlideNumH(_Params.perCoreSlideNumH);
    tilingData.set_perCoreTailSlideNumH(_Params.perCoreTailSlideNumH);
    tilingData.set_extraTailSlideCoreNumH(_Params.extraTailSlideCoreNumH);

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    TilingPrintParam(context);
    return true;
}

void UpsampleBicubic2dGradTiling::TilingPrintParam(const gert::TilingContext* context)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print UpsampleBicubic2dGrad tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, ">>> ubSize %ld", _Params.UBSize);
    OP_LOGD(nodeName, ">>> dataType %d", _Params.dataType);
    OP_LOGD(nodeName, ">>> CoreNum %ld", _Params.CoreNum);
    OP_LOGD(nodeName, ">>> CoreNumW %ld", _Params.CoreNumW);
    OP_LOGD(nodeName, ">>> CoreNumH %ld", _Params.CoreNumH);
    OP_LOGD(nodeName, ">>> alignCorners %d", _Params.alignCorners);
    OP_LOGD(nodeName, ">>> scalesH %f", _Params.scalesH);
    OP_LOGD(nodeName, ">>> scalesW %f", _Params.scalesW);
    OP_LOGD(nodeName, ">>> singleCoreKW %ld", _Params.singleCoreKW);
    OP_LOGD(nodeName, ">>> singleCoreKH %ld", _Params.singleCoreKH);
    OP_LOGD(nodeName, ">>> needExpandW %d", _Params.needExpandW);
    OP_LOGD(nodeName, ">>> needExpandH %d", _Params.needExpandH);

    OP_LOGD(nodeName, ">>> batch %ld", _Params.batch);
    OP_LOGD(nodeName, ">>> inputN %ld", _Params.inputN);
    OP_LOGD(nodeName, ">>> inputC %ld", _Params.inputC);
    OP_LOGD(nodeName, ">>> inputH %ld", _Params.inputH);
    OP_LOGD(nodeName, ">>> inputW %ld", _Params.inputW);
    OP_LOGD(nodeName, ">>> outputH %ld", _Params.outputH);
    OP_LOGD(nodeName, ">>> outputW %ld", _Params.outputW);

    OP_LOGD(nodeName, ">>> perCoreSlideNumW %ld", _Params.perCoreSlideNumW);
    OP_LOGD(nodeName, ">>> perCoreTailSlideNumW %ld", _Params.perCoreTailSlideNumW);
    OP_LOGD(nodeName, ">>> extraTailSlideCoreNumW %ld", _Params.extraTailSlideCoreNumW);
    OP_LOGD(nodeName, ">>> perCoreSlideNumH %ld", _Params.perCoreSlideNumH);
    OP_LOGD(nodeName, ">>> perCoreTailSlideNumH %ld", _Params.perCoreTailSlideNumH);
    OP_LOGD(nodeName, ">>> extraTailSlideCoreNumH %ld", _Params.extraTailSlideCoreNumH);

    OP_LOGD(nodeName, ">>> slideSize %ld", _Params.slideSize);
    OP_LOGD(nodeName, ">>> radioMatrixSize %ld", _Params.radioMatrixSize);
    OP_LOGD(nodeName, ">>> intermediateMatrixSize %ld", _Params.intermediateMatrixSize);
}

bool UpsampleBicubic2dGradTiling::SetLaunchInfoDC(gert::TilingContext* context)
{
    context->SetBlockDim(_Params.CoreNum);
    context->SetTilingKey(static_cast<int64_t>(UpsampleBicubic2dGradTilingKey::DETERMINISTIC_MODE));

    uint64_t workspaceSize = (_Params.intermediateMatrixSize + _Params.radioMatrixSize * _Params.CoreNum * NUM_TWO) *
                                 GetDtypeSize() +
                             16 * 1024 * 1024;
    if (AddWorkspaces(context, workspaceSize)) {
        return true;
    } else {
        return false;
    }
}

ge::graphStatus UpsampleBicubic2dGradTiling::runTiling(gert::TilingContext* context)
{
    OP_CHECK_IF(
        !GetPlatformInfo(context),
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "get platforminfo failed; propagate the first error reported by the failing stage"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!GetCheckAttr(context),
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                                       "check attr failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckInOutShapes(context),
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                                       "check shape failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);
    auto tempGetInputDesc = context->GetInputDesc(0);
    OP_CHECK_IF(
        tempGetInputDesc == nullptr,
        error_report::ReportInvalidArgument(context->GetNodeName(), "inputDesc",
                                            "the input desc of required input inputDesc is missing in the tiling "
                                            "context; check that the graph node provides all required inputs"),
        return ge::GRAPH_FAILED);
    _Params.dataType = tempGetInputDesc->GetDataType();
    OP_CHECK_IF(_Params.dataType != ge::DataType::DT_FLOAT && _Params.dataType != ge::DataType::DT_FLOAT16 &&
                    _Params.dataType != ge::DataType::DT_BF16,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                                       "check dtype failed; propagate the first error reported by the failing stage"),
                return ge::GRAPH_FAILED);

    if (IsDeterministicCalc(context)) {
        OP_CHECK_IF(
            !GetTilingDataDC(context),
            OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                                   "get tiling data failed; propagate the first error reported by the failing stage"),
            return ge::GRAPH_FAILED);
        // tilingdata
        OP_CHECK_IF(
            !SetTilingDataDC(context),
            OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                                   "set tiling data failed; propagate the first error reported by the failing stage"),
            return ge::GRAPH_FAILED);
        // launchinfo: tilingkey, workspace, blockdim
        OP_CHECK_IF(
            !SetLaunchInfoDC(context),
            OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                                   "set launchinfo failed; propagate the first error reported by the failing stage"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            !GetTilingData(context),
            OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                                   "get tiling data failed; propagate the first error reported by the failing stage"),
            return ge::GRAPH_FAILED);
        // tilingdata
        OP_CHECK_IF(
            !SetTilingData(context),
            OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                                   "set tiling data failed; propagate the first error reported by the failing stage"),
            return ge::GRAPH_FAILED);
        // launchinfo: tilingkey, workspace, blockdim
        OP_CHECK_IF(
            !SetLaunchInfo(context),
            OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                                   "set launchinfo failed; propagate the first error reported by the failing stage"),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForUpsampleBicubic2dGrad(gert::TilingContext* context)
{
    UpsampleBicubic2dGradTiling tiling_handle;
    context->SetScheduleMode(SCHEDULE_MODE);
    return tiling_handle.runTiling(context);
}

ge::graphStatus TilingPrepareForUpsampleBicubic2dGrad(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr,
                error_report::ReportTilingStageError(
                    context->GetNodeName(),
                    "platformInfoPtr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<UpsampleBicubic2dGradCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr,
                error_report::ReportTilingStageError(
                    context->GetNodeName(),
                    "compileInfoPtr is null in the tiling context; check the TilingContext construction"),
                return ge::GRAPH_FAILED);

    std::string val;
    platformInfoPtr->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", val);
    OP_CHECK_IF(val.empty(),
                error_report::ReportInvalidArgument(context->GetNodeName(), "supports",
                                                    "upsampleBicubic2dGrad supports only ASCEND910B for now"),
                return false);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    platformInfoPtr->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

    OP_CHECK_IF((compileInfoPtr->aicNum == 0 || compileInfoPtr->aivNum == 0 || compileInfoPtr->ubSize == 0 ||
                 compileInfoPtr->l1Size == 0 || compileInfoPtr->l0CSize == 0 || compileInfoPtr->l0ASize == 0 ||
                 compileInfoPtr->l0BSize == 0),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    context->GetNodeName(), "platform",
                    (std::to_string(compileInfoPtr->aicNum) + ", " + std::to_string(compileInfoPtr->aivNum) + ", " +
                     std::to_string(compileInfoPtr->ubSize) + ", " + std::to_string(compileInfoPtr->l1Size))
                        .c_str(),
                    "platform info is invalid, aicNum=, aivNum=, ubSize=, l1Size=, l0CSize=, l0ASize=, l0BSize="),
                return ge::GRAPH_FAILED);

    OP_LOGI(context->GetNodeName(), "Parse compile info success, soc: %d",
            static_cast<int>(compileInfoPtr->socVersion));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpsampleBicubic2dGrad)
    .Tiling(TilingForUpsampleBicubic2dGrad)
    .TilingParse<UpsampleBicubic2dGradCompileInfo>(TilingPrepareForUpsampleBicubic2dGrad);
} // namespace optiling
