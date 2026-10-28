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
 * \file three_interpolate_backward_tiling.cc
 * \brief
 */
#include <cstdint>
#include <sstream>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "log/log.h"
#include "util/math_util.h"

#include "platform/platform_info.h"
#include "three_interpolate_backward_tiling.h"
#include "op_host/tiling_util.h"

namespace {
constexpr uint32_t C0 = 16;
constexpr uint32_t N0 = 16;
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t RATIO_TILING = 6;
constexpr uint32_t BLOCK_BYTE_SIZE = 32;
constexpr uint32_t INDEX_INPUT_GRAD_X = 0u;
constexpr uint32_t INDEX_INPUT_IDX = 1u;
constexpr uint32_t INDEX_INPUT_WEIGHT = 2u;
constexpr uint32_t OUTPUT_GRAD_Y = 0u;
constexpr uint32_t CONST_3 = 3u;
constexpr uint32_t WORKSPACE_16MBYTE_SIZE = 16u * 1024u * 1024u;

enum class DIR : uint32_t { DIR_INPUT, DIR_OUTPUT, DIR_MAX };
enum class DIM : size_t { DIM_0, DIM_1, DIM_2, DIM_3, DIM_4 };
enum class MULTI_CORE_MODE : uint32_t { MOLTI_CORE_MODE_N_AXIS, MOLTI_CORE_MODE_B_AXIS };

inline static uint32_t CeilDiv(uint32_t value, uint32_t factor)
{
    return (factor == 0) ? value : (value + factor - 1) / factor;
}

} // namespace

namespace optiling {

void TilingDebugPrint(gert::TilingContext* context, ThreeInterpolateBackwardTilingData& tiling_data)
{
    OP_LOGD(context, "ThreeInterpolateBackward TilingResultDebugPrint.");
    OP_LOGD(context, "tiling:%lu.", context->GetTilingKey());
    OP_LOGD(context, "%s:%u.", "used_core_num", tiling_data.get_used_core_num());
    OP_LOGD(context, "%s:%u.", "bs", tiling_data.get_bs());
    OP_LOGD(context, "%s:%u.", "c1", tiling_data.get_c1());
    OP_LOGD(context, "%s:%u.", "ms", tiling_data.get_ms());
    OP_LOGD(context, "%s:%u.", "ns", tiling_data.get_ns());
    OP_LOGD(context, "%s:%u.", "each_core_proc_num", tiling_data.get_each_core_proc_num());
    OP_LOGD(context, "%s:%u.", "each_core_loop_times", tiling_data.get_each_core_loop_times());
    OP_LOGD(context, "%s:%u.", "each_core_each_loop_n_cnt", tiling_data.get_each_core_each_loop_n_cnt());
    OP_LOGD(context, "%s:%u.", "each_core_last_loop_n_cnt", tiling_data.get_each_core_last_loop_n_cnt());
    OP_LOGD(context, "%s:%u.", "last_core_proc_num", tiling_data.get_last_core_proc_num());
    OP_LOGD(context, "%s:%u.", "last_core_loop_times", tiling_data.get_last_core_loop_times());
    OP_LOGD(context, "%s:%u.", "last_core_each_loop_n_cnt", tiling_data.get_last_core_each_loop_n_cnt());
    OP_LOGD(context, "%s:%u.", "last_core_last_loop_n_cnt", tiling_data.get_last_core_last_loop_n_cnt());
    OP_LOGD(context, "%s:%u.", "weight_move_block_size", tiling_data.get_weight_move_block_size());
    OP_LOGD(context, "%s:%u.", "idx_move_block_size", tiling_data.get_idx_move_block_size());
    OP_LOGD(context, "%s:%u.", "grad_x_move_block_size", tiling_data.get_grad_x_move_block_size());
    OP_LOGD(context, "%s:%u.", "grad_y_move_block_size", tiling_data.get_grad_y_move_block_size());
    OP_LOGD(context, "%s:%u.", "c_move_num", tiling_data.get_c_move_num());
    OP_LOGD(context, "%s:%u.", "c_last_loop_move_num", tiling_data.get_c_last_loop_move_num());
    OP_LOGD(context, "%s:%u.", "c_move_loop_times", tiling_data.get_c_move_loop_times());
    OP_LOGD(context, "%s:%u.", "multi_core_mode", tiling_data.get_mulit_core_mode());
    OP_LOGD(context, "%s:%u.", "each_core_proc_batch_num", tiling_data.get_each_core_proc_batch_num());
    OP_LOGD(context, "%s:%u.", "core_proc_batch_padding_idx", tiling_data.get_core_proc_batch_padding_idx());
}

template <typename DATA_T, typename IDX_T>
ge::graphStatus TilingProcess(ThreeInterpolateBackwardTilingData& tiling_host, const uint32_t& aicore_num,
                              const int64_t& ub_platform_byte_size, bool force_multi_core_mode_1_flag,
                              gert::TilingContext* context)
{
    // 实现tiling 逻辑
    OP_LOGD(context, "aicore_num %u ub_platform_byte_size:%ld.", aicore_num, ub_platform_byte_size);
    if (tiling_host.get_bs() >= RATIO_TILING * tiling_host.get_ns() || force_multi_core_mode_1_flag) {
        tiling_host.set_mulit_core_mode(static_cast<uint32_t>(MULTI_CORE_MODE::MOLTI_CORE_MODE_B_AXIS));
        tiling_host.set_used_core_num((tiling_host.get_bs() <= aicore_num) ? tiling_host.get_bs() : aicore_num);
        tiling_host.set_each_core_proc_batch_num(tiling_host.get_bs() / tiling_host.get_used_core_num());
        tiling_host.set_core_proc_batch_padding_idx(tiling_host.get_bs() % tiling_host.get_used_core_num());
        tiling_host.set_each_core_proc_num(tiling_host.get_ns());
        tiling_host.set_last_core_proc_num(tiling_host.get_ns());
    } else {
        tiling_host.set_mulit_core_mode(static_cast<uint32_t>(MULTI_CORE_MODE::MOLTI_CORE_MODE_N_AXIS));
        tiling_host.set_used_core_num((tiling_host.get_ns() <= aicore_num) ? tiling_host.get_ns() : aicore_num);
        tiling_host.set_each_core_proc_num(tiling_host.get_ns() / tiling_host.get_used_core_num());
        tiling_host.set_last_core_proc_num(tiling_host.get_ns() - tiling_host.get_each_core_proc_num() *
                                                                      (tiling_host.get_used_core_num() - 1));
    }

    // double buffer
    uint32_t ub_free_byte_size = ub_platform_byte_size / BUFFER_NUM;

    // 单核循环搬运次数为
    tiling_host.set_each_core_loop_times(CeilDiv(tiling_host.get_each_core_proc_num(), N0));
    tiling_host.set_each_core_each_loop_n_cnt(N0);
    tiling_host.set_each_core_last_loop_n_cnt(tiling_host.get_each_core_proc_num() -
                                              N0 * (tiling_host.get_each_core_loop_times() - 1));

    // 最后一个核循环搬运次数为
    tiling_host.set_last_core_loop_times(CeilDiv(tiling_host.get_last_core_proc_num(), N0));
    tiling_host.set_last_core_each_loop_n_cnt(N0);
    tiling_host.set_last_core_last_loop_n_cnt(tiling_host.get_last_core_proc_num() -
                                              N0 * (tiling_host.get_last_core_loop_times() - 1));

    tiling_host.set_weight_move_block_size(CONST_3 * N0 * sizeof(DATA_T) / BLOCK_BYTE_SIZE);
    tiling_host.set_idx_move_block_size(CONST_3 * N0 * sizeof(IDX_T) / BLOCK_BYTE_SIZE);

    ub_free_byte_size = ub_free_byte_size -
                        (tiling_host.get_weight_move_block_size() + CONST_3 * C0 * N0 * sizeof(DATA_T));

    tiling_host.set_c_move_num(ub_free_byte_size / (C0 * N0 * sizeof(DATA_T) + CONST_3 * C0 * N0 * sizeof(DATA_T)));
    tiling_host.set_c_move_num(tiling_host.get_c_move_num() >= tiling_host.get_c1() ? tiling_host.get_c1() :
                                                                                      tiling_host.get_c_move_num());
    if (tiling_host.get_c_move_num() == 0) {
        OP_LOGD(context, "c_move_num is 0.");
        return ge::GRAPH_FAILED;
    }

    tiling_host.set_c_move_loop_times(CeilDiv(tiling_host.get_c1(), tiling_host.get_c_move_num()));
    tiling_host.set_c_last_loop_move_num(tiling_host.get_c1() -
                                         (tiling_host.get_c_move_num() * (tiling_host.get_c_move_loop_times() - 1)));

    tiling_host.set_grad_x_move_block_size(tiling_host.get_c_move_num() * N0 * C0 * sizeof(DATA_T) / BLOCK_BYTE_SIZE);
    tiling_host.set_grad_y_move_block_size(CONST_3 * tiling_host.get_grad_x_move_block_size());

    auto total_used_bytes = (tiling_host.get_weight_move_block_size() + tiling_host.get_idx_move_block_size() +
                             tiling_host.get_grad_x_move_block_size() + tiling_host.get_grad_y_move_block_size()) *
                            BLOCK_BYTE_SIZE;
    if (total_used_bytes > ub_platform_byte_size) {
        OP_LOGD(context, "ub insufficient.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingGetDimSizeByIndex(gert::TilingContext* context, uint32_t index, DIM dim, DIR dir,
                                        uint32_t& dim_size)
{
    auto shape_ptr = (dir == DIR::DIR_INPUT) ? context->GetInputShape(index) : context->GetOutputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_ptr);

    auto shape_val = shape_ptr->GetStorageShape();
    OP_CHECK_IF(static_cast<size_t>(dim) > shape_val.GetDimNum(), OP_LOGE(context, "shape_val limit error."),
                return ge::GRAPH_FAILED);

    int64_t dim_val = shape_val.GetDim(static_cast<size_t>(dim));
    // 维度超过 int32 承载范围时，写入 uint32 tiling 字段会隐式截断，将回绕后的错误维度下发
    // kernel 导致寻址错位且静默产出错误结果，必须在此显式拒绝（issue #1436）
    OP_CHECK_IF(dim_val < 0 || dim_val > INT32_MAX, OP_LOGE(context, "dim size %ld is invalid.", dim_val),
                return ge::GRAPH_FAILED);
    dim_size = static_cast<uint32_t>(dim_val);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingBasicShapeInfo(gert::TilingContext* context, ThreeInterpolateShapeInfo& shape_info)
{
    ge::graphStatus ret = ge::GRAPH_FAILED;
    ret = TilingGetDimSizeByIndex(context, INDEX_INPUT_GRAD_X, DIM::DIM_0, DIR::DIR_INPUT, shape_info.bs);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "TilingGetDimSizeByIndex failed."), return ret);

    ret = TilingGetDimSizeByIndex(context, OUTPUT_GRAD_Y, DIM::DIM_2, DIR::DIR_OUTPUT, shape_info.ms);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "TilingGetDimSizeByIndex failed."), return ret);

    ret = TilingGetDimSizeByIndex(context, INDEX_INPUT_GRAD_X, DIM::DIM_1, DIR::DIR_INPUT, shape_info.c1);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "TilingGetDimSizeByIndex failed."), return ret);

    ret = TilingGetDimSizeByIndex(context, INDEX_INPUT_GRAD_X, DIM::DIM_2, DIR::DIR_INPUT, shape_info.ns);
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "TilingGetDimSizeByIndex failed."), return ret);

    return ge::GRAPH_SUCCESS;
}

// 校验 idx/weight (bs, ns, 3) 与 grad_x (bs, c1, ns)、grad_y (bs, c1, ms) 的形状一致性（issue #1407）。
// 形状不一致时 kernel 会按 tiling 下发的维度静默截断计算并产出错误梯度，必须在 tiling 侧拦截。
// grad_y 的 rank 与 m 维合法性已由 TilingCheckOutputMs 保证，此处只比对前三维维度值。
ge::graphStatus TilingCheckShapeConsistency(gert::TilingContext* context, const ThreeInterpolateShapeInfo& shape_info)
{
    auto idx_shape_ptr = context->GetInputShape(INDEX_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, idx_shape_ptr);
    auto weight_shape_ptr = context->GetInputShape(INDEX_INPUT_WEIGHT);
    OP_CHECK_NULL_WITH_CONTEXT(context, weight_shape_ptr);
    auto grad_y_shape_ptr = context->GetOutputShape(OUTPUT_GRAD_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_y_shape_ptr);

    const auto& idx_shape = idx_shape_ptr->GetStorageShape();
    const auto& weight_shape = weight_shape_ptr->GetStorageShape();
    const auto& grad_y_shape = grad_y_shape_ptr->GetStorageShape();

    OP_CHECK_IF(idx_shape.GetDimNum() != CONST_3 || weight_shape.GetDimNum() != CONST_3,
                OP_LOGE(context, "idx and weight must be 3-dim tensor."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(idx_shape.GetDim(0) != weight_shape.GetDim(0) || idx_shape.GetDim(1) != weight_shape.GetDim(1) ||
                    idx_shape.GetDim(2) != weight_shape.GetDim(2),
                OP_LOGE(context, "idx and weight must have same shape."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(idx_shape.GetDim(0) != static_cast<int64_t>(shape_info.bs) ||
                    idx_shape.GetDim(1) != static_cast<int64_t>(shape_info.ns) ||
                    idx_shape.GetDim(2) != static_cast<int64_t>(CONST_3),
                OP_LOGE(context, "idx/weight shape must be (bs, ns, 3) to match grad_x."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(grad_y_shape.GetDim(0) != static_cast<int64_t>(shape_info.bs) ||
                    grad_y_shape.GetDim(1) != static_cast<int64_t>(shape_info.c1) ||
                    grad_y_shape.GetDim(2) != static_cast<int64_t>(shape_info.ms),
                OP_LOGE(context, "grad_y shape must be (bs, c1, ms) to match grad_x and attr m."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 5HD 的 m 来自 grad_y 的 H 维（dim2）。不在 InferShape 拦截，这里拒绝 m<=0。
ge::graphStatus TilingCheckOutputMs(gert::TilingContext* context)
{
    auto shape_ptr = context->GetOutputShape(OUTPUT_GRAD_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_ptr);
    const auto& shape_val = shape_ptr->GetStorageShape();
    OP_CHECK_IF(shape_val.GetDimNum() <= static_cast<size_t>(DIM::DIM_2),
                OP_LOGE(context, "grad_y dim num is invalid."), return ge::GRAPH_FAILED);
    int64_t ms = shape_val.GetDim(static_cast<size_t>(DIM::DIM_2));
    OP_CHECK_IF(ms <= 0, OP_LOGE(context, "attr m must be > 0."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void SaveTilingDataToBuffer(gert::TilingContext* context, ThreeInterpolateBackwardTilingData& tiling_host)
{
    tiling_host.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling_host.GetDataSize());
}

ge::graphStatus TilingMainProcess(ThreeInterpolateBackwardTilingData& tiling_host,
                                  const ThreeInterpolateBackwardCompileInfo* compileInfo, gert::TilingContext* context,
                                  ge::DataType grad_x_dtype, ge::DataType idx_dtype)
{
    uint64_t tiling_key = 0;
    auto ret = context->GetDeterministic();
    auto aicore_num_used = (ret == 0) ? compileInfo->aicore_num : 1;
    ge::graphStatus tiling_ret = ge::GRAPH_FAILED;
    if (grad_x_dtype == ge::DT_FLOAT && idx_dtype == ge::DT_INT32) {
        tiling_ret = TilingProcess<float, int32_t>(tiling_host, aicore_num_used, compileInfo->ub_platform_byte_size,
                                                   false, context);

        tiling_key = static_cast<uint64_t>(ThreeInterpolateBackwardTilingType::TILING_MODE_FP32_INT32);
    } else if (grad_x_dtype == ge::DT_FLOAT && idx_dtype == ge::DT_INT64) {
        tiling_ret = TilingProcess<float, int64_t>(tiling_host, aicore_num_used, compileInfo->ub_platform_byte_size,
                                                   false, context);

        tiling_key = static_cast<uint64_t>(ThreeInterpolateBackwardTilingType::TILING_MODE_FP32_INT64);
    } else if (grad_x_dtype == ge::DT_FLOAT16 && idx_dtype == ge::DT_INT32) {
        tiling_ret = TilingProcess<uint16_t, int32_t>(tiling_host, aicore_num_used, compileInfo->ub_platform_byte_size,
                                                      true, context);

        tiling_key = static_cast<uint64_t>(ThreeInterpolateBackwardTilingType::TILING_MODE_FP16_INT32);
    } else if (grad_x_dtype == ge::DT_FLOAT16 && idx_dtype == ge::DT_INT64) {
        tiling_ret = TilingProcess<uint16_t, int64_t>(tiling_host, aicore_num_used, compileInfo->ub_platform_byte_size,
                                                      true, context);

        tiling_key = static_cast<uint64_t>(ThreeInterpolateBackwardTilingType::TILING_MODE_FP16_INT64);
    } else {
        OP_LOGD(context, "ThreeInterpolateBackward data type is not supported.");
        return ge::GRAPH_FAILED;
    }

    context->SetTilingKey(tiling_key);
    OP_CHECK_IF(tiling_ret != ge::GRAPH_SUCCESS, OP_LOGE(context, "TilingProcess failed."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4ThreeInterpolateBackward(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4ThreeInterpolateBackward running begin");

    // ascend950（arch35）regbase 芯片分发到 ND 确定性行私有 tiling
    if (Ops::Cv::OpTiling::IsRegbaseSocVersion(context)) {
        OP_LOGI(context, "Enter Tiling4ThreeInterpolateBackwardRegbase");
        return Tiling4ThreeInterpolateBackwardRegbase(context);
    }

    auto grad_x_desc = context->GetInputDesc(INDEX_INPUT_GRAD_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, grad_x_desc);

    const auto& grad_x_format = static_cast<ge::Format>(ge::GetPrimaryFormat(grad_x_desc->GetStorageFormat()));
    OP_CHECK_IF(grad_x_format != ge::FORMAT_NC1HWC0, OP_LOGE(context, "input format invalid, tiling failed."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(TilingCheckOutputMs(context) != ge::GRAPH_SUCCESS, OP_LOGE(context, "TilingCheckOutputMs failed."),
                return ge::GRAPH_FAILED);

    auto idx_desc = context->GetInputDesc(INDEX_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, idx_desc);

    ThreeInterpolateShapeInfo shape_info;
    OP_CHECK_IF(TilingBasicShapeInfo(context, shape_info) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "TilingBasicShapeInfo failed."), return ge::GRAPH_FAILED);

    OP_CHECK_IF(TilingCheckShapeConsistency(context, shape_info) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "TilingCheckShapeConsistency failed."), return ge::GRAPH_FAILED);

    ThreeInterpolateBackwardTilingData tiling_host;
    tiling_host.set_bs(shape_info.bs);
    tiling_host.set_c1(shape_info.c1);
    tiling_host.set_ms(shape_info.ms);
    tiling_host.set_ns(shape_info.ns);

    auto compileInfo = reinterpret_cast<const ThreeInterpolateBackwardCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    OP_CHECK_IF(TilingMainProcess(tiling_host, compileInfo, context, grad_x_desc->GetDataType(),
                                  idx_desc->GetDataType()) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "tiling failed."), return ge::GRAPH_FAILED);

    size_t sysWorkspaceSize = WORKSPACE_16MBYTE_SIZE;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

    SaveTilingDataToBuffer(context, tiling_host);
    context->SetBlockDim(tiling_host.get_used_core_num());
    TilingDebugPrint(context, tiling_host);
    OP_LOGD(context, "Tiling4ThreeInterpolateBackward running end");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ThreeInterpolateBackward(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<ThreeInterpolateBackwardCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aicore_num = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->aicore_num <= 0),
                OP_LOGE(context, "TilingPrepare4ThreeInterpolateBackward failed to get core num."),
                return ge::GRAPH_FAILED);

    uint64_t ub_platform_byte_size = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_platform_byte_size);
    compileInfo->ub_platform_byte_size = static_cast<int64_t>(ub_platform_byte_size);
    OP_CHECK_IF((compileInfo->ub_platform_byte_size <= 0),
                OP_LOGE(context, "TilingPrepare4ThreeInterpolateBackward failed to get ub size."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ThreeInterpolateBackward)
    .Tiling(Tiling4ThreeInterpolateBackward)
    .TilingParse<ThreeInterpolateBackwardCompileInfo>(TilingPrepare4ThreeInterpolateBackward);
} // namespace optiling
