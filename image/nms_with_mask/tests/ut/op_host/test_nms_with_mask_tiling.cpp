/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <vector>
#include "../../../op_host/arch35/nms_with_mask_tiling_arch35.h"
#include "base/context_builder/op_tiling_parse_context_builder.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "platform/platform_infos_def.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

constexpr uint64_t TILING_KEY_FOR_MULTICORE = 10000UL;
constexpr int64_t BIT_PER_BYTE = 8;

class NMSWithMaskTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "NMSWithMaskTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "NMSWithMaskTiling TearDown" << std::endl; }
};

void CalculateTilingData(NMSWithMaskTilingData& tilingData, int64_t boxesNum, int64_t vectorCoreNum, int64_t groupSize)
{
    tilingData.boxesNum = boxesNum;
    tilingData.groupSize = groupSize;
    tilingData.groupNum = (boxesNum + groupSize - 1) / groupSize;
    tilingData.blockNum = tilingData.groupNum * (tilingData.groupNum + 1) / 2;
    tilingData.usedCoreNum = tilingData.blockNum > vectorCoreNum ? vectorCoreNum : tilingData.blockNum;
    tilingData.headCoreNum = tilingData.blockNum % tilingData.usedCoreNum;
    tilingData.headCoreNum = tilingData.headCoreNum == 0 ? tilingData.usedCoreNum : tilingData.headCoreNum;
    tilingData.blockPerHead = (tilingData.blockNum + tilingData.usedCoreNum - 1) / tilingData.usedCoreNum;
}

void ExecuteNMSWithMaskTilingCase(uint32_t boxesNum, ge::graphStatus expectedResult, const string& expectedTilingData,
                                  const std::vector<size_t>& expectedWorkspaces)
{
    const uint32_t m_dim = 5;
    gert::StorageShape box_scores = {{boxesNum, m_dim}, {boxesNum, m_dim}};
    gert::StorageShape selected_boxes = {{boxesNum, m_dim}, {boxesNum, m_dim}};
    gert::StorageShape selected_idx = {{boxesNum}, {boxesNum}};
    gert::StorageShape selected_mask = {{boxesNum}, {boxesNum}};
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};

    gert::TilingContextPara tilingContextPara(
        "NMSWithMask", {{box_scores, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{selected_boxes, ge::DT_FLOAT, ge::FORMAT_ND},
         {selected_idx, ge::DT_INT32, ge::FORMAT_ND},
         {selected_mask, ge::DT_UINT8, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("iou_threshold", Ops::Cv::AnyValue::CreateFrom<float>(0.5))}, &compileInfo);

    uint64_t expectedTilingKey = (expectedResult == ge::GRAPH_SUCCESS) ? TILING_KEY_FOR_MULTICORE : 0;
    ExecuteTestCase(tilingContextPara, expectedResult, expectedTilingKey, expectedTilingData, expectedWorkspaces);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_test_float32_case1)
{
    const uint32_t boxesNum = 16;
    const uint32_t m_dim = 5;
    gert::StorageShape box_scores = {{boxesNum, m_dim}, {boxesNum, m_dim}};
    gert::StorageShape selected_boxes = {{boxesNum, m_dim}, {boxesNum, m_dim}};
    gert::StorageShape selected_idx = {{boxesNum}, {boxesNum}};
    gert::StorageShape selected_mask = {{boxesNum}, {boxesNum}};

    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    NMSWithMaskTilingData tilingData;
    CalculateTilingData(tilingData, boxesNum, compileInfo.coreNum, 256);

    gert::TilingContextPara tilingContextPara(
        "NMSWithMask", {{box_scores, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{selected_boxes, ge::DT_FLOAT, ge::FORMAT_ND},
         {selected_idx, ge::DT_INT32, ge::FORMAT_ND},
         {selected_mask, ge::DT_UINT8, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("iou_threshold", Ops::Cv::AnyValue::CreateFrom<float>(0.5))}, &compileInfo);

    uint64_t expectedTilingKey = TILING_KEY_FOR_MULTICORE;
    string expectTilingData = "16 1 256 1 1 1 1 1056964608 ";
    // size_t totalWorkspaceBytes = tilingData.blockNum * tilingData.groupSize * tilingData.groupSize / BIT_PER_BYTE;
    std::vector<size_t> expectWorkspaces = {4294975487};
    // expectWorkspaces[0] += totalWorkspaceBytes;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectedTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_accepts_one_below_documented_max_boxes_num)
{
    ExecuteNMSWithMaskTilingCase(39935, ge::GRAPH_SUCCESS, "39935 48 256 156 12246 6 256 1056964608 ", {4395286527});
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_accepts_documented_max_boxes_num)
{
    ExecuteNMSWithMaskTilingCase(39936, ge::GRAPH_SUCCESS, "39936 48 256 156 12246 6 256 1056964608 ", {4395286527});
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_one_above_documented_max_boxes_num)
{
    ExecuteNMSWithMaskTilingCase(39937, ge::GRAPH_FAILED, "", {});
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_nan_iou_threshold)
{
    const uint32_t boxesNum = 16;
    const uint32_t m_dim = 5;
    gert::StorageShape box_scores = {{boxesNum, m_dim}, {boxesNum, m_dim}};
    gert::StorageShape selected_boxes = {{boxesNum, m_dim}, {boxesNum, m_dim}};
    gert::StorageShape selected_idx = {{boxesNum}, {boxesNum}};
    gert::StorageShape selected_mask = {{boxesNum}, {boxesNum}};

    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    const float nanThreshold = std::numeric_limits<float>::quiet_NaN();
    gert::TilingContextPara tilingContextPara(
        "NMSWithMask", {{box_scores, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{selected_boxes, ge::DT_FLOAT, ge::FORMAT_ND},
         {selected_idx, ge::DT_INT32, ge::FORMAT_ND},
         {selected_mask, ge::DT_UINT8, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("iou_threshold", Ops::Cv::AnyValue::CreateFrom<float>(nanThreshold))},
        &compileInfo);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

namespace {
using TensorDesc = gert::TilingContextPara::TensorDescription;
using OpAttr = gert::TilingContextPara::OpAttr;

TensorDesc Make2d(int64_t rows, int64_t cols, ge::DataType dtype = ge::DT_FLOAT)
{
    return {{{rows, cols}, {rows, cols}}, dtype, ge::FORMAT_ND};
}

TensorDesc Make3d(int64_t d0, int64_t d1, int64_t d2, ge::DataType dtype = ge::DT_FLOAT)
{
    return {{{d0, d1, d2}, {d0, d1, d2}}, dtype, ge::FORMAT_ND};
}

TensorDesc Make1d(int64_t len, ge::DataType dtype) { return {{{len}, {len}}, dtype, ge::FORMAT_ND}; }

std::vector<OpAttr> IouAttrs(float value = 0.5f)
{
    return {OpAttr("iou_threshold", Ops::Cv::AnyValue::CreateFrom<float>(value))};
}

std::vector<TensorDesc> DefaultOutputs(int64_t boxesNum, ge::DataType dtype = ge::DT_FLOAT)
{
    return {Make2d(boxesNum, 5, dtype), Make1d(boxesNum, ge::DT_INT32), Make1d(boxesNum, ge::DT_UINT8)};
}

void ExpectTilingFailure(const std::vector<TensorDesc>& inputs, const std::vector<TensorDesc>& outputs,
                         const std::vector<OpAttr>& attrs, void* compileInfo, uint64_t tilingDataSize = 4096)
{
    gert::TilingContextPara para("NMSWithMask", inputs, outputs, attrs, compileInfo, "Ascend950", 64, 262144,
                                 tilingDataSize);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

} // namespace

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_input_rank_3d)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make3d(16, 5, 1)}, DefaultOutputs(16), IouAttrs(), &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_zero_boxes)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(0, 5)}, DefaultOutputs(0), IouAttrs(), &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_input_second_dim_not_5)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(16, 4)}, DefaultOutputs(16), IouAttrs(), &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_selected_boxes_rank_3d)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(16, 5)}, {Make3d(16, 5, 1), Make1d(16, ge::DT_INT32), Make1d(16, ge::DT_UINT8)},
                        IouAttrs(), &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_selected_boxes_dim0_mismatch)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(16, 5)}, {Make2d(15, 5), Make1d(16, ge::DT_INT32), Make1d(16, ge::DT_UINT8)},
                        IouAttrs(), &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_selected_boxes_dim1_mismatch)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(16, 5)}, {Make2d(16, 4), Make1d(16, ge::DT_INT32), Make1d(16, ge::DT_UINT8)},
                        IouAttrs(), &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_selected_idx_rank_2d)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(16, 5)}, {Make2d(16, 5), Make2d(16, 1, ge::DT_INT32), Make1d(16, ge::DT_UINT8)},
                        IouAttrs(), &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_selected_idx_dim0_mismatch)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(16, 5)}, {Make2d(16, 5), Make1d(15, ge::DT_INT32), Make1d(16, ge::DT_UINT8)},
                        IouAttrs(), &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_selected_mask_rank_2d)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(16, 5)}, {Make2d(16, 5), Make1d(16, ge::DT_INT32), Make2d(16, 1, ge::DT_UINT8)},
                        IouAttrs(), &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_selected_mask_dim0_mismatch)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(16, 5)}, {Make2d(16, 5), Make1d(16, ge::DT_INT32), Make1d(15, ge::DT_UINT8)},
                        IouAttrs(), &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_input_dtype)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(16, 5, ge::DT_INT32)}, DefaultOutputs(16), IouAttrs(), &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_selected_boxes_dtype_mismatch)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(16, 5)},
                        {Make2d(16, 5, ge::DT_FLOAT16), Make1d(16, ge::DT_INT32), Make1d(16, ge::DT_UINT8)}, IouAttrs(),
                        &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_selected_idx_dtype)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(16, 5)}, {Make2d(16, 5), Make1d(16, ge::DT_FLOAT), Make1d(16, ge::DT_UINT8)},
                        IouAttrs(), &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_selected_mask_dtype)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(16, 5)}, {Make2d(16, 5), Make1d(16, ge::DT_INT32), Make1d(16, ge::DT_INT32)},
                        IouAttrs(), &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_missing_iou_attr)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(16, 5)}, DefaultOutputs(16), {}, &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_negative_iou_attr)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(16, 5)}, DefaultOutputs(16), IouAttrs(-0.1f), &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_iou_attr_above_one)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(16, 5)}, DefaultOutputs(16), IouAttrs(1.1f), &compileInfo);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_rejects_zero_tiling_data_capacity)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    ExpectTilingFailure({Make2d(16, 5)}, DefaultOutputs(16), IouAttrs(), &compileInfo, 0);
}

namespace {

ge::graphStatus RunTilingDirect(bool withCompileInfo, bool withTilingData, bool withWorkspace, bool withAttrs)
{
    fe::PlatFormInfos platform;
    platform.Init();
    std::map<std::string, std::string> socInfo = {{"ai_core_cnt", "64"}, {"core_type_list", "AICore"}};
    platform.SetPlatformRes("SoCInfo", socInfo);
    platform.SetCoreNumByCoreType("AICore");
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    auto tilingData = gert::TilingData::CreateCap(4096);
    auto workspace = gert::ContinuousVector::Create<size_t>(1);
    gert::Tensor input({{16, 5}, {16, 5}}, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                       gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor selectedBoxes({{16, 5}, {16, 5}},
                               gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                               gert::TensorPlacement::kOnHost, ge::DT_FLOAT, nullptr);
    gert::Tensor selectedIdx({{16}, {16}}, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                             gert::TensorPlacement::kOnHost, ge::DT_INT32, nullptr);
    gert::Tensor selectedMask({{16}, {16}}, gert::StorageFormat(ge::FORMAT_ND, ge::FORMAT_ND, gert::ExpandDimsType()),
                              gert::TensorPlacement::kOnHost, ge::DT_UINT8, nullptr);
    auto faker = gert::TilingContextFaker();
    faker.SetOpType("NMSWithMask")
        .NodeIoNum(1, 3)
        .InputTensors({&input})
        .OutputTensors({&selectedBoxes, &selectedIdx, &selectedMask})
        .PlatformInfo(&platform);
    faker.CompileInfo(withCompileInfo ? &compileInfo : nullptr);
    if (withAttrs) {
        faker.Attr("iou_threshold", 0.5f);
    }
    if (withTilingData) {
        faker.TilingData(tilingData.get());
    }
    if (withWorkspace) {
        faker.Workspace(reinterpret_cast<gert::ContinuousVector*>(workspace.get()));
    }
    auto holder = faker.Build();
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto impl = registry->GetOpImpl("NMSWithMask");
    EXPECT_NE(impl, nullptr);
    EXPECT_NE(impl->tiling, nullptr);
    return impl->tiling(holder.GetContext());
}
} // namespace

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_entry_rejects_null_context)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto impl = registry->GetOpImpl("NMSWithMask");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->tiling, nullptr);
    EXPECT_EQ(impl->tiling(nullptr), ge::GRAPH_FAILED);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_set_tiling_rejects_null_compile_info)
{
    EXPECT_EQ(RunTilingDirect(false, true, true, true), ge::GRAPH_FAILED);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_set_tiling_rejects_null_attrs)
{
    EXPECT_EQ(RunTilingDirect(true, true, true, false), ge::GRAPH_FAILED);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_set_tiling_rejects_null_workspace)
{
    EXPECT_EQ(RunTilingDirect(true, true, false, true), ge::GRAPH_FAILED);
}

namespace {
ge::graphStatus CallTilingParseProbe(const void* compileInfo, const void* platformInfo)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto impl = registry->GetOpImpl("NMSWithMask");
    EXPECT_NE(impl, nullptr);
    EXPECT_NE(impl->tiling_parse, nullptr);
    auto parse = reinterpret_cast<ge::graphStatus (*)(gert::TilingParseContext*)>(impl->tiling_parse);
    gert::OpTilingParseContextBuilder builder;
    builder.OpType(ge::AscendString("NMSWithMask"))
        .OpName(ge::AscendString("NMSWithMask"))
        .IONum(1, 1)
        .CompiledJson("{}")
        .CompiledInfo(compileInfo)
        .PlatformInfo(platformInfo);
    auto holder = builder.Build();
    return parse(holder.GetContext());
}
} // namespace

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_parse_probe)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    fe::PlatFormInfos platform;
    platform.Init();
    std::map<std::string, std::string> socInfo = {
        {"ai_core_cnt", "64"}, {"vector_core_cnt", "64"}, {"core_type_list", "AICore"}};
    std::map<std::string, std::string> aiCoreSpec = {{"ub_size", "262144"}};
    std::map<std::string, std::string> version = {{"NpuArch", "3510"}, {"Short_SoC_version", "Ascend950"}};
    platform.SetPlatformRes("SoCInfo", socInfo);
    platform.SetPlatformRes("AICoreSpec", aiCoreSpec);
    platform.SetPlatformRes("VectorCoreSpec", aiCoreSpec);
    platform.SetCoreNumByCoreType("AICore");
    platform.SetPlatformRes("version", version);
    EXPECT_EQ(CallTilingParseProbe(&compileInfo, &platform), ge::GRAPH_SUCCESS);
    EXPECT_GT(compileInfo.coreNum, 0);
    EXPECT_GT(compileInfo.ubSize, 0);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_parse_probe_null_compile_info)
{
    fe::PlatFormInfos platform;
    platform.Init();
    EXPECT_EQ(CallTilingParseProbe(nullptr, &platform), ge::GRAPH_FAILED);
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_parse_probe_invalid_platform)
{
    optiling::NMSWithMaskCompileInfo compileInfo = {100, 48, 196608};
    fe::PlatFormInfos platform;
    platform.Init();
    EXPECT_EQ(CallTilingParseProbe(&compileInfo, &platform), ge::GRAPH_FAILED);
}
