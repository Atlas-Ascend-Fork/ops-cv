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
 * \file test_integral_image_tiling.cpp
 * \brief IntegralImage Tiling UT
 *
 * 覆盖点：
 *   - two-phase（2D 且 W >= 128）与 legacy（小 W / 3D HWC）两条分支
 *   - tilingKey 随输入 dtype 与 sdepth 的取值变化
 *   - 可选输出 sqsum/tilted 的使能判定（形态 1/2/3）
 *   - colTileWidth + 8 <= rowTileWidth 不变式（Phase B 复用 UB 缓冲不得越界）
 *   - 高通道 3D HWC（C>=9）的 legacy UB 预算：blockWidth 必须收缩到真实 UB 占用以内
 *   - 非法 sdepth、非法 rank、块数超过平台核数的负向用例
 */

#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "../../../op_host/integral_image_compile_info.h"
#include "../../../op_kernel/integral_image_tiling_data.h"
#include "../../../op_kernel/integral_image_tiling_key.h"
#include "platform/platform_infos_def.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class IntegralImageTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "IntegralImageTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "IntegralImageTiling TearDown" << std::endl; }
};

namespace {

constexpr int64_t kReservedWorkspace = 16LL * 1024 * 1024;
constexpr int64_t kAccBytes = 4;

gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

gert::TilingContextPara::TensorDescription MakeTensor(const std::vector<int64_t>& dims, ge::DataType dtype)
{
    return gert::TilingContextPara::TensorDescription(MakeShape(dims), dtype, ge::FORMAT_ND);
}

// 按 HWC 语义构造输出 Shape：(H,W) -> (H+1,W+1)，(H,W,C) -> (H+1,W+1,C)
std::vector<int64_t> MakeOutDims(const std::vector<int64_t>& inDims)
{
    std::vector<int64_t> outDims(inDims);
    if (outDims.size() >= 2) {
        outDims[0] += 1;
        outDims[1] += 1;
    }
    return outDims;
}

bool RunTilingWithOutputs(const std::vector<int64_t>& inDims, ge::DataType inDtype, int64_t sdepth,
                          const std::vector<gert::TilingContextPara::TensorDescription>& outputs, TilingInfo& info,
                          uint64_t coreNum = 64, uint64_t ubSize = 262144)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeTensor(inDims, inDtype)};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        gert::TilingContextPara::OpAttr("sdepth", Ops::Cv::AnyValue::CreateFrom<int64_t>(sdepth))};
    optiling::IntegralImageCompileInfo compileInfo;
    gert::TilingContextPara para("IntegralImage", inputs, outputs, attrs, &compileInfo, "Ascend910B", coreNum, ubSize,
                                 4096);
    return ExecuteTiling(para, info);
}

// outNum：声明的输出个数，1=sat，2=+sqsum，3=+tilted
bool RunTiling(const std::vector<int64_t>& inDims, ge::DataType inDtype, ge::DataType satDtype, int64_t sdepth,
               size_t outNum, TilingInfo& info, uint64_t coreNum = 64, uint64_t ubSize = 262144)
{
    const std::vector<int64_t> outDims = MakeOutDims(inDims);
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {MakeTensor(outDims, satDtype)};
    if (outNum >= 2) {
        outputs.push_back(MakeTensor(outDims, ge::DT_FLOAT));
    }
    if (outNum >= 3) {
        outputs.push_back(MakeTensor(outDims, satDtype));
    }
    return RunTilingWithOutputs(inDims, inDtype, sdepth, outputs, info, coreNum, ubSize);
}

const IntegralImageTilingData* GetTilingData(const TilingInfo& info)
{
    return reinterpret_cast<const IntegralImageTilingData*>(info.tilingData.get());
}

// two-phase 公共检查：Phase B 的列 tile 不得越过 Phase A 缓冲（含 8 元素零填充槽）
void ExpectTwoPhaseValid(const TilingInfo& info)
{
    const IntegralImageTilingData* tilingData = GetTilingData(info);
    EXPECT_EQ(tilingData->twoPhase, 1);
    EXPECT_GE(tilingData->colTileWidth, 32);
    EXPECT_EQ(tilingData->colTileWidth % 8, 0);
    EXPECT_LE(tilingData->colTileWidth + 8, tilingData->rowTileWidth);
    EXPECT_EQ(tilingData->colTileCount, (tilingData->width + tilingData->colTileWidth - 1) / tilingData->colTileWidth);
    EXPECT_EQ(info.blockNum, static_cast<size_t>(tilingData->activeCoreNum));
    EXPECT_GE(info.blockNum, 1U);
}

} // namespace

// 2D UINT8 宽图：two-phase 分支，tilingKey = U8_I32
TEST_F(IntegralImageTiling, integral_image_tiling_2d_u8_two_phase)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling({8, 2048}, ge::DT_UINT8, ge::DT_INT32, -1, 3, info));
    const IntegralImageTilingData* tilingData = GetTilingData(info);
    EXPECT_EQ(tilingData->height, 8);
    EXPECT_EQ(tilingData->width, 2048);
    EXPECT_EQ(tilingData->channel, 1);
    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(INTEGRAL_IMAGE_TPL_SCH_MODE_U8_I32));
    ExpectTwoPhaseValid(info);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(info.workspaceSizes[0], kReservedWorkspace + 8LL * 2048 * kAccBytes * 4);
}

// 2D FLOAT32 宽图：two-phase 分支，tilingKey = F32_F32
TEST_F(IntegralImageTiling, integral_image_tiling_2d_f32_two_phase)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling({16, 4096}, ge::DT_FLOAT, ge::DT_FLOAT, -1, 1, info));
    const IntegralImageTilingData* tilingData = GetTilingData(info);
    EXPECT_EQ(tilingData->channel, 1);
    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(INTEGRAL_IMAGE_TPL_SCH_MODE_F32_F32));
    ExpectTwoPhaseValid(info);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(info.workspaceSizes[0], kReservedWorkspace + 16LL * 4096 * kAccBytes * 4);
}

// 2D FLOAT16：tilingKey = F16_F32
TEST_F(IntegralImageTiling, integral_image_tiling_2d_f16)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling({8, 256}, ge::DT_FLOAT16, ge::DT_FLOAT, -1, 3, info));
    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(INTEGRAL_IMAGE_TPL_SCH_MODE_F16_F32));
    ExpectTwoPhaseValid(info);
}

// UINT8 + sdepth=1：tilingKey = U8_F32
TEST_F(IntegralImageTiling, integral_image_tiling_2d_u8_sdepth_f32)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling({8, 256}, ge::DT_UINT8, ge::DT_FLOAT, 1, 3, info));
    EXPECT_EQ(info.tilingKey, GET_TPL_TILING_KEY(INTEGRAL_IMAGE_TPL_SCH_MODE_U8_F32));
    ExpectTwoPhaseValid(info);
}

// 2D 小图（W < 128）：legacy 分支，单列块
TEST_F(IntegralImageTiling, integral_image_tiling_2d_small_w_legacy)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling({8, 64}, ge::DT_UINT8, ge::DT_INT32, -1, 3, info));
    const IntegralImageTilingData* tilingData = GetTilingData(info);
    EXPECT_EQ(tilingData->twoPhase, 0);
    EXPECT_EQ(tilingData->height, 8);
    EXPECT_EQ(tilingData->width, 64);
    EXPECT_GE(tilingData->blockWidth, 32);
    EXPECT_EQ(info.blockNum, 1U);
    ASSERT_EQ(info.workspaceSizes.size(), 1U);
    EXPECT_EQ(info.workspaceSizes[0], kReservedWorkspace + 8LL * 64 * kAccBytes * 3);
}

// 3D HWC：legacy 分支，通道维保持不变
TEST_F(IntegralImageTiling, integral_image_tiling_3d_hwc_legacy)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling({4, 6, 3}, ge::DT_UINT8, ge::DT_INT32, -1, 3, info));
    const IntegralImageTilingData* tilingData = GetTilingData(info);
    EXPECT_EQ(tilingData->twoPhase, 0);
    EXPECT_EQ(tilingData->height, 4);
    EXPECT_EQ(tilingData->width, 6);
    EXPECT_EQ(tilingData->channel, 3);
    EXPECT_EQ(info.blockNum, 1U);
}

// 形态 1：只给 sat，sqsum/tilted 均不计算
TEST_F(IntegralImageTiling, integral_image_tiling_form1_sat_only)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling({8, 2048}, ge::DT_UINT8, ge::DT_INT32, -1, 1, info));
    const IntegralImageTilingData* tilingData = GetTilingData(info);
    EXPECT_EQ(tilingData->sqsumEnabled, 0);
    EXPECT_EQ(tilingData->tiltedEnabled, 0);
}

// 形态 2：给 sat + sqsum，tilted 不计算
TEST_F(IntegralImageTiling, integral_image_tiling_form2_sat_sqsum)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling({8, 2048}, ge::DT_UINT8, ge::DT_INT32, -1, 2, info));
    const IntegralImageTilingData* tilingData = GetTilingData(info);
    EXPECT_EQ(tilingData->sqsumEnabled, 1);
    EXPECT_EQ(tilingData->tiltedEnabled, 0);
}

// 形态 3：三路全部计算
TEST_F(IntegralImageTiling, integral_image_tiling_form3_all)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling({8, 2048}, ge::DT_UINT8, ge::DT_INT32, -1, 3, info));
    const IntegralImageTilingData* tilingData = GetTilingData(info);
    EXPECT_EQ(tilingData->sqsumEnabled, 1);
    EXPECT_EQ(tilingData->tiltedEnabled, 1);
}

// 可选输出传空 Tensor（rank0）：同样按未使能处理
TEST_F(IntegralImageTiling, integral_image_tiling_optional_empty_tensor)
{
    std::vector<gert::TilingContextPara::TensorDescription> inputs = {MakeTensor({8, 2048}, ge::DT_UINT8)};
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        MakeTensor({9, 2049}, ge::DT_INT32),
        gert::TilingContextPara::TensorDescription(gert::StorageShape(), ge::DT_FLOAT, ge::FORMAT_ND),
        gert::TilingContextPara::TensorDescription(gert::StorageShape(), ge::DT_INT32, ge::FORMAT_ND)};
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        gert::TilingContextPara::OpAttr("sdepth", Ops::Cv::AnyValue::CreateFrom<int64_t>(-1))};
    optiling::IntegralImageCompileInfo compileInfo;
    gert::TilingContextPara para("IntegralImage", inputs, outputs, attrs, &compileInfo, "Ascend910B");
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    const IntegralImageTilingData* tilingData = GetTilingData(info);
    EXPECT_EQ(tilingData->sqsumEnabled, 0);
    EXPECT_EQ(tilingData->tiltedEnabled, 0);
}

// sdepth=2 不属于合法属性取值，应拒绝
TEST_F(IntegralImageTiling, integral_image_tiling_sdepth2_failed)
{
    TilingInfo info;
    EXPECT_FALSE(RunTiling({8, 256}, ge::DT_UINT8, ge::DT_INT32, 2, 3, info));
}

// sdepth=0 仅对 UINT8 有效，FLOAT32 输入应拒绝
TEST_F(IntegralImageTiling, integral_image_tiling_sdepth0_f32_failed)
{
    TilingInfo info;
    EXPECT_FALSE(RunTiling({8, 256}, ge::DT_FLOAT, ge::DT_INT32, 0, 3, info));
}

// rank=1 非法
TEST_F(IntegralImageTiling, integral_image_tiling_rank1_failed)
{
    TilingInfo info;
    EXPECT_FALSE(RunTiling({64}, ge::DT_UINT8, ge::DT_INT32, -1, 1, info));
}

// 3D 宽图：所需列块数超过平台核数，应拒绝。
// W=262144 对应的分块数超过本用例配置的 64 个 AIV 核。
TEST_F(IntegralImageTiling, integral_image_tiling_wide_3d_failed)
{
    TilingInfo info;
    EXPECT_FALSE(RunTiling({1, 262144, 2}, ge::DT_UINT8, ge::DT_INT32, -1, 1, info));
}

// Shape 含 0 维非法
TEST_F(IntegralImageTiling, integral_image_tiling_zero_dim_failed)
{
    TilingInfo info;
    EXPECT_FALSE(RunTiling({0, 64}, ge::DT_UINT8, ge::DT_INT32, -1, 1, info));
}

// 3D HWC C=9：UB 预算包含 dstSqBatch 的批处理与 HWC 交错容量，选择可容纳的块宽。
TEST_F(IntegralImageTiling, integral_image_tiling_3d_c9_legacy_ub_fit)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling({8, 512, 9}, ge::DT_UINT8, ge::DT_INT32, -1, 3, info));
    const IntegralImageTilingData* tilingData = GetTilingData(info);
    EXPECT_EQ(tilingData->twoPhase, 0);
    EXPECT_EQ(tilingData->channel, 9);
    EXPECT_GE(tilingData->blockWidth, 32);
    EXPECT_EQ(tilingData->blockWidth % 32, 0);
    // 此配置由 HWC 交错容量决定 dstSqBatch 的分配大小。
    EXPECT_GT(tilingData->blockWidth * tilingData->channel, 8 * (tilingData->blockWidth + 8));
    EXPECT_EQ(info.blockNum, 1U);
}

// 3D HWC C=33：UB 占用随 C 近似线性增长（sat/accBig/accSq 各 blockWidth*C 个 4B 元素），
// 根据通道数与平台 UB 容量选择相应的 blockWidth。
TEST_F(IntegralImageTiling, integral_image_tiling_3d_c33_legacy_ub_fit)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling({4, 256, 33}, ge::DT_UINT8, ge::DT_INT32, -1, 3, info));
    const IntegralImageTilingData* tilingData = GetTilingData(info);
    EXPECT_EQ(tilingData->twoPhase, 0);
    EXPECT_EQ(tilingData->channel, 33);
    EXPECT_GE(tilingData->blockWidth, 32);
    EXPECT_EQ(tilingData->blockWidth % 32, 0);
    EXPECT_LE(tilingData->blockWidth, 256);
}

// A valid 192 KiB configuration where the HWC zero boundary is wider than the block.
TEST_F(IntegralImageTiling, integral_image_tiling_3d_c100_sqsum_boundary)
{
    TilingInfo info;
    ASSERT_TRUE(RunTiling({2, 1, 100}, ge::DT_UINT8, ge::DT_INT32, -1, 2, info, 40, 192 * 1024));
    const IntegralImageTilingData* tilingData = GetTilingData(info);
    EXPECT_EQ(tilingData->twoPhase, 0);
    EXPECT_EQ(tilingData->blockWidth, 64);
    EXPECT_GT(tilingData->channel, tilingData->blockWidth + 8);
    EXPECT_EQ(tilingData->sqsumEnabled, 1);
    EXPECT_EQ(tilingData->tiltedEnabled, 0);
}

// At the minimum block width, C=100 needs 63152 bytes plus the 8192-byte operator reserve.
// Account for the SDK's own UB reservation before testing that usable-memory boundary.
TEST_F(IntegralImageTiling, integral_image_tiling_3d_c100_zero_boundary_ub_limit)
{
    constexpr uint64_t probeUbBytes = 192 * 1024;
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    std::map<std::string, std::string> aicoreSpec = {{"ub_size", std::to_string(probeUbBytes)}};
    std::map<std::string, std::string> version = {{"Short_SoC_version", "Ascend910B"}, {"NpuArch", "2201"}};
    platformInfo.SetPlatformRes("AICoreSpec", aicoreSpec);
    platformInfo.SetPlatformRes("version", version);
    platform_ascendc::PlatformAscendC platform(&platformInfo);
    uint64_t usableUbBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, usableUbBytes);
    ASSERT_GT(usableUbBytes, 0U);
    ASSERT_LE(usableUbBytes, probeUbBytes);
    const uint64_t platformReserveBytes = probeUbBytes - usableUbBytes;
    constexpr uint64_t requiredUsableUbBytes = 63152 + 8192;
    const uint64_t rawUbBytes = requiredUsableUbBytes + platformReserveBytes;
    SCOPED_TRACE("platform UB reservation: " + std::to_string(platformReserveBytes));

    TilingInfo info;
    ASSERT_TRUE(RunTiling({2, 1, 100}, ge::DT_UINT8, ge::DT_INT32, -1, 2, info, 40, rawUbBytes));
    EXPECT_EQ(GetTilingData(info)->blockWidth, 32);
    TilingInfo insufficient;
    EXPECT_FALSE(RunTiling({2, 1, 100}, ge::DT_UINT8, ge::DT_INT32, -1, 2, insufficient, 40, rawUbBytes - 32));
}

TEST_F(IntegralImageTiling, integral_image_tiling_sat_shape_check)
{
    for (const auto& dims : std::vector<std::vector<int64_t>>{{1, 1}, {2, 1}, {1, 2}, {4}, {2, 2, 1}}) {
        TilingInfo info;
        EXPECT_FALSE(RunTilingWithOutputs({1, 1}, ge::DT_UINT8, -1, {MakeTensor(dims, ge::DT_INT32)}, info));
    }
}

TEST_F(IntegralImageTiling, integral_image_tiling_optional_shape_check)
{
    for (size_t index : {1U, 2U}) {
        for (const auto& dims : std::vector<std::vector<int64_t>>{{1, 1}, {2, 1}, {1, 2}, {4}, {2, 2, 1}}) {
            SCOPED_TRACE("output index: " + std::to_string(index));
            std::vector<gert::TilingContextPara::TensorDescription> outputs = {
                MakeTensor({2, 2}, ge::DT_INT32), MakeTensor({2, 2}, ge::DT_FLOAT), MakeTensor({2, 2}, ge::DT_INT32)};
            outputs[index] = MakeTensor(dims, index == 1 ? ge::DT_FLOAT : ge::DT_INT32);
            TilingInfo info;
            EXPECT_FALSE(RunTilingWithOutputs({1, 1}, ge::DT_UINT8, -1, outputs, info));
        }
    }
}

TEST_F(IntegralImageTiling, integral_image_tiling_hwc_output_shape_check)
{
    for (size_t index : {0U, 1U, 2U}) {
        for (const auto& dims : std::vector<std::vector<int64_t>>{{3, 4, 2}, {3, 4, 4}, {3, 4}, {3, 4, 1, 3}}) {
            SCOPED_TRACE("output index: " + std::to_string(index));
            std::vector<gert::TilingContextPara::TensorDescription> outputs = {MakeTensor({3, 4, 3}, ge::DT_INT32),
                                                                               MakeTensor({3, 4, 3}, ge::DT_FLOAT),
                                                                               MakeTensor({3, 4, 3}, ge::DT_INT32)};
            outputs[index] = MakeTensor(dims, index == 1 ? ge::DT_FLOAT : ge::DT_INT32);
            TilingInfo info;
            EXPECT_FALSE(RunTilingWithOutputs({2, 3, 3}, ge::DT_UINT8, -1, outputs, info));
        }
    }
}

TEST_F(IntegralImageTiling, integral_image_tiling_output_storage_shape_check)
{
    gert::StorageShape shape = MakeShape({2, 2});
    shape.MutableStorageShape().SetDim(0, 1);
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        gert::TilingContextPara::TensorDescription(shape, ge::DT_INT32, ge::FORMAT_ND)};
    TilingInfo info;
    EXPECT_FALSE(RunTilingWithOutputs({1, 1}, ge::DT_UINT8, -1, outputs, info));
}

TEST_F(IntegralImageTiling, integral_image_tiling_sat_dtype_check)
{
    for (int64_t depth : {-1, 0, 1}) {
        SCOPED_TRACE("sdepth: " + std::to_string(depth));
        TilingInfo info;
        EXPECT_FALSE(RunTiling({1, 1}, ge::DT_UINT8, depth == 1 ? ge::DT_INT32 : ge::DT_FLOAT, depth, 3, info));
    }
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        TilingInfo info;
        EXPECT_FALSE(RunTiling({1, 1}, dtype, ge::DT_INT32, -1, 3, info));
    }
}

TEST_F(IntegralImageTiling, integral_image_tiling_optional_dtype_check)
{
    for (int64_t depth : {-1, 1}) {
        const auto accType = depth == 1 ? ge::DT_FLOAT : ge::DT_INT32;
        for (size_t index : {1U, 2U}) {
            std::vector<gert::TilingContextPara::TensorDescription> outputs = {
                MakeTensor({2, 2}, accType), MakeTensor({2, 2}, ge::DT_FLOAT), MakeTensor({2, 2}, accType)};
            const auto invalidType = index == 1 || depth == 1 ? ge::DT_INT32 : ge::DT_FLOAT;
            outputs[index] = MakeTensor({2, 2}, invalidType);
            TilingInfo info;
            EXPECT_FALSE(RunTilingWithOutputs({1, 1}, ge::DT_UINT8, depth, outputs, info));
        }
    }
}

TEST_F(IntegralImageTiling, integral_image_tiling_f32_optional_outputs)
{
    for (int64_t width : {64, 131}) {
        for (size_t outNum : {1U, 2U, 3U}) {
            TilingInfo info;
            ASSERT_TRUE(RunTiling({8, width}, ge::DT_UINT8, ge::DT_FLOAT, 1, outNum, info));
            EXPECT_EQ(GetTilingData(info)->sqsumEnabled, outNum >= 2 ? 1 : 0);
            EXPECT_EQ(GetTilingData(info)->tiltedEnabled, outNum >= 3 ? 1 : 0);
        }
    }
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        MakeTensor({9, 65}, ge::DT_FLOAT), MakeTensor({0}, ge::DT_FLOAT), MakeTensor({0}, ge::DT_FLOAT)};
    TilingInfo info;
    ASSERT_TRUE(RunTilingWithOutputs({8, 64}, ge::DT_UINT8, 1, outputs, info));
    EXPECT_EQ(GetTilingData(info)->sqsumEnabled, 0);
    EXPECT_EQ(GetTilingData(info)->tiltedEnabled, 0);
}

TEST_F(IntegralImageTiling, integral_image_tiling_spatial_dimension_limit)
{
    for (const auto& dims : std::vector<std::vector<int64_t>>{{INT64_MAX, 1}, {1, INT64_MAX}}) {
        TilingInfo info;
        EXPECT_FALSE(RunTilingWithOutputs(dims, ge::DT_UINT8, -1, {MakeTensor({2, 2}, ge::DT_INT32)}, info));
    }
}
