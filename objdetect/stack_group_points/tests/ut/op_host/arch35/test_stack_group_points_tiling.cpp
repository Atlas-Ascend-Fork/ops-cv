/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include "../../../../op_host/arch35/stack_group_points_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class StackGroupPointsTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "StackGroupPointsTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "StackGroupPointsTiling TearDown" << std::endl; }
};

static constexpr uint64_t SOC_CORE_NUM = 64;
static constexpr uint64_t SOC_UB_SIZE = 262144;
static constexpr uint64_t TILING_DATA_SIZE = 4096;

// tiling 数据依赖输入（TilingInputsDataDependency）：batch 计数张量值需喂给 faker
// （isConst + constValue，仓内 crop_and_resize UT 先例）。取值与 shape 语义一致：
// N=4096 行特征 / M=16 行索引，各拆 1 批。
static const int32_t fbcDataOne[] = {4096};
static const int32_t ibcDataOne[] = {16};

TEST_F(StackGroupPointsTiling, stack_group_points_tiling_fp16)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara(
        "StackGroupPoints",
        {
            {featuresShape, ge::DT_FLOAT16, ge::FORMAT_ND},
            {fbcShape, ge::DT_INT32, ge::FORMAT_ND, true, const_cast<void*>(static_cast<const void*>(fbcDataOne))},
            {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
            {ibcShape, ge::DT_INT32, ge::FORMAT_ND, true, const_cast<void*>(static_cast<const void*>(ibcDataOne))},
        },
        {
            {yShape, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE, TILING_DATA_SIZE);
    uint64_t expectTilingKey = 1;
    std::string expectTilingData = "16 32 128 1 4096 65536 64 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(StackGroupPointsTiling, stack_group_points_tiling_fp32)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara(
        "StackGroupPoints",
        {
            {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {fbcShape, ge::DT_INT32, ge::FORMAT_ND, true, const_cast<void*>(static_cast<const void*>(fbcDataOne))},
            {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
            {ibcShape, ge::DT_INT32, ge::FORMAT_ND, true, const_cast<void*>(static_cast<const void*>(ibcDataOne))},
        },
        {
            {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE, TILING_DATA_SIZE);
    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "16 32 128 1 4096 65536 64 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// 可表达边界：m*c*nsample=8e18 接近 INT64_MAX 仍应成功（issue #1086 回归）
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_total_elements_int64_boundary)
{
    gert::StorageShape featuresShape = {{1, 2000000000}, {1, 2000000000}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{2000000000, 2}, {2000000000, 2}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{2000000000, 2000000000, 2}, {2000000000, 2000000000, 2}};

    static const int32_t fbcDataBoundary[] = {1};
    static const int32_t ibcDataBoundary[] = {2000000000};
    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara(
        "StackGroupPoints",
        {
            {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {fbcShape, ge::DT_INT32, ge::FORMAT_ND, true, const_cast<void*>(static_cast<const void*>(fbcDataBoundary))},
            {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
            {ibcShape, ge::DT_INT32, ge::FORMAT_ND, true, const_cast<void*>(static_cast<const void*>(ibcDataBoundary))},
        },
        {
            {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE, TILING_DATA_SIZE);
    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "2000000000 2000000000 2 1 1 8000000000000000000 64 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// ===== 边界校验失败用例 =====

// Negative: features C=0, kernel 中 index / nsample % c 会除零
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_features_c_zero)
{
    gert::StorageShape featuresShape = {{4096, 0}, {4096, 0}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 0, 128}, {16, 0, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: indices nsample=0, kernel 中 index % nsample 会除零
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_nsample_zero)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 0}, {16, 0}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 0}, {16, 32, 0}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: indices_batch_cnt B=0
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_batch_cnt_b_zero)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{0}, {0}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{0}, {0}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: features_batch_cnt dim0 != indices_batch_cnt dim0
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_fbc_b_mismatch)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{2}, {2}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{3}, {3}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: features dtype 不支持（DT_DOUBLE）
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_features_dtype_double)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_DOUBLE, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_DOUBLE, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: features 非 2D（3D）
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_features_not_2d)
{
    gert::StorageShape featuresShape = {{4096, 32, 8}, {4096, 32, 8}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: m*c*nsample=1.2e19 超过 INT64_MAX，乘法有符号溢出，tiling 应失败（issue #1086）
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_total_elements_overflow)
{
    gert::StorageShape featuresShape = {{1, 2000000000}, {1, 2000000000}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{2000000000, 3}, {2000000000, 3}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{2000000000, 2000000000, 3}, {2000000000, 2000000000, 3}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// ===== 输出描述符一致性校验用例 =====

// Negative: 输出 y dtype 域外（bfloat16），应拒绝
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_output_dtype_out_of_domain)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: 输出 y 与 features 跨位失配（y=float16 / features=float32，kernel 按
// features dtype 写元素），应拒绝
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_output_dtype_cross)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: 输出 y rank 失配（声明 2D，推导为 3D），应拒绝
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_output_rank_2d)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32}, {16, 32}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: 输出 y rank 失配（声明 4D，元素数相同），应拒绝
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_output_rank_4d)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128, 1}, {16, 32, 128, 1}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: 输出声明元素数下溢（声明 16*32*64=32768 < 写入 16*32*128=65536，
// 设备越界写），应拒绝
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_output_elem_underflow)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 64}, {16, 32, 64}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 正例：合法空首轴欠写形态——indices m=0（kernel 写 0 元素）而声明输出 (16,32,128)
// （声明元素数 >= 写入量，内存安全），校验放行
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_pos_output_underwrite)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{0, 128}, {0, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    static const int32_t fbcDataM0[] = {4096};
    static const int32_t ibcDataM0[] = {0};
    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara(
        "StackGroupPoints",
        {
            {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {fbcShape, ge::DT_INT32, ge::FORMAT_ND, true, const_cast<void*>(static_cast<const void*>(fbcDataM0))},
            {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
            {ibcShape, ge::DT_INT32, ge::FORMAT_ND, true, const_cast<void*>(static_cast<const void*>(ibcDataM0))},
        },
        {
            {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE, TILING_DATA_SIZE);
    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "0 32 128 1 4096 0 1 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

// Negative: features_batch_cnt dtype 域外（bfloat16，仅支持 INT32），应拒绝
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_fbc_dtype_bfloat16)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_BF16, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: indices dtype 域外（bool，仅支持 INT32），应拒绝
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_indices_dtype_bool)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_BOOL, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: indices_batch_cnt dtype 域外（bfloat16，仅支持 INT32），应拒绝
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_ibc_dtype_bfloat16)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_BF16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: indices rank 失配（声明 1D，要求 2D），应拒绝
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_indices_not_2d)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{128}, {128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: features_batch_cnt rank 失配（声明 2D，要求 1D），应拒绝
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_fbc_not_1d)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1, 1}, {1, 1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: features_batch_cnt rank 失配（声明 0D 标量，要求 1D），应拒绝
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_fbc_0d)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{}, {}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: indices_batch_cnt rank 失配（声明 2D，要求 1D），应拒绝
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_ibc_not_1d)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1, 1}, {1, 1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: indices_batch_cnt rank 失配（声明 0D 标量，要求 1D），应拒绝
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_ibc_0d)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{}, {}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: 必选输入缺失（节点仅挂 3 个输入，第 4 输入 indices_batch_cnt 无 desc），
// desc 获取为空，应拒绝
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_missing_input)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: 输出声明元素数计算中溢出与零维组合——y 前两维乘积溢出 int64 且末维为 0，
// 真实声明元素数为 0（0 与任何维值相乘必为 0），低于写入量 m*c*nsample，应拒绝
// （不可将溢出视为容量充足：零维使元素数确定为 0）
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_output_elem_zero_dim_overflow)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{INT64_MAX, 2, 0}, {INT64_MAX, 2, 0}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// Negative: features 元素总数 n*c 溢出 int64（n=INT64_MAX, c=2），读侧越界判据
// 会回绕，应拒绝
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_neg_featlen_overflow)
{
    gert::StorageShape featuresShape = {{INT64_MAX, 2}, {INT64_MAX, 2}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{1, 128}, {1, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{1, 2, 128}, {1, 2, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED);
}

// 正例：batch 计数张量 Host 数据不可得（编译侧 tiling 无值 staging，计数输入未声明
// const 值）时，值域校验跳过、tiling 正常成功（运行时 tiling 由数据依赖声明保证
// 值可见）
TEST_F(StackGroupPointsTiling, stack_group_points_tiling_pos_skip_value_validation)
{
    gert::StorageShape featuresShape = {{4096, 32}, {4096, 32}};
    gert::StorageShape fbcShape = {{1}, {1}};
    gert::StorageShape indicesShape = {{16, 128}, {16, 128}};
    gert::StorageShape ibcShape = {{1}, {1}};
    gert::StorageShape yShape = {{16, 32, 128}, {16, 32, 128}};

    StackGroupPointsCompileInfo compileInfo = {0, 0};
    gert::TilingContextPara tilingContextPara("StackGroupPoints",
                                              {
                                                  {featuresShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {fbcShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {indicesShape, ge::DT_INT32, ge::FORMAT_ND},
                                                  {ibcShape, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {yShape, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", SOC_CORE_NUM, SOC_UB_SIZE,
                                              TILING_DATA_SIZE);
    uint64_t expectTilingKey = 0;
    std::string expectTilingData = "16 32 128 1 4096 65536 64 ";
    std::vector<size_t> expectWorkspaces = {16777216};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
