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
 * \file test_lbp_feature_tiling.cpp
 * \brief LbpFeature Tiling UT：校验 tilingKey / workspace / tilingData（冻结基线）。
 *        平台常量按 910B4：40 核 / UB 192KB。
 *        注：UT 假平台未填充 lib API workspace 字段，GetLibApiWorkSpaceSize() 返回
 *        uint32 最大值（4294967295），此值仅用于对齐 UT 环境基线。
 */

#include <gtest/gtest.h>
#include <iostream>
#include <cstring>
#include <string>
#include <vector>

#include "../../../op_kernel/lbp_feature_tiling_data.h"
#include "../../../op_kernel/lbp_feature_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace Ops::Cv;

static const std::vector<size_t> UT_FAKE_WORKSPACE = {4294967295U};

class LbpFeatureTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LbpFeatureTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "LbpFeatureTiling TearDown" << std::endl; }

    static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
    {
        gert::StorageShape ss;
        ss.MutableOriginShape().SetDimNum(dims.size());
        ss.MutableStorageShape().SetDimNum(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            ss.MutableOriginShape().SetDim(i, dims[i]);
            ss.MutableStorageShape().SetDim(i, dims[i]);
        }
        return ss;
    }

    static void RunOkCase(const std::vector<int64_t>& xShape, const std::vector<int64_t>& yShape, ge::DataType dtype,
                          int64_t radius, int64_t neighbors, uint64_t expectTilingKey,
                          const std::string& expectTilingData)
    {
        struct LbpFeatureCompileInfo {
        } compileInfo;
        gert::TilingContextPara::TensorDescription xDesc(MakeStorageShape(xShape), dtype, ge::FORMAT_ND);
        gert::TilingContextPara::TensorDescription yDesc(MakeStorageShape(yShape), ge::DT_INT32, ge::FORMAT_ND);
        gert::TilingContextPara para(
            "LbpFeature", {xDesc}, {yDesc},
            {gert::TilingContextPara::OpAttr("radius", AnyValue::CreateFrom<int64_t>(radius)),
             gert::TilingContextPara::OpAttr("neighbors", AnyValue::CreateFrom<int64_t>(neighbors))},
            &compileInfo, "", 40, 196608, 4096);

        ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, UT_FAKE_WORKSPACE);
    }

    static void RunFailCase(const std::vector<int64_t>& xShape, const std::vector<int64_t>& yShape, ge::DataType dtype,
                            int64_t radius, int64_t neighbors)
    {
        struct LbpFeatureCompileInfo {
        } compileInfo;
        gert::TilingContextPara::TensorDescription xDesc(MakeStorageShape(xShape), dtype, ge::FORMAT_ND);
        gert::TilingContextPara::TensorDescription yDesc(MakeStorageShape(yShape), ge::DT_INT32, ge::FORMAT_ND);
        gert::TilingContextPara para(
            "LbpFeature", {xDesc}, {yDesc},
            {gert::TilingContextPara::OpAttr("radius", AnyValue::CreateFrom<int64_t>(radius)),
             gert::TilingContextPara::OpAttr("neighbors", AnyValue::CreateFrom<int64_t>(neighbors))},
            &compileInfo, "", 40, 196608, 4096);
        ExecuteTestCase(para, ge::GRAPH_FAILED, 0, "", {0});
    }
};

// ── 用例1：{2,3,63,65} 四维 batch RowSplit ──
TEST_F(LbpFeatureTiling, lbp_feature_tiling_test1)
{
    RunOkCase(
        {2, 3, 63, 65}, {2, 3, 61, 63}, ge::DT_UINT8, 1, 8, GET_TPL_TILING_KEY(LBP_FEATURE_SCH_MODE_ROW_SPLIT),
        "34359738369 99033355935738 270582939709 279172874303 1571958030342 4294967297 34359738368 25769803792 "
        "270582939664 4294967360 4294967300 -274877907009 -279172874306 279172874239 0 0 0 0 0 0 0 0 0 0 0 0 "
        "-274877906943 -274877907008 4294967230 4294967361 0 0 0 0 0 0 0 0 0 0 0 4294967296 -279172874240 -1 "
        "274877907008 65 0 0 0 0 0 0 0 0 0 0 0 4294967297 4294967232 283467841535 283467841601 0 0 0 0 0 0 0 0 0 0 0 "
        "4575657221408423936 4575657222454121421 1056964607 1045697485 1034924238 0 0 0 0 0 0 0 0 0 0 0 "
        "4539628420094492672 4491236500197683506 4444965756047720448 4491236500649803776 0 0 0 0 0 0 0 0 0 0 0 "
        "-9223372036854775808 1034924238 4575657222454121421 1056964607 1045697485 0 0 0 0 0 0 0 0 0 0 0 "
        "4491236501731934208 4444965756047720448 4491236499584450560 4539628420094492672 0 0 0 0 0 0 0 0 0 0 0 "
        "4294967296 8589934595 8589934591 -4294967296 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4294967296 8589934593 8589934594 "
        "4294967298 0 0 0 0 0 0 0 0 0 0 0 0 0 8589934593 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 1 0 4294967296 4294967298 "
        "0 0 0 0 0 0 0 0 0 0 0 4294967296 1 4294967297 8589934594 2 0 0 0 0 0 0 0 0 0 0 0 ");
}

// ── 用例2：{6,70} 二维 RowSplit ──
TEST_F(LbpFeatureTiling, lbp_feature_tiling_test_2d)
{
    RunOkCase(
        {6, 70}, {4, 68}, ge::DT_UINT8, 1, 8, GET_TPL_TILING_KEY(LBP_FEATURE_SCH_MODE_ROW_SPLIT),
        "34359738369 1168231104932 292057776132 300647710726 17179869185 4294967297 8589934592 2 292057776130 "
        "4294967392 4294967298 -296352743494 -300647710791 300647710719 0 0 0 0 0 0 0 0 0 0 0 0 -296352743423 "
        "-296352743493 4294967225 4294967366 0 0 0 0 0 0 0 0 0 0 0 4294967296 -300647710720 -1 296352743493 70 0 0 0 0 "
        "0 0 0 0 0 0 0 4294967297 4294967227 304942678015 304942678086 0 0 0 0 0 0 0 0 0 0 0 4575657221408423936 "
        "4575657222454121421 1056964607 1045697485 1034924238 0 0 0 0 0 0 0 0 0 0 0 4539628420094492672 "
        "4491236500197683506 4444965756047720448 4491236500649803776 0 0 0 0 0 0 0 0 0 0 0 -9223372036854775808 "
        "1034924238 4575657222454121421 1056964607 1045697485 0 0 0 0 0 0 0 0 0 0 0 4491236501731934208 "
        "4444965756047720448 4491236499584450560 4539628420094492672 0 0 0 0 0 0 0 0 0 0 0 4294967296 8589934595 "
        "8589934591 -4294967296 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4294967296 8589934593 8589934594 4294967298 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 8589934593 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 1 0 4294967296 4294967298 0 0 0 0 0 0 0 0 0 0 0 "
        "4294967296 1 4294967297 8589934594 2 0 0 0 0 0 0 0 0 0 0 0 ");
}

// ── 用例3：{8,130} 二维列并行（outW=128 触发 Tile2D）──
TEST_F(LbpFeatureTiling, lbp_feature_tiling_test_2d_colparallel)
{
    RunOkCase({8, 130}, {6, 128}, ge::DT_UINT8, 1, 8, GET_TPL_TILING_KEY(LBP_FEATURE_SCH_MODE_TILE_2D),
              "34359738369 3298534884368 549755813894 558345748488 25769803777 4294967297 4294967296 1 274877906945 "
              "8589934656 4294967308 -554050781314 -558345748611 558345748479 0 0 0 0 0 0 0 0 0 0 0 0 -554050781183 "
              "-554050781313 4294967165 4294967426 0 0 0 0 0 0 0 0 0 0 0 4294967296 -558345748480 -1 554050781313 130 "
              "0 0 0 0 0 0 0 0 0 0 0 4294967297 4294967167 562640715775 562640715906 0 0 0 0 0 0 0 0 0 0 0 "
              "4575657221408423936 4575657222454121421 1056964607 1045697485 1034924238 0 0 0 0 0 0 0 0 0 0 0 "
              "4539628420094492672 4491236500197683506 4444965756047720448 4491236500649803776 0 0 0 0 0 0 0 0 0 0 0 "
              "-9223372036854775808 1034924238 4575657222454121421 1056964607 1045697485 0 0 0 0 0 0 0 0 0 0 0 "
              "4491236501731934208 4444965756047720448 4491236499584450560 4539628420094492672 0 0 0 0 0 0 0 0 0 0 0 0 "
              "8589934595 8589934591 -4294967296 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4294967296 8589934593 8589934594 "
              "4294967298 0 0 0 0 0 0 0 0 0 0 0 0 0 8589934593 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 1 0 4294967296 "
              "4294967298 0 0 0 0 0 0 0 0 0 0 0 4294967296 1 4294967297 8589934594 2 0 0 0 0 0 0 0 0 0 0 0 ");
}

// ── 用例4：{10,4102} FLOAT32 整行溢出分支（Tile2D）──
TEST_F(LbpFeatureTiling, lbp_feature_tiling_test_2d_tile_overflow)
{
    RunOkCase(
        {10, 4102}, {8, 4100}, ge::DT_FLOAT, 1, 8, GET_TPL_TILING_KEY(LBP_FEATURE_SCH_MODE_TILE_2D),
        "34359738369 140874927349820 17609365913608 17617955848202 34359738369 4294967297 8589934592 2 3573412790273 "
        "21474837312 4294967336 -17613660884998 -17617955852295 17617955848191 0 0 0 0 0 0 0 0 0 0 0 0 -17613660880895 "
        "-17613660884997 4294963193 4294971398 0 0 0 0 0 0 0 0 0 0 0 4294967296 -17617955848192 -1 17613660884997 4102 "
        "0 0 0 0 0 0 0 0 0 0 0 4294967297 4294963195 17622250815487 17622250819590 0 0 0 0 0 0 0 0 0 0 0 "
        "4575657221408423936 4575657222454121421 1056964607 1045697485 1034924238 0 0 0 0 0 0 0 0 0 0 0 "
        "4539628420094492672 4491236500197683506 4444965756047720448 4491236500649803776 0 0 0 0 0 0 0 0 0 0 0 "
        "-9223372036854775808 1034924238 4575657222454121421 1056964607 1045697485 0 0 0 0 0 0 0 0 0 0 0 "
        "4491236501731934208 4444965756047720448 4491236499584450560 4539628420094492672 0 0 0 0 0 0 0 0 0 0 0 0 "
        "8589934595 8589934591 -4294967296 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4294967296 8589934593 8589934594 4294967298 0 0 "
        "0 0 0 0 0 0 0 0 0 0 0 8589934593 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 1 0 4294967296 4294967298 0 0 0 0 0 0 0 0 "
        "0 0 0 4294967296 1 4294967297 8589934594 2 0 0 0 0 0 0 0 0 0 0 0 ");
}

// ── 用例5：{3,6,70} 三维（前导维折叠进 batchCount）RowSplit ──
TEST_F(LbpFeatureTiling, lbp_feature_tiling_3d)
{
    RunOkCase(
        {3, 6, 70}, {3, 4, 68}, ge::DT_UINT8, 1, 8, GET_TPL_TILING_KEY(LBP_FEATURE_SCH_MODE_ROW_SPLIT),
        "34359738369 3504693314796 292057776132 300647710726 51539607555 4294967297 8589934592 2 292057776130 "
        "4294967392 4294967298 -296352743494 -300647710791 300647710719 0 0 0 0 0 0 0 0 0 0 0 0 -296352743423 "
        "-296352743493 4294967225 4294967366 0 0 0 0 0 0 0 0 0 0 0 4294967296 -300647710720 -1 296352743493 70 0 0 0 0 "
        "0 0 0 0 0 0 0 4294967297 4294967227 304942678015 304942678086 0 0 0 0 0 0 0 0 0 0 0 4575657221408423936 "
        "4575657222454121421 1056964607 1045697485 1034924238 0 0 0 0 0 0 0 0 0 0 0 4539628420094492672 "
        "4491236500197683506 4444965756047720448 4491236500649803776 0 0 0 0 0 0 0 0 0 0 0 -9223372036854775808 "
        "1034924238 4575657222454121421 1056964607 1045697485 0 0 0 0 0 0 0 0 0 0 0 4491236501731934208 "
        "4444965756047720448 4491236499584450560 4539628420094492672 0 0 0 0 0 0 0 0 0 0 0 4294967296 8589934595 "
        "8589934591 -4294967296 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4294967296 8589934593 8589934594 4294967298 0 0 0 0 0 0 0 "
        "0 0 0 0 0 0 8589934593 4294967297 0 0 0 0 0 0 0 0 0 0 0 0 1 0 4294967296 4294967298 0 0 0 0 0 0 0 0 0 0 0 "
        "4294967296 1 4294967297 8589934594 2 0 0 0 0 0 0 0 0 0 0 0 ");
}

// ── 失败用例：一维输入 ──
TEST_F(LbpFeatureTiling, lbp_feature_tiling_invalid_1d) { RunFailCase({70}, {70}, ge::DT_UINT8, 1, 8); }

// ── 失败用例：radius = -1 ──
TEST_F(LbpFeatureTiling, lbp_feature_tiling_invalid_neg_radius) { RunFailCase({6, 70}, {6, 70}, ge::DT_UINT8, -1, 8); }

// ── 失败用例：neighbors = 0 ──
TEST_F(LbpFeatureTiling, lbp_feature_tiling_invalid_neighbors_zero)
{
    RunFailCase({6, 70}, {6, 70}, ge::DT_UINT8, 1, 0);
}

// ── 失败用例：neighbors = 32（数组越界守卫）──
TEST_F(LbpFeatureTiling, lbp_feature_tiling_invalid_neighbors_32)
{
    RunFailCase({6, 70}, {6, 70}, ge::DT_UINT8, 1, 32);
}

// ── 失败用例：H = 2（H <= 2*radius，输出高度为 0）──
TEST_F(LbpFeatureTiling, lbp_feature_tiling_invalid_h_too_small) { RunFailCase({2, 70}, {0, 68}, ge::DT_UINT8, 1, 8); }

// ── 失败用例：W = 2（W <= 2*radius）──
TEST_F(LbpFeatureTiling, lbp_feature_tiling_invalid_w_too_small) { RunFailCase({6, 2}, {4, 0}, ge::DT_UINT8, 1, 8); }

// ── 失败用例：首维为 0（batchCount <= 0）──
TEST_F(LbpFeatureTiling, lbp_feature_tiling_invalid_zero_batch)
{
    RunFailCase({0, 3, 63, 65}, {0, 3, 61, 63}, ge::DT_UINT8, 1, 8);
}
