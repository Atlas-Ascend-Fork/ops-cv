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
 * \file test_scharr_tiling.cpp
 * \brief Scharr Tiling UT：校验 tilingKey / workspace / tilingData（冻结基线）。
 *        平台常量按 910B4：40 核 / UB 192KB。
 *        注：UT 假平台未填充 lib API workspace 字段，GetLibApiWorkSpaceSize() 返回
 *        uint32 最大值（4294967295），此值仅用于对齐 UT 环境基线。
 */

#include <gtest/gtest.h>
#include <iostream>
#include <cstring>
#include <vector>

#include "../../../op_kernel/scharr_tiling_data.h"
#include "../../../op_kernel/scharr_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace Ops::Cv;

static const std::vector<size_t> UT_FAKE_WORKSPACE = {4294967295U};

class ScharrTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ScharrTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ScharrTiling TearDown" << std::endl; }

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

    static void RunOneCase(const std::vector<int64_t>& shape, ge::DataType dtype, const std::string& direction,
                           const std::string& border, const std::string& expectTilingData)
    {
        struct ScharrCompileInfo {
        } compileInfo;
        gert::TilingContextPara::TensorDescription xDesc(MakeStorageShape(shape), dtype, ge::FORMAT_ND);
        gert::TilingContextPara::TensorDescription yDesc(MakeStorageShape(shape), ge::DT_FLOAT, ge::FORMAT_ND);
        gert::TilingContextPara para(
            "Scharr", {xDesc}, {yDesc},
            {gert::TilingContextPara::OpAttr("border_type", Ops::Cv::AnyValue::CreateFrom<std::string>(border)),
             gert::TilingContextPara::OpAttr("direction", Ops::Cv::AnyValue::CreateFrom<std::string>(direction))},
            &compileInfo, "", 40, 196608, 4096);
        uint64_t expectTilingKey = GET_TPL_TILING_KEY(SCHARR_SCH_MODE_DEFAULT);
        ExecuteTestCase(para, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, UT_FAKE_WORKSPACE);
    }
};

TEST_F(ScharrTiling, scharr_tiling_u8_x_reflect101_2d)
{
    RunOneCase({65, 64}, ge::DT_UINT8, "X", "REFLECT_101",
               "279172878400 4294967360 3 8589934613 0 0 279172874304 274877906944 412316860416 412316860480 "
               "274877907072 71468255811872 11682311079424 4398046511104 1536 0 0 9483287791616 2464 0 0 ");
}

TEST_F(ScharrTiling, scharr_tiling_f32_y_replicate_2d)
{
    RunOneCase({65, 64}, ge::DT_FLOAT, "Y", "REPLICATE",
               "279172878400 4294967360 5 13 0 0 279172874304 309237645344 309237645312 309237645376 274877907072 "
               "71468255824736 359815180189696 79164837199872 158329674427392 237494511645696 316659348864000 "
               "318033738334208 320232761664064 0 4294967297 ");
}

TEST_F(ScharrTiling, scharr_tiling_u8_x_constant_fallback_w2)
{
    RunOneCase({16, 2}, ge::DT_UINT8, "X", "CONSTANT",
               "68719476768 4294967298 1 16 0 0 64 549755813888 412316860416 137438953600 34359738432 2199023255840 "
               "19928648253440 7696581394432 2560 0 0 15530601745664 4128 0 8589934592 ");
}
