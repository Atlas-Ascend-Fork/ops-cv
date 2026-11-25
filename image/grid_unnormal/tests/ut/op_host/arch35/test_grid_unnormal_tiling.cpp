/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

#include <map>
#include <string>
#include "experiment/metadef/common/util/error_manager/error_manager.h"

namespace {
namespace errmsg_test {
inline void ClearErrorMessages() { (void)ErrorManager::GetInstance().GetRawErrorMessages(); }

inline void ExpectFirstError(const std::string& expectedCode, const std::map<std::string, std::string>& expectedArgs)
{
    auto errors = ErrorManager::GetInstance().GetRawErrorMessages();
    ASSERT_FALSE(errors.empty());
    EXPECT_EQ(errors.front().error_id, expectedCode);
    for (const auto& expected : expectedArgs) {
        const auto actual = errors.front().args_map.find(expected.first);
        ASSERT_NE(actual, errors.front().args_map.end());
        EXPECT_EQ(actual->second, expected.second);
    }
}
} // namespace errmsg_test
} // namespace

#include "any_value.h"
#include "../../../../op_kernel/arch35/grid_unnormal_tiling_data.h"

using namespace std;
using namespace ge;

class GridUnnormalTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GridUnnormalTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "GridUnnormalTiling TearDown" << std::endl; }
};

const GridUnnormalTilingData* GetTilingData(const TilingInfo& info)
{
    if (info.tilingData == nullptr || info.tilingDataSize < sizeof(GridUnnormalTilingData)) {
        return nullptr;
    }
    return reinterpret_cast<const GridUnnormalTilingData*>(info.tilingData.get());
}

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    shape.MutableOriginShape().SetDimNum(dims.size());
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape.MutableOriginShape().SetDim(i, dims[i]);
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return shape;
}

void ExpectEmptyTiling(const std::vector<int64_t>& shape)
{
    struct GridUnnormalCompileInfo {
    } compileInfo;
    const gert::StorageShape storageShape = MakeStorageShape(shape);
    const std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {storageShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {storageShape, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    const std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {storageShape, ge::DT_FLOAT, ge::FORMAT_ND},
        {storageShape, ge::DT_INT32, ge::FORMAT_ND},
    };
    gert::TilingContextPara tilingContextPara("GridUnnormal", inputs, outputs, {}, &compileInfo, "Ascend950", 48,
                                              262144, 4096);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(tilingContextPara, info));
    ASSERT_EQ(info.blockNum, 1U);
    const auto* td = GetTilingData(info);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->totalNum, 0);
    EXPECT_EQ(td->perCoreNum, 0);
    EXPECT_EQ(td->alignCorners, 0);
}

// fp32, align_corners=false, 小 shape → tiling success
TEST_F(GridUnnormalTiling, tiling_fp32_align_false)
{
    struct GridUnnormalCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom(false)),
    };
    gert::TilingContextPara tilingContextPara(
        "GridUnnormal",
        {
            {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // grid
            {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // assist
        },
        {
            {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND}, // diff
            {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_INT32, ge::FORMAT_ND}, // position
        },
        attrs, &compileInfo, "Ascend950", 48, 262144, 4096);
    TilingInfo info;
    bool ok = ExecuteTiling(tilingContextPara, info);
    ASSERT_TRUE(ok);
    ASSERT_EQ(info.blockNum, 30U);
    const auto* td = GetTilingData(info);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->totalNum, 60);
    EXPECT_EQ(td->perCoreNum, 2);
    EXPECT_EQ(td->ubFactor, 7936);
    EXPECT_EQ(td->alignCorners, 0);
}

// fp16, align_corners=true, 多核 shape → tiling success
TEST_F(GridUnnormalTiling, tiling_fp16_align_true)
{
    struct GridUnnormalCompileInfo {
    } compileInfo;
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        gert::TilingContextPara::OpAttr("align_corners", Ops::Cv::AnyValue::CreateFrom(true)),
    };
    gert::TilingContextPara tilingContextPara(
        "GridUnnormal",
        {
            {{{16, 128, 32, 2}, {16, 128, 32, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // grid
            {{{16, 128, 32, 2}, {16, 128, 32, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // assist
        },
        {
            {{{16, 128, 32, 2}, {16, 128, 32, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // diff
            {{{16, 128, 32, 2}, {16, 128, 32, 2}}, ge::DT_INT32, ge::FORMAT_ND},   // position
        },
        attrs, &compileInfo, "Ascend950", 48, 262144, 4096);
    TilingInfo info;
    bool ok = ExecuteTiling(tilingContextPara, info);
    ASSERT_TRUE(ok);
    ASSERT_EQ(info.blockNum, 48U);
    const auto* td = GetTilingData(info);
    ASSERT_NE(td, nullptr);
    EXPECT_EQ(td->totalNum, 131072);
    EXPECT_EQ(td->perCoreNum, 2731);
    EXPECT_EQ(td->ubFactor, 12672);
    EXPECT_EQ(td->alignCorners, 1);
}

TEST_F(GridUnnormalTiling, tiling_empty_tensor) { ExpectEmptyTiling({0, 6, 5, 2}); }

TEST_F(GridUnnormalTiling, tiling_empty_tensor_middle_axis) { ExpectEmptyTiling({1, 0, 5, 2}); }

TEST_F(GridUnnormalTiling, tiling_empty_tensor_third_axis) { ExpectEmptyTiling({1, 6, 0, 2}); }

TEST_F(GridUnnormalTiling, tiling_empty_tensor_multiple_axes) { ExpectEmptyTiling({0, 0, 5, 2}); }

TEST_F(GridUnnormalTiling, tiling_rejects_non_4d_grid)
{
    struct GridUnnormalCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("GridUnnormal",
                                              {
                                                  {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{64}, {64}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 48, 262144, 4096);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, info));
}

TEST_F(GridUnnormalTiling, tiling_rejects_last_dim_not_2)
{
    struct GridUnnormalCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("GridUnnormal",
                                              {
                                                  {{{1, 6, 5, 3}, {1, 6, 5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 6, 5, 3}, {1, 6, 5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 6, 5, 3}, {1, 6, 5, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 6, 5, 3}, {1, 6, 5, 3}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 48, 262144, 4096);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, info));
}

TEST_F(GridUnnormalTiling, tiling_rejects_mismatched_input_shapes)
{
    struct GridUnnormalCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("GridUnnormal",
                                              {
                                                  {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 6, 5, 1}, {1, 6, 5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 48, 262144, 4096);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, info));
}

TEST_F(GridUnnormalTiling, tiling_rejects_mismatched_input_dtypes)
{
    struct GridUnnormalCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("GridUnnormal",
                                              {
                                                  {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 48, 262144, 4096);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, info));
}

TEST_F(GridUnnormalTiling, tiling_rejects_unsupported_input_dtype)
{
    struct GridUnnormalCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("GridUnnormal",
                                              {
                                                  {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1, 6, 5, 2}, {1, 6, 5, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 48, 262144, 4096);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, info));
}

TEST_F(GridUnnormalTiling, tiling_rejects_negative_dimension)
{
    struct GridUnnormalCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("GridUnnormal",
                                              {
                                                  {{{1, -1, 5, 2}, {1, -1, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, -1, 5, 2}, {1, -1, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, -1, 5, 2}, {1, -1, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, -1, 5, 2}, {1, -1, 5, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 48, 262144, 4096);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, info));
}

TEST_F(GridUnnormalTiling, tiling_rejects_even_negative_dimensions)
{
    struct GridUnnormalCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("GridUnnormal",
                                              {
                                                  {{{-1, -1, 5, 2}, {-1, -1, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{-1, -1, 5, 2}, {-1, -1, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{-1, -1, 5, 2}, {-1, -1, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{-1, -1, 5, 2}, {-1, -1, 5, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 48, 262144, 4096);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, info));
}

TEST_F(GridUnnormalTiling, tiling_rejects_unknown_rank_dimensions)
{
    struct GridUnnormalCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("GridUnnormal",
                                              {
                                                  {{{-2, -2, 5, 2}, {-2, -2, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{-2, -2, 5, 2}, {-2, -2, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{-2, -2, 5, 2}, {-2, -2, 5, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{-2, -2, 5, 2}, {-2, -2, 5, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 48, 262144, 4096);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, info));
}

TEST_F(GridUnnormalTiling, tiling_rejects_insufficient_ub)
{
    struct GridUnnormalCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("GridUnnormal",
                                              {
                                                  {{{1, 4, 8, 2}, {1, 4, 8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 4, 8, 2}, {1, 4, 8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 4, 8, 2}, {1, 4, 8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 4, 8, 2}, {1, 4, 8, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 48, 8192, 4096);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, info));
}

TEST_F(GridUnnormalTiling, tiling_rejects_zero_core_count)
{
    struct GridUnnormalCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("GridUnnormal",
                                              {
                                                  {{{1, 4, 8, 2}, {1, 4, 8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 4, 8, 2}, {1, 4, 8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 4, 8, 2}, {1, 4, 8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 4, 8, 2}, {1, 4, 8, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 0, 262144, 4096);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, info));
}

TEST_F(GridUnnormalTiling, tiling_rejects_zero_ub)
{
    struct GridUnnormalCompileInfo {
    } compileInfo;
    gert::TilingContextPara tilingContextPara("GridUnnormal",
                                              {
                                                  {{{1, 4, 8, 2}, {1, 4, 8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 4, 8, 2}, {1, 4, 8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 4, 8, 2}, {1, 4, 8, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 4, 8, 2}, {1, 4, 8, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 48, 0, 4096);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, info));
}

// grid rank != 4 → shape error
TEST_F(GridUnnormalTiling, errmsg_grid_rank_wrong)
{
    struct GridUnnormalCompileInfo {
    } compileInfo;
    // rank-3 grid: the check reports EZ0010 (SHAPE family)
    gert::TilingContextPara tilingContextPara("GridUnnormal",
                                              {
                                                  {{{2, 4, 4}, {2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 4, 4}, {2, 4, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 4}, {2, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 4}, {2, 4}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {}, &compileInfo, "Ascend950", 48, 262144, 4096);
    errmsg_test::ClearErrorMessages();
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(tilingContextPara, info));
    auto errors = ErrorManager::GetInstance().GetRawErrorMessages();
    ASSERT_FALSE(errors.empty());
    EXPECT_EQ(errors.front().error_id, std::string("EZ0010"));
}
