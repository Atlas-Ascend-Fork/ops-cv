/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <vector>

#include "infershape_case_executor.h"
#include "infershape_context_faker.h"

namespace {

// Helper: 4 inputs (images NHWC + size + scale + translation), 1 output (y NHWC)
// sizeVal: if non-null, size is const with these values; if null, size is non-const
void ExpectShape(const gert::StorageShape& imagesShape, const gert::StorageShape& sizeShape, const int32_t* sizeVal,
                 ge::graphStatus expectedStatus, const std::vector<std::vector<int64_t>>& expectedShapes = {})
{
    gert::InfershapeContextPara context("ScaleAndTranslate",
                                        {{imagesShape, ge::DT_FLOAT, ge::FORMAT_NHWC},
                                         {sizeShape, ge::DT_INT32, ge::FORMAT_ND, sizeVal != nullptr,
                                          const_cast<void*>(static_cast<const void*>(sizeVal))},
                                         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                         {{{2}, {2}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                        {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_NHWC}});
    ExecuteTestCase(context, expectedStatus, expectedShapes);
}

} // namespace

// ===== 正常路径：const size 读值推导输出 =====

TEST(ScaleAndTranslateInfershape, infers_static_shape_with_const_size)
{
    const int32_t size[] = {4, 5};
    ExpectShape({{2, 8, 9, 3}, {2, 8, 9, 3}}, {{2}, {2}}, size, ge::GRAPH_SUCCESS, {{2, 4, 5, 3}});
}

TEST(ScaleAndTranslateInfershape, infers_static_shape_batch_one)
{
    const int32_t size[] = {50, 60};
    ExpectShape({{1, 100, 200, 3}, {1, 100, 200, 3}}, {{2}, {2}}, size, ge::GRAPH_SUCCESS, {{1, 50, 60, 3}});
}

TEST(ScaleAndTranslateInfershape, preserves_dynamic_dimensions)
{
    const int32_t size[] = {4, 5};
    ExpectShape({{-1, 8, 9, -1}, {-1, 8, 9, -1}}, {{2}, {2}}, size, ge::GRAPH_SUCCESS, {{-1, 4, 5, -1}});
}

TEST(ScaleAndTranslateInfershape, size_non_const_keeps_output_unknown)
{
    ExpectShape({{2, 8, 9, 3}, {2, 8, 9, 3}}, {{2}, {2}}, nullptr, ge::GRAPH_SUCCESS, {{2, -1, -1, 3}});
}

TEST(ScaleAndTranslateInfershape, size_unknown_dim_keeps_output_unknown)
{
    ExpectShape({{2, 8, 9, 3}, {2, 8, 9, 3}}, {{-1}, {-1}}, nullptr, ge::GRAPH_SUCCESS, {{2, -1, -1, 3}});
}

// ===== unknown-rank 处理 =====

TEST(ScaleAndTranslateInfershape, images_unknown_rank_passes_through)
{
    ExpectShape({{-2}, {-2}}, {{2}, {2}}, nullptr, ge::GRAPH_SUCCESS, {{-2}});
}

TEST(ScaleAndTranslateInfershape, size_unknown_rank_keeps_output_unknown)
{
    ExpectShape({{2, 8, 9, 3}, {2, 8, 9, 3}}, {{-2}, {-2}}, nullptr, ge::GRAPH_SUCCESS, {{2, -1, -1, 3}});
}

// ===== 校验失败路径 =====

TEST(ScaleAndTranslateInfershape, rejects_images_rank_not_4)
{
    const int32_t size[] = {4, 5};
    ExpectShape({{2, 8, 3}, {2, 8, 3}}, {{2}, {2}}, size, ge::GRAPH_FAILED);
}

TEST(ScaleAndTranslateInfershape, rejects_images_rank_5)
{
    const int32_t size[] = {4, 5};
    ExpectShape({{2, 8, 9, 3, 1}, {2, 8, 9, 3, 1}}, {{2}, {2}}, size, ge::GRAPH_FAILED);
}

TEST(ScaleAndTranslateInfershape, rejects_size_rank_not_1)
{
    ExpectShape({{2, 8, 9, 3}, {2, 8, 9, 3}}, {{2, 2}, {2, 2}}, nullptr, ge::GRAPH_FAILED);
}

TEST(ScaleAndTranslateInfershape, rejects_size_elements_not_2)
{
    ExpectShape({{2, 8, 9, 3}, {2, 8, 9, 3}}, {{3}, {3}}, nullptr, ge::GRAPH_FAILED);
}

TEST(ScaleAndTranslateInfershape, rejects_negative_size_value)
{
    const int32_t size[] = {-1, 5};
    ExpectShape({{2, 8, 9, 3}, {2, 8, 9, 3}}, {{2}, {2}}, size, ge::GRAPH_FAILED);
}

TEST(ScaleAndTranslateInfershape, rejects_zero_size_value)
{
    const int32_t size[] = {0, 5};
    ExpectShape({{2, 8, 9, 3}, {2, 8, 9, 3}}, {{2}, {2}}, size, ge::GRAPH_FAILED);
}

TEST(ScaleAndTranslateInfershape, rejects_both_negative_size_values)
{
    const int32_t size[] = {-1, -1};
    ExpectShape({{2, 8, 9, 3}, {2, 8, 9, 3}}, {{2}, {2}}, size, ge::GRAPH_FAILED);
}
