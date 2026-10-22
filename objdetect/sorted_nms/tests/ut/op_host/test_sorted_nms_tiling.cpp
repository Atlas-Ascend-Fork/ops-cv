/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>

#include <gtest/gtest.h>

#include "any_value.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
struct CompileInfo {
    uint32_t coreNum{64};
    uint64_t ubSize{262144};
};

gert::StorageShape ScalarShape() { return {gert::Shape{}, gert::Shape{}}; }

gert::TilingContextPara MakeContext(const gert::StorageShape& maxOutputShape, const gert::StorageShape& iouShape,
                                    const gert::StorageShape& scoreShape)
{
    static CompileInfo compileInfo;
    return gert::TilingContextPara(
        "SortedNMS",
        {
            {gert::StorageShape({2, 4}, {2, 4}), ge::DT_FLOAT, ge::FORMAT_ND},
            {gert::StorageShape({2}, {2}), ge::DT_FLOAT, ge::FORMAT_ND},
            {gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND},
            {maxOutputShape, ge::DT_INT32, ge::FORMAT_ND},
            {iouShape, ge::DT_FLOAT, ge::FORMAT_ND},
            {scoreShape, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {{gert::StorageShape({2}, {2}), ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("offset", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        "Ascend950", 64, compileInfo.ubSize);
}

gert::StorageShape OneElementShape() { return {gert::Shape{1}, gert::Shape{1}}; }

TEST(SortedNMSTiling, accepts_scalar_parameters)
{
    auto context = MakeContext(ScalarShape(), ScalarShape(), ScalarShape());
    TilingInfo info;
    EXPECT_TRUE(ExecuteTiling(context, info));
}

TEST(SortedNMSTiling, accepts_one_dimensional_single_element_parameters)
{
    const auto one = OneElementShape();
    auto context = MakeContext(one, one, one);
    TilingInfo info;
    EXPECT_TRUE(ExecuteTiling(context, info));
}

TEST(SortedNMSTiling, rejects_rank_two_single_element_parameter)
{
    const auto rankTwo = gert::StorageShape({1, 1}, {1, 1});
    const auto one = OneElementShape();
    auto context = MakeContext(rankTwo, one, one);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}
} // namespace
