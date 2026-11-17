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
#include <limits>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
namespace {
template <typename A>
std::vector<typename A::OpAttr> Attrs(std::vector<int64_t> k = {3, 3, 3}, std::string mode = "REFLECT", float cval = 0)
{
    using V = Ops::Cv::AnyValue;
    return {{"kernel_size", V::CreateFrom<std::vector<int64_t>>(k)},
            {"mode", V::CreateFrom<std::string>(mode)},
            {"cval", V::CreateFrom<float>(cval)}};
}
void ShapeCase(std::vector<int64_t> dims, bool pass, std::vector<int64_t> k = {3, 3, 3}, std::string mode = "REFLECT",
               float cval = 0)
{
    gert::StorageShape s;
    for (auto d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    gert::InfershapeContextPara::TensorDescription desc(s, ge::DT_FLOAT, ge::FORMAT_ND);
    gert::InfershapeContextPara p("MedianFilter3D", {desc}, {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                  Attrs<gert::InfershapeContextPara>(k, mode, cval));
    if (pass) {
        ExecuteTestCase(p, ge::GRAPH_SUCCESS, {dims});
    } else {
        ExecuteTestCase(p, ge::GRAPH_FAILED);
    }
}
void TileCase(std::vector<int64_t> dims, ge::DataType dt, bool pass, std::vector<int64_t> k = {3, 3, 3},
              std::string mode = "REFLECT", float cval = 0)
{
    gert::StorageShape s;
    for (auto d : dims) {
        s.MutableOriginShape().AppendDim(d);
        s.MutableStorageShape().AppendDim(d);
    }
    gert::TilingContextPara::TensorDescription desc(s, dt, ge::FORMAT_ND);
    uint64_t compileInfo = 0;
    gert::TilingContextPara p("MedianFilter3D", {desc}, {desc}, Attrs<gert::TilingContextPara>(k, mode, cval),
                              &compileInfo);
    if (pass) {
        TilingInfo info;
        ASSERT_TRUE(ExecuteTiling(p, info));
        EXPECT_EQ(info.tilingKey, 0);
        EXPECT_GT(info.blockNum, 0);
    } else {
        ExecuteTestCase(p, ge::GRAPH_FAILED);
    }
}
} // namespace
