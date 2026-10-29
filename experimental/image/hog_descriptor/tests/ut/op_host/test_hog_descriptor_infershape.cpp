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
#include "base/registry/op_impl_space_registry_v2.h"
#include "hog_descriptor_test_attrs.h"

namespace {
using InferAttr = gert::InfershapeContextPara::OpAttr;

void ExpectHogInferFailure(const HogTestAttrs& attrs, const gert::StorageShape& shape = {{128, 128}, {128, 128}})
{
    gert::InfershapeContextPara para("HOGDescriptor", {{shape, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeHogAttrs<InferAttr>(attrs));
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}
} // namespace

TEST(HogDescriptorInfershape, Fp16InputDescriptorShape)
{
    gert::InfershapeContextPara para("HOGDescriptor", {{{{128, 128}, {128, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeHogAttrs<InferAttr>());
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>{{8100}});
}

TEST(HogDescriptorInfershape, SingletonBatchIsFlattened)
{
    gert::InfershapeContextPara para("HOGDescriptor", {{{{1, 24, 32, 1}, {1, 24, 32, 1}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeHogAttrs<InferAttr>());
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>{{216}});
}

TEST(HogDescriptorInfershape, RgbBatch)
{
    gert::InfershapeContextPara para("HOGDescriptor", {{{{2, 24, 32, 3}, {2, 24, 32, 3}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, MakeHogAttrs<InferAttr>());
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>{{2, 216}});
}

TEST(HogDescriptorInfershape, MalformedPairsAreRejected)
{
    HogTestAttrs p;
    p.ppc = {8};
    ExpectHogInferFailure(p);
    p = HogTestAttrs{};
    p.cpb = {};
    ExpectHogInferFailure(p);
    p = HogTestAttrs{};
    p.stride = {1, 1, 1};
    ExpectHogInferFailure(p);
}

TEST(HogDescriptorInfershape, ZeroDivisorsAreRejected)
{
    HogTestAttrs p;
    p.ppc = {0, 8};
    ExpectHogInferFailure(p);
    p = HogTestAttrs{};
    p.stride = {1, 0};
    ExpectHogInferFailure(p);
}

TEST(HogDescriptorInfershape, InvalidNormalizationAndEps)
{
    HogTestAttrs p;
    p.norm = "unknown";
    ExpectHogInferFailure(p);
    p = HogTestAttrs{};
    p.eps = std::numeric_limits<float>::quiet_NaN();
    ExpectHogInferFailure(p);
}

TEST(HogDescriptorInfershape, RanksBelowTwoAreRejected)
{
    ExpectHogInferFailure(HogTestAttrs{}, {{}, {}});
    ExpectHogInferFailure(HogTestAttrs{}, {{128}, {128}});
}

TEST(HogDescriptorInfershape, RanksAboveFourAreRejected)
{
    ExpectHogInferFailure(HogTestAttrs{}, {{1, 16, 16, 1, 1}, {1, 16, 16, 1, 1}});
}
