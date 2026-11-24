/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "any_value.h"
#include "base/err_msg.h"
#include "tiling_case_executor.h"
#include "tiling_context_faker.h"

namespace {
struct ReportedError {
    std::string code;
    std::vector<std::string> keys;
    std::vector<std::string> values;
};

std::vector<ReportedError>& ReportedErrors()
{
    static std::vector<ReportedError> errors;
    return errors;
}
} // namespace

// Test-side strong definition that preempts the weak liberror_manager
// symbol at dynamic-link time (the executable's definition wins).  It only
// records each structured (error code, key, value) report so tests can
// assert the reported classification instead of merely the failure return.
namespace error_message {
int32_t ReportPredefinedErrMsg(const char* error_code, const std::vector<const char*>& key,
                               const std::vector<const char*>& value)
{
    ReportedError error{error_code != nullptr ? error_code : "", {}, {}};
    for (const char* k : key) {
        error.keys.emplace_back(k != nullptr ? k : "");
    }
    for (const char* v : value) {
        error.values.emplace_back(v != nullptr ? v : "");
    }
    ReportedErrors().push_back(std::move(error));
    return 0;
}
} // namespace error_message

namespace {
struct CompileInfo {
    uint32_t coreNum{64};
    uint64_t ubSize{262144};
};

gert::StorageShape ScalarShape() { return gert::StorageShape({}, {}); }

gert::TilingContextPara MakeContext(const gert::StorageShape& maxOutputShape, const gert::StorageShape& iouShape,
                                    const gert::StorageShape& scoreShape, int64_t boxesNum = 2,
                                    ge::DataType boxesDtype = ge::DT_FLOAT, uint64_t ubSize = 262144,
                                    ge::Format boxesFormat = ge::FORMAT_ND)
{
    static CompileInfo compileInfo;
    compileInfo = CompileInfo{64, ubSize};
    return gert::TilingContextPara(
        "SortedNMS",
        {
            {gert::StorageShape({boxesNum, 4}, {boxesNum, 4}), boxesDtype, boxesFormat},
            {gert::StorageShape({boxesNum}, {boxesNum}), boxesDtype, ge::FORMAT_ND},
            {gert::StorageShape({boxesNum}, {boxesNum}), ge::DT_INT32, ge::FORMAT_ND},
            {maxOutputShape, ge::DT_INT32, ge::FORMAT_ND},
            {iouShape, boxesDtype, ge::FORMAT_ND},
            {scoreShape, boxesDtype, ge::FORMAT_ND},
        },
        {{gert::StorageShape({boxesNum}, {boxesNum}), ge::DT_INT32, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("offset", Ops::Cv::AnyValue::CreateFrom<int64_t>(0))}, &compileInfo,
        "Ascend950", 64, compileInfo.ubSize);
}

gert::StorageShape OneElementShape() { return gert::StorageShape({1}, {1}); }

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

TEST(SortedNMSTiling, rejects_non_nd_storage_format)
{
    auto context = MakeContext(ScalarShape(), ScalarShape(), ScalarShape(), 2, ge::DT_FLOAT, 262144, ge::FORMAT_NCHW);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(SortedNMSTiling, rejects_insufficient_ub_for_simt_reserve)
{
    // Pairwise-mask strategy (1025 boxes) with float16 boxes cannot reserve
    // the 128 KiB SIMT Data Cache from a 64 KiB UB.
    auto context = MakeContext(ScalarShape(), ScalarShape(), ScalarShape(), 1025, ge::DT_FLOAT16, 64U * 1024U);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));
}

TEST(SortedNMSTiling, reports_ez1008_for_platform_resource_failure)
{
    // Assert the actual structured report for a platform-resource failure:
    // the code must be EZ1008 with the opName/reason fields (registered by
    // nnopbase in the aclnn production runtime; probe-verified that the weak
    // symbol degrades to a safe no-op when libnnopbase is not loaded).
    ReportedErrors().clear();
    auto context = MakeContext(ScalarShape(), ScalarShape(), ScalarShape(), 1025, ge::DT_FLOAT16, 64U * 1024U);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));

    ASSERT_FALSE(ReportedErrors().empty());
    const auto& error = ReportedErrors().back();
    EXPECT_EQ(error.code, "EZ1008");
    ASSERT_EQ(error.keys.size(), 2U);
    EXPECT_EQ(error.keys[0], "opName");
    EXPECT_EQ(error.keys[1], "reason");
    ASSERT_GE(error.values.size(), 2U);
    EXPECT_EQ(error.values[0], "SortedNMS");
    EXPECT_NE(error.values[1].find("SIMT Data Cache"), std::string::npos);
}

TEST(SortedNMSTiling, classifies_shape_errors_as_shape_codes)
{
    // A rank-two scalar parameter must report the shape classification with
    // the parameter name, not a generic argument/parameter-value error.
    ReportedErrors().clear();
    const auto rankTwo = gert::StorageShape({1, 1}, {1, 1});
    const auto one = OneElementShape();
    auto context = MakeContext(rankTwo, one, one);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));

    ASSERT_FALSE(ReportedErrors().empty());
    const auto& error = ReportedErrors().back();
    EXPECT_EQ(error.code, "EZ0009") << "actual: " << error.code;
    const std::string shapeParam = [&error]() -> std::string {
        const auto it = std::find(error.keys.begin(), error.keys.end(), "param_name");
        return it != error.keys.end() ? error.values[static_cast<size_t>(it - error.keys.begin())] : "";
    }();
    EXPECT_EQ(shapeParam, "max_output_size");
}

TEST(SortedNMSTiling, classifies_dtype_errors_as_dtype_codes)
{
    // An unsupported boxes dtype must report the dtype classification with
    // the boxes parameter name and the actual dtype.
    ReportedErrors().clear();
    auto context = MakeContext(ScalarShape(), ScalarShape(), ScalarShape(), 2, ge::DT_INT32);
    TilingInfo info;
    EXPECT_FALSE(ExecuteTiling(context, info));

    ASSERT_FALSE(ReportedErrors().empty());
    const auto& error = ReportedErrors().back();
    EXPECT_EQ(error.code, "EZ0019") << "actual: " << error.code;
    const auto it = std::find(error.keys.begin(), error.keys.end(), "param_name");
    ASSERT_TRUE(it != error.keys.end());
    EXPECT_EQ(error.values[static_cast<size_t>(it - error.keys.begin())], "boxes");
}
} // namespace
