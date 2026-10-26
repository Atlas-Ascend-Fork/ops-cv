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
#include <string>

#include <gtest/gtest.h>

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

namespace {

struct YoloCompileInfo {};
YoloCompileInfo g_compile_info;

gert::TilingContextPara MakeCase(ge::DataType input_dtype, ge::DataType coord_dtype, ge::DataType obj_dtype,
                                 ge::DataType classes_dtype)
{
    return gert::TilingContextPara(
        "Yolo", {{{{1, 255, 13, 13}, {1, 255, 13, 13}}, input_dtype, ge::FORMAT_ND}},
        {{{{1, 12, 192}, {1, 12, 192}}, coord_dtype, ge::FORMAT_ND},
         {{{1, 528}, {1, 528}}, obj_dtype, ge::FORMAT_ND},
         {{{1, 80, 528}, {1, 80, 528}}, classes_dtype, ge::FORMAT_ND}},
        {gert::TilingContextPara::OpAttr("boxes", Ops::Cv::AnyValue::CreateFrom<int64_t>(3)),
         gert::TilingContextPara::OpAttr("coords", Ops::Cv::AnyValue::CreateFrom<int64_t>(4)),
         gert::TilingContextPara::OpAttr("classes", Ops::Cv::AnyValue::CreateFrom<int64_t>(80)),
         gert::TilingContextPara::OpAttr("yolo_version", Ops::Cv::AnyValue::CreateFrom<std::string>("V3")),
         gert::TilingContextPara::OpAttr("softmax", Ops::Cv::AnyValue::CreateFrom<bool>(false)),
         gert::TilingContextPara::OpAttr("background", Ops::Cv::AnyValue::CreateFrom<bool>(false))},
        &g_compile_info, "Ascend950", 64, 262144, 4096);
}

bool RunCase(const char* name, const gert::TilingContextPara& para)
{
    TilingInfo info;
    const bool success = ExecuteTiling(para, info);
    std::cout << "case=" << name << " tiling_success=" << (success ? "true" : "false") << std::endl;
    return success;
}

} // namespace

TEST(YoloOutputDtypeContract, HostTilingRequiresAllOutputsToMatchInput)
{
    struct DtypeCase {
        const char* name;
        ge::DataType input;
        ge::DataType coord;
        ge::DataType obj;
        ge::DataType classes;
        bool expected_success;
    };

    const DtypeCase cases[] = {
        {"all_fp32", ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, true},
        {"all_fp16", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, true},
        {"fp32_coord_fp16", ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT, false},
        {"fp32_obj_fp16", ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT, false},
        {"fp32_classes_fp16", ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, false},
        {"fp16_coord_fp32", ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, false},
        {"fp16_obj_fp32", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16, false},
        {"fp16_classes_fp32", ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, false},
        {"invalid_int32_input", ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, false},
    };

    for (const auto& test_case : cases) {
        const bool actual_success = RunCase(
            test_case.name, MakeCase(test_case.input, test_case.coord, test_case.obj, test_case.classes));
        EXPECT_EQ(actual_success, test_case.expected_success) << test_case.name;
    }
}
