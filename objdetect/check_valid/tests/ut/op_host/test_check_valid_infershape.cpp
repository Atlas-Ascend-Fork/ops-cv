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
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class CheckValidInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "CheckValidInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "CheckValidInfershape TearDown" << std::endl; }
};

TEST_F(CheckValidInfershape, cv_fp32_normal)
{
    gert::InfershapeContextPara infershapeContextPara("CheckValid",
                                                      {
                                                          {{{8, 4}, {8, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      },
                                                      std::vector<gert::InfershapeContextPara::OpAttr>{});
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(CheckValidInfershape, cv_fp16_normal)
{
    gert::InfershapeContextPara infershapeContextPara("CheckValid",
                                                      {
                                                          {{{8, 4}, {8, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                          {{{3}, {3}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      },
                                                      std::vector<gert::InfershapeContextPara::OpAttr>{});
    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(CheckValidInfershape, cv_fp32_empty)
{
    gert::InfershapeContextPara infershapeContextPara("CheckValid",
                                                      {
                                                          {{{0, 4}, {0, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      },
                                                      std::vector<gert::InfershapeContextPara::OpAttr>{});
    std::vector<std::vector<int64_t>> expectOutputShape = {{0, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(CheckValidInfershape, cv_fp32_large)
{
    gert::InfershapeContextPara infershapeContextPara("CheckValid",
                                                      {
                                                          {{{100000, 4}, {100000, 4}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      },
                                                      std::vector<gert::InfershapeContextPara::OpAttr>{});
    std::vector<std::vector<int64_t>> expectOutputShape = {{100000, 1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(CheckValidInfershape, cv_invalid_rank1)
{
    gert::InfershapeContextPara infershapeContextPara("CheckValid",
                                                      {
                                                          {{{8}, {8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      },
                                                      std::vector<gert::InfershapeContextPara::OpAttr>{});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(CheckValidInfershape, cv_invalid_rank3)
{
    gert::InfershapeContextPara infershapeContextPara("CheckValid",
                                                      {
                                                          {{{8, 4, 1}, {8, 4, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      },
                                                      std::vector<gert::InfershapeContextPara::OpAttr>{});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(CheckValidInfershape, cv_invalid_last_dim)
{
    gert::InfershapeContextPara infershapeContextPara("CheckValid",
                                                      {
                                                          {{{8, 3}, {8, 3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      },
                                                      std::vector<gert::InfershapeContextPara::OpAttr>{});
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED);
}

TEST_F(CheckValidInfershape, cv_unknown_rank)
{
    gert::InfershapeContextPara infershapeContextPara("CheckValid",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3}, {3}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},
                                                      },
                                                      std::vector<gert::InfershapeContextPara::OpAttr>{});
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(CheckValidInfershape, cv_invalid_bbox_negative_rows)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        for (int64_t rows : {-3, -2}) {
            SCOPED_TRACE(testing::Message() << "dtype=" << dtype << ", bbox N=" << rows);
            gert::InfershapeContextPara para(
                "CheckValid", {{{{rows, 4}, {rows, 4}}, dtype, ge::FORMAT_ND}, {{{3}, {3}}, dtype, ge::FORMAT_ND}},
                {{{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}}, std::vector<gert::InfershapeContextPara::OpAttr>{});
            ExecuteTestCase(para, ge::GRAPH_FAILED);
        }
    }
}

TEST_F(CheckValidInfershape, cv_img_metas_shape_contract)
{
    struct MetaCase {
        const char* name;
        gert::StorageShape shape;
        bool valid;
    };
    const MetaCase metaCases[] = {
        {"length3", {{3}, {3}}, true},          {"length16", {{16}, {16}}, true},
        {"dynamic_length", {{-1}, {-1}}, true}, {"unknown_rank", {{-2}, {-2}}, true},
        {"empty", {{0}, {0}}, false},           {"length1", {{1}, {1}}, false},
        {"length2", {{2}, {2}}, false},         {"scalar", {{}, {}}, false},
        {"rank2", {{1, 3}, {1, 3}}, false},     {"rank3", {{1, 1, 3}, {1, 1, 3}}, false},
    };
    struct BboxCase {
        gert::StorageShape shape;
        std::vector<int64_t> outputShape;
    };
    const BboxCase bboxCases[] = {
        {{{8, 4}, {8, 4}}, {8, 1}},
        {{{0, 4}, {0, 4}}, {0, 1}},
        {{{-1, 4}, {-1, 4}}, {-1, 1}},
        {{{-2}, {-2}}, {-2}},
    };
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        for (const auto& bbox : bboxCases) {
            for (const auto& meta : metaCases) {
                SCOPED_TRACE(testing::Message() << "dtype=" << dtype << ", meta=" << meta.name
                                                << ", output first dim=" << bbox.outputShape[0]);
                gert::InfershapeContextPara para(
                    "CheckValid", {{bbox.shape, dtype, ge::FORMAT_ND}, {meta.shape, dtype, ge::FORMAT_ND}},
                    {{{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}}, std::vector<gert::InfershapeContextPara::OpAttr>{});
                ExecuteTestCase(para, meta.valid ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED, {bbox.outputShape});
            }
        }
    }
}

TEST_F(CheckValidInfershape, cv_missing_img_metas)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_FLOAT}) {
        for (const gert::StorageShape& bbox : {gert::StorageShape({8, 4}, {8, 4}), gert::StorageShape({0, 4}, {0, 4}),
                                               gert::StorageShape({-1, 4}, {-1, 4}), gert::StorageShape({-2}, {-2})}) {
            SCOPED_TRACE(testing::Message() << "dtype=" << dtype << ", bbox=" << bbox.GetOriginShape().GetDim(0));
            gert::InfershapeContextPara para("CheckValid", {{bbox, dtype, ge::FORMAT_ND}},
                                             {{{{}, {}}, ge::DT_INT8, ge::FORMAT_ND}},
                                             std::vector<gert::InfershapeContextPara::OpAttr>{});
            ExecuteTestCase(para, ge::GRAPH_FAILED);
        }
    }
}
