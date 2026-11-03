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
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

using namespace Ops::Cv; // AnyValue 在这个命名空间里

class LbpFeatureInfershape : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "LbpFeatureInfershape SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "LbpFeatureInfershape TearDown" << std::endl; }
};

// ════════════════════════════════════════════════════════════
// 用例1：4D 输入，radius=1，非对齐尺寸
//   N=2偶, C=3奇, H=63奇, W=65奇
//   期望：H: 63-2=61, W: 65-2=63 → {2, 3, 61, 63}
//   输出 dtype：INT32（LBP code 是 8bit 整数，但 kernel 统一以 int32 输出）
// ════════════════════════════════════════════════════════════
TEST_F(LbpFeatureInfershape, lbp_feature_infershape_test1)
{
    // 构造上下文
    gert::InfershapeContextPara infershapeContextPara(
        "LbpFeature",
        {
            {{{2, 3, 63, 65}, {2, 3, 63, 65}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            // 输出 out 的 shape 先置空，交由 infershape 去推导 —— dtype 为 INT32
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            // 传入属性，让 infershape 能读到 radius=1
            gert::InfershapeContextPara::OpAttr("radius", AnyValue::CreateFrom<int64_t>(1)),
            gert::InfershapeContextPara::OpAttr("neighbors", AnyValue::CreateFrom<int64_t>(8)),
        });

    // 设定预期结果
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3, 61, 63},
    };

    // 调用接口执行用例
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ════════════════════════════════════════════════════════════
// 用例2：2D 输入 {6,70} —— 与 tiling 的 L1 用例形状呼应
//   H: 6-2=4, W: 70-2=68 → {4, 68}
// ════════════════════════════════════════════════════════════
TEST_F(LbpFeatureInfershape, lbp_feature_infershape_test_2d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "LbpFeature",
        {
            {{{6, 70}, {6, 70}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{4, 68}, {4, 68}} /*dtype修正*/, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("radius", AnyValue::CreateFrom<int64_t>(1)),
            gert::InfershapeContextPara::OpAttr("neighbors", AnyValue::CreateFrom<int64_t>(8)),
        });

    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 68},
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ════════════════════════════════════════════════════════════
// 用例3：radius=2 —— 防止 attr 没被读、减2被硬编码的最关键用例
//   若实现错误地硬编码"各减2"，实际会推出 {4,68} 而非 {2,66} → 变红
// ════════════════════════════════════════════════════════════
TEST_F(LbpFeatureInfershape, lbp_feature_infershape_test_radius2)
{
    gert::InfershapeContextPara infershapeContextPara(
        "LbpFeature",
        {
            {{{6, 70}, {6, 70}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("radius", AnyValue::CreateFrom<int64_t>(2)), // radius=2！
            gert::InfershapeContextPara::OpAttr("neighbors", AnyValue::CreateFrom<int64_t>(8)),
        });

    // H: 6-2*2=2, W: 70-2*2=66 → {2, 66}（不是 {4,68}！）
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 66},
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ════════════════════════════════════════════════════════════
// 用例4：3D 输入 {3,6,70} —— 演示维度折叠
//   前 1 维折叠进 batch（batchCount=3），末两维参与 LBP
//   H: 6-2=4, W: 70-2=68 → {3, 4, 68}
// ════════════════════════════════════════════════════════════
TEST_F(LbpFeatureInfershape, lbp_feature_infershape_test_3d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "LbpFeature",
        {
            {{{3, 6, 70}, {3, 6, 70}}, ge::DT_UINT8, ge::FORMAT_ND}, // 三维！
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("radius", AnyValue::CreateFrom<int64_t>(1)),
            gert::InfershapeContextPara::OpAttr("neighbors", AnyValue::CreateFrom<int64_t>(8)),
        });

    // 首维原样保留，末两维各减 2 → {3, 4, 68}
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {3, 4, 68},
    };

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ════════════════════════════════════════════════════════════
// 用例5：H=2, radius=1 —— H <= 2*radius，输出高度为 0，应拒绝
//   踩等号边界（H=2 恰好 = 2*radius）
// ════════════════════════════════════════════════════════════
TEST_F(LbpFeatureInfershape, lbp_feature_infershape_invalid_h_too_small)
{
    gert::InfershapeContextPara infershapeContextPara(
        "LbpFeature",
        {
            {{{2, 70}, {2, 70}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("radius", AnyValue::CreateFrom<int64_t>(1)),
            gert::InfershapeContextPara::OpAttr("neighbors", AnyValue::CreateFrom<int64_t>(8)),
        });

    // 守卫应拒绝：H=2 不大于 2*radius=2
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// ════════════════════════════════════════════════════════════
// 用例6：W=2 —— W <= 2*radius，输出宽度为 0，应拒绝
// ════════════════════════════════════════════════════════════
TEST_F(LbpFeatureInfershape, lbp_feature_infershape_invalid_w_too_small)
{
    gert::InfershapeContextPara infershapeContextPara(
        "LbpFeature",
        {
            {{{6, 2}, {6, 2}}, ge::DT_UINT8, ge::FORMAT_ND}, // W=2 = 2*radius
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("radius", AnyValue::CreateFrom<int64_t>(1)),
            gert::InfershapeContextPara::OpAttr("neighbors", AnyValue::CreateFrom<int64_t>(8)),
        });

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// ════════════════════════════════════════════════════════════
// 用例7：首维为 0 —— batchCount = 0×3 = 0，应拒绝
//   对应实现中【新增守卫②】：前导维 <= 0 检查
//   （修复前：{0,3,63,65} 会静默推出 {0,3,61,63} 并返回 SUCCESS）
// ════════════════════════════════════════════════════════════
TEST_F(LbpFeatureInfershape, lbp_feature_infershape_invalid_zero_batch)
{
    gert::InfershapeContextPara infershapeContextPara(
        "LbpFeature",
        {
            {{{0, 3, 63, 65}, {0, 3, 63, 65}}, ge::DT_UINT8, ge::FORMAT_ND}, // 首维 0！
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("radius", AnyValue::CreateFrom<int64_t>(1)),
            gert::InfershapeContextPara::OpAttr("neighbors", AnyValue::CreateFrom<int64_t>(8)),
        });

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// ════════════════════════════════════════════════════════════
// 用例8：1D 输入 {70} —— 维度不足，应拒绝
// ════════════════════════════════════════════════════════════
TEST_F(LbpFeatureInfershape, lbp_feature_infershape_invalid_1d)
{
    gert::InfershapeContextPara infershapeContextPara(
        "LbpFeature",
        {
            {{{70}, {70}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("radius", AnyValue::CreateFrom<int64_t>(1)),
            gert::InfershapeContextPara::OpAttr("neighbors", AnyValue::CreateFrom<int64_t>(8)),
        });

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}

// ════════════════════════════════════════════════════════════
// 用例9：radius=-1 —— 负半径会让输出比输入还大（6-2×(-1)=8），必须拒绝
//   对应实现中【新增守卫①】：radius < 0 检查
//   （修复前：{6,70} 会静默推出 {8,72} 并返回 SUCCESS——输出比输入还大！）
// ════════════════════════════════════════════════════════════
TEST_F(LbpFeatureInfershape, lbp_feature_infershape_invalid_neg_radius)
{
    gert::InfershapeContextPara infershapeContextPara(
        "LbpFeature",
        {
            {{{6, 70}, {6, 70}}, ge::DT_UINT8, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            gert::InfershapeContextPara::OpAttr("radius", AnyValue::CreateFrom<int64_t>(-1)), // 负半径！
            gert::InfershapeContextPara::OpAttr("neighbors", AnyValue::CreateFrom<int64_t>(8)),
        });

    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, {});
}
