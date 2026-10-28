/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>
#include <iostream>
#include <gtest/gtest.h>
#include "op_api/utils/ut_op_util.h"
#include "matrix_calculation_ops.h"
#include "op_tiling/op_tiling_util.h"
#include "register/op_tiling_registry.h"
#include "experiment_ops.h"
#include "selection_ops.h"
#include "test_common.h"
#include "common_unittest.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "../../../op_host/three_interpolate_backward_tiling.h"

using namespace std;
using namespace ge;

class TestThreeInterpolateBackwardTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "test ThreeInterpolateBackwardTiling SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "test ThreeInterpolateBackwardTiling TearDown" << std::endl; }
};

namespace {
constexpr uint32_t INDEX_INPUT_GRAD_X = 0u;
constexpr uint32_t INDEX_INPUT_IDX = 1u;
constexpr uint32_t INDEX_INPUT_WEIGHT = 2u;
constexpr uint32_t INDEX_OUTPUT_GRAD_Y = 0u;
constexpr uint32_t C0 = 16;

DataType StringToDtype(std::string dtype_string)
{
    auto find_it = optiling::STR_TO_DATATYPE.find(dtype_string);
    if (find_it != optiling::STR_TO_DATATYPE.end()) {
        return find_it->second;
    }
    return ge::DT_FLOAT16;
}

void add_input_desc_by_idx(Operator& op, int64_t idx, std::vector<int64_t> input_shape,
                           std::vector<std::string> data_dtypes, Format format)
{
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    op_info->MutableInputDesc(idx)->SetShape(GeShape(input_shape));
    op_info->MutableInputDesc(idx)->SetOriginShape(GeShape(input_shape));
    op_info->MutableInputDesc(idx)->SetFormat(format);
    op_info->MutableInputDesc(idx)->SetOriginFormat(format);
    op_info->MutableInputDesc(idx)->SetDataType(StringToDtype(data_dtypes[idx]));
}

void add_output_desc_by_idx(Operator& op, int64_t idx, std::vector<int64_t> input_shape,
                            std::vector<std::string> data_dtypes, Format format)
{
    auto op_info = OpDescUtils::GetOpDescFromOperator(op);
    op_info->MutableOutputDesc(idx)->SetShape(GeShape(input_shape));
    op_info->MutableOutputDesc(idx)->SetOriginShape(GeShape(input_shape));
    op_info->MutableOutputDesc(idx)->SetFormat(format);
    op_info->MutableOutputDesc(idx)->SetOriginFormat(format);
    op_info->MutableOutputDesc(idx)->SetDataType(StringToDtype(data_dtypes[idx]));
}

void run_parse_test(optiling::ThreeInterpolateBackwardCompileInfo& compile_info)
{
    std::string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 0,
            "load3d_constraints": "1",
            "Intrinsic_fix_pipe_l0c2out": false,
            "Intrinsic_data_move_l12ub": true,
            "Intrinsic_data_move_l0c2ub": true,
            "Intrinsic_data_move_out2l1_nd2nz": false,
            "UB_SIZE": 262144,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
            "CORE_NUM": 48}})";

    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();

    std::string op_type("ThreeInterpolateBackward");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
                                                                                            intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);
}

// shape_info {b ,c, m, n}
void run_case(std::vector<int64_t> shape_info, std::vector<std::string> data_dtypes, std::string& expect_tiling)
{
    auto test_op = op::ThreeInterpolateBackward("ThreeInterpolateBackward");
    auto bs = shape_info[0];
    auto cs = shape_info[1];
    auto ms = shape_info[2];
    auto ns = shape_info[3];
    auto c1 = (cs + C0 - 1) / C0;

    std::vector<int64_t> grad_x_shape = {bs, c1, ns, 1, C0};
    std::vector<int64_t> index_shape = {bs, ns, 3};
    std::vector<int64_t> weight_shape = {bs, ns, 3};
    std::vector<int64_t> grad_y_shape = {bs, c1, ms, 1, C0};

    add_input_desc_by_idx(test_op, INDEX_INPUT_GRAD_X, grad_x_shape, data_dtypes, FORMAT_NC1HWC0);
    add_input_desc_by_idx(test_op, INDEX_INPUT_IDX, index_shape, data_dtypes, FORMAT_ND);
    add_input_desc_by_idx(test_op, INDEX_INPUT_WEIGHT, weight_shape, data_dtypes, FORMAT_ND);
    add_output_desc_by_idx(test_op, INDEX_OUTPUT_GRAD_Y, grad_y_shape, data_dtypes, FORMAT_NC1HWC0);

    optiling::ThreeInterpolateBackwardCompileInfo compile_info;
    run_parse_test(compile_info);
    std::unique_ptr<uint8_t[]> tilingdata;
    EXPECT_EQ(TilingTest(test_op, &compile_info, tilingdata), ge::GRAPH_SUCCESS);
    gert::TilingData* raw_tiling_data = reinterpret_cast<gert::TilingData*>(tilingdata.get());
    ASSERT_NE(raw_tiling_data, nullptr);
    EXPECT_EQ(to_string<uint32_t>(raw_tiling_data->GetData(), raw_tiling_data->GetDataSize()), expect_tiling);
}
} // namespace

TEST_F(TestThreeInterpolateBackwardTiling, test_case_1_5_6_1_fp32)
{
    std::string expect_tiling = "1 1 1 6 1 1 1 16 1 1 1 16 1 6 6 32 96 1 1 1 0 0 0 0 ";
    run_case({1, 5, 6, 1}, {"float", "int32", "float"}, expect_tiling);
}

TEST_F(TestThreeInterpolateBackwardTiling, test_case_10_50_600_10_fp32)
{
    std::string expect_tiling = "10 10 4 600 10 1 1 16 1 1 1 16 1 6 6 128 384 4 4 1 0 0 0 0 ";
    run_case({10, 50, 600, 10}, {"float", "int32", "float"}, expect_tiling);
}

TEST_F(TestThreeInterpolateBackwardTiling, test_case_40_600_100_12_fp32)
{
    std::string expect_tiling = "12 40 38 100 12 1 1 16 1 1 1 16 1 6 6 1216 3648 38 38 1 0 0 0 0 ";
    run_case({40, 600, 100, 12}, {"float", "int32", "float"}, expect_tiling);
}

TEST_F(TestThreeInterpolateBackwardTiling, test_case_100_20_20_11_fp32)
{
    std::string expect_tiling = "48 100 2 20 11 11 1 16 11 11 1 16 11 6 6 64 192 2 2 1 1 2 4 0 ";
    run_case({100, 20, 20, 11}, {"float", "int32", "float"}, expect_tiling);
}

TEST_F(TestThreeInterpolateBackwardTiling, test_case_10_200_30_30_fp32)
{
    std::string expect_tiling = "30 10 13 30 30 1 1 16 1 1 1 16 1 6 6 416 1248 13 13 1 0 0 0 0 ";
    run_case({10, 200, 30, 30}, {"float", "int32", "float"}, expect_tiling);
}

TEST_F(TestThreeInterpolateBackwardTiling, test_case_40_200_30_30_fp32)
{
    std::string expect_tiling = "30 40 13 30 30 1 1 16 1 1 1 16 1 6 6 416 1248 13 13 1 0 0 0 0 ";
    run_case({40, 200, 30, 30}, {"float", "int32", "float"}, expect_tiling);
}

// issue #1436：ns ≈ 2.74e8 时各核基址偏移 core_offset * C0 已超出 uint32 承载
// （47 * 5711418 * 16 ≈ 4.3e9 > 2^32），tiling 各字段须完整下发不回绕（kernel 侧偏移已 uint64 化）
TEST_F(TestThreeInterpolateBackwardTiling, test_case_1_1_2_274148085_fp32)
{
    std::string expect_tiling = "48 1 1 2 274148085 5711418 356964 16 10 5711439 356965 16 15 6 6 32 96 1 1 1 0 0 0 0 ";
    run_case({1, 1, 2, 274148085}, {"float", "int32", "float"}, expect_tiling);
}

TEST_F(TestThreeInterpolateBackwardTiling, test_case_4_20_30_30_fp32)
{
    std::string expect_tiling = "30 4 2 30 30 1 1 16 1 1 1 16 1 6 6 64 192 2 2 1 0 0 0 0 ";
    run_case({4, 20, 30, 30}, {"float", "int32", "float"}, expect_tiling);
}

TEST_F(TestThreeInterpolateBackwardTiling, test_case_2_5_6_6_fp16)
{
    std::string expect_tiling = "2 2 1 6 6 6 1 16 6 6 1 16 6 3 6 16 48 1 1 1 1 1 0 0 ";
    run_case({2, 5, 6, 6}, {"float16", "int32", "float16"}, expect_tiling);
}

TEST_F(TestThreeInterpolateBackwardTiling, test_case_3_20_20_3_fp16)
{
    std::string expect_tiling = "3 3 2 20 3 3 1 16 3 3 1 16 3 3 6 32 96 2 2 1 1 1 0 0 ";
    run_case({3, 20, 20, 3}, {"float16", "int32", "float16"}, expect_tiling);
}

TEST_F(TestThreeInterpolateBackwardTiling, test_case_10_400_200_30_fp16)
{
    std::string expect_tiling = "10 10 25 200 30 30 2 16 14 30 2 16 14 3 6 400 1200 25 25 1 1 1 0 0 ";
    run_case({10, 400, 200, 30}, {"float16", "int32", "float16"}, expect_tiling);
}

TEST_F(TestThreeInterpolateBackwardTiling, test_case_100_10_30_30_fp16)
{
    std::string expect_tiling = "48 100 1 30 30 30 2 16 14 30 2 16 14 3 6 16 48 1 1 1 1 2 4 0 ";
    run_case({100, 10, 30, 30}, {"float16", "int32", "float16"}, expect_tiling);
}

TEST_F(TestThreeInterpolateBackwardTiling, test_case_1_10_600_60_fp16)
{
    std::string expect_tiling = "1 1 1 600 60 60 4 16 12 60 4 16 12 3 6 16 48 1 1 1 1 1 0 0 ";
    run_case({1, 10, 600, 60}, {"float16", "int32", "float16"}, expect_tiling);
}

TEST_F(TestThreeInterpolateBackwardTiling, test_case_1000_100_100_10_fp16)
{
    std::string expect_tiling = "48 1000 7 100 10 10 1 16 10 10 1 16 10 3 6 112 336 7 7 1 1 20 40 0 ";
    run_case({1000, 100, 100, 10}, {"float16", "int32", "float16"}, expect_tiling);
}

namespace {
// issue #1407 / #1436 负例辅助：独立指定各张量形状，期望 tiling 校验失败
void run_case_expect_fail(std::vector<int64_t> grad_x_shape, std::vector<int64_t> idx_shape,
                          std::vector<int64_t> weight_shape, std::vector<int64_t> grad_y_shape,
                          std::vector<std::string> data_dtypes)
{
    auto test_op = op::ThreeInterpolateBackward("ThreeInterpolateBackward");
    add_input_desc_by_idx(test_op, INDEX_INPUT_GRAD_X, grad_x_shape, data_dtypes, FORMAT_NC1HWC0);
    add_input_desc_by_idx(test_op, INDEX_INPUT_IDX, idx_shape, data_dtypes, FORMAT_ND);
    add_input_desc_by_idx(test_op, INDEX_INPUT_WEIGHT, weight_shape, data_dtypes, FORMAT_ND);
    add_output_desc_by_idx(test_op, INDEX_OUTPUT_GRAD_Y, grad_y_shape, data_dtypes, FORMAT_NC1HWC0);

    optiling::ThreeInterpolateBackwardCompileInfo compile_info;
    run_parse_test(compile_info);
    std::unique_ptr<uint8_t[]> tilingdata;
    EXPECT_NE(TilingTest(test_op, &compile_info, tilingdata), ge::GRAPH_SUCCESS);
}
} // namespace

// issue #1407：idx/weight 的 n 维与 grad_x 的 ns 维不一致，须拒绝
TEST_F(TestThreeInterpolateBackwardTiling, shape_check_idx_n_mismatch_rejected)
{
    run_case_expect_fail({2, 1, 4, 1, C0}, {2, 5, 3}, {2, 5, 3}, {2, 1, 6, 1, C0}, {"float", "int32", "float"});
}

// issue #1407：idx/weight 的 batch 维与 grad_x 的 bs 维不一致，须拒绝
TEST_F(TestThreeInterpolateBackwardTiling, shape_check_idx_batch_mismatch_rejected)
{
    run_case_expect_fail({2, 1, 4, 1, C0}, {3, 4, 3}, {3, 4, 3}, {2, 1, 6, 1, C0}, {"float", "int32", "float"});
}

// issue #1407：idx/weight 末维不为 3，须拒绝
TEST_F(TestThreeInterpolateBackwardTiling, shape_check_idx_last_dim_not_3_rejected)
{
    run_case_expect_fail({2, 1, 4, 1, C0}, {2, 4, 2}, {2, 4, 2}, {2, 1, 6, 1, C0}, {"float", "int32", "float"});
}

// issue #1407：weight 与 idx 形状不一致，须拒绝
TEST_F(TestThreeInterpolateBackwardTiling, shape_check_weight_diff_idx_rejected)
{
    run_case_expect_fail({2, 1, 4, 1, C0}, {2, 4, 3}, {2, 5, 3}, {2, 1, 6, 1, C0}, {"float", "int32", "float"});
}

// issue #1407：idx/weight rank 不为 3，须拒绝
TEST_F(TestThreeInterpolateBackwardTiling, shape_check_idx_rank_2_rejected)
{
    run_case_expect_fail({2, 1, 4, 1, C0}, {2, 4}, {2, 4}, {2, 1, 6, 1, C0}, {"float", "int32", "float"});
}

// issue #1407：grad_y 的 bs 维与 grad_x 不一致，须拒绝
TEST_F(TestThreeInterpolateBackwardTiling, shape_check_grad_y_bs_mismatch_rejected)
{
    run_case_expect_fail({2, 1, 4, 1, C0}, {2, 4, 3}, {2, 4, 3}, {1, 1, 6, 1, C0}, {"float", "int32", "float"});
}

// issue #1436：维度值超出 int32 承载范围（ns = 2^31），写入 uint32 tiling 字段会隐式截断，须拒绝
TEST_F(TestThreeInterpolateBackwardTiling, shape_check_ns_over_int32_max_rejected)
{
    run_case_expect_fail({1, 1, 2147483648, 1, C0}, {1, 2147483648, 3}, {1, 2147483648, 3}, {1, 1, 2, 1, C0},
                         {"float", "int32", "float"});
}
