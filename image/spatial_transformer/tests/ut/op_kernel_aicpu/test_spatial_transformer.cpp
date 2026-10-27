/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <complex>
#include "gtest/gtest.h"
#ifndef private
#define private public
#define protected public
#endif
#include "utils/aicpu_test_utils.h"
#include "cpu_kernel_utils.h"
#include "node_def_builder.h"
#undef private
#undef protected
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

Eigen::half half_zero = Eigen::half(0);

class TEST_SPATIAL_TRANSFORMER_UT : public testing::Test {};

#define CREATE_NODEDEF(shapes, data_types, datas, format, stn_ori_channel, use_default_theta, default_theta) \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();                                         \
    NodeDefBuilder(node_def.get(), "SpatialTransformer", "SpatialTransformer")                               \
        .Input({"x", data_types[0], shapes[0], datas[0], format})                                            \
        .Input({"theta", data_types[1], shapes[1], datas[1]})                                                \
        .Output({"y", data_types[2], shapes[2], datas[2], format})                                           \
        .Attr("stn_ori_channel", stn_ori_channel)                                                            \
        .Attr("use_default_theta", use_default_theta)                                                        \
        .Attr("default_theta", default_theta);
namespace {
constexpr uint64_t kRankValidationBufferSize = 1 * 1 * 2 * 3 * 16;
void RunRankValidationCase(const vector<vector<int64_t>>& shapes, Format format)
{
    vector<DataType> data_types = {DT_INT8, DT_FLOAT16, DT_INT8};
    int8_t input_x[kRankValidationBufferSize] = {-39, -47, -37, 4, -70, -47};
    Eigen::half input_theta[6] = {Eigen::half(-1), Eigen::half(-2), Eigen::half(0),
                                  Eigen::half(0),  Eigen::half(0),  Eigen::half(0)};
    int8_t output[kRankValidationBufferSize] = {0};
    vector<void*> datas = {(void*)input_x, (void*)input_theta, (void*)output};
    vector<int64_t> use_default_theta = {1, 0, 1, 0, 1, 1};
    vector<float> default_theta = {1.0f, 0.0f, 1.5f, 0.0f};
    CREATE_NODEDEF(shapes, data_types, datas, format, 1, use_default_theta, default_theta);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

void RunThetaAttrValidationCase(const vector<int64_t>& use_default_theta, const vector<float>& default_theta,
                                uint32_t expected_status)
{
    vector<DataType> data_types = {DT_INT8, DT_FLOAT16, DT_INT8};
    vector<vector<int64_t>> shapes = {{1, 1, 2, 3}, {2, 3}, {1, 1, 2, 3}};
    int8_t input_x[6] = {-39, -47, -37, 4, -70, -47};
    Eigen::half input_theta[6] = {Eigen::half(1), Eigen::half(0), Eigen::half(0),
                                  Eigen::half(0), Eigen::half(1), Eigen::half(0)};
    int8_t output[6] = {0};
    vector<void*> datas = {(void*)input_x, (void*)input_theta, (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, FORMAT_NCHW, 1, use_default_theta, default_theta);
    RUN_KERNEL(node_def, HOST, expected_status);
}

void RunSharedTheta5DBatchCase(int32_t stn_ori_channel)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 1, 1, 1, 16}, {2, 3}, {2, 1, 1, 1, 16}};
    constexpr uint64_t data_size = 2 * 1 * 1 * 1 * 16;
    float input_x[data_size] = {0.0f};
    for (int32_t batch = 0; batch < 2; ++batch) {
        for (int32_t channel = 0; channel < stn_ori_channel; ++channel) {
            input_x[batch * 16 + channel] = static_cast<float>(batch * 16 + channel + 1);
        }
    }
    float input_theta[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    float output[data_size] = {0.0f};
    vector<void*> datas = {(void*)input_x, (void*)input_theta, (void*)output};
    vector<int64_t> use_default_theta;
    vector<float> default_theta;
    CREATE_NODEDEF(shapes, data_types, datas, FORMAT_NC1HWC0, stn_ori_channel, use_default_theta, default_theta);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult(output, input_x, data_size));
}
} // namespace

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, EMPTY_THETA_ATTRS_SUCC) { RunThetaAttrValidationCase({}, {}, KERNEL_STATUS_OK); }

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, USE_DEFAULT_THETA_LENGTH_FAIL)
{
    RunThetaAttrValidationCase({0}, {}, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, USE_DEFAULT_THETA_VALUE_FAIL)
{
    RunThetaAttrValidationCase({0, 0, 0, 0, 0, 2}, {}, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, DEFAULT_THETA_LENGTH_FAIL)
{
    RunThetaAttrValidationCase({1, 0, 1, 0, 0, 0}, {1.0f}, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, 4D_SUCC)
{
    vector<DataType> data_types = {DT_INT8, DT_FLOAT16, DT_INT8};
    vector<vector<int64_t>> shapes = {{1, 1, 2, 3}, {2, 3}, {1, 1, 2, 3}};

    constexpr uint64_t input1_size = 1 * 1 * 2 * 3;
    int8_t input1[input1_size] = {-39, -47, -37, 4, -70, -47};

    constexpr uint64_t input2_size = 6;

    Eigen::half input2[input2_size] = {Eigen::half(-1), Eigen::half(-2), Eigen::half(0),
                                       Eigen::half(0),  Eigen::half(0),  Eigen::half(0)};

    constexpr uint64_t output_size = 1 * 1 * 2 * 3;
    int8_t output[output_size] = {0};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    vector<int64_t> use_default_theta = {1, 0, 1, 0, 1, 1};
    vector<float> default_theta = {1.0f, 0.0f, 1.5f, 0.0f};
    CREATE_NODEDEF(shapes, data_types, datas, FORMAT_NCHW, 1, use_default_theta, default_theta);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    int8_t output_exp[output_size] = {-35, 0, 0, 0, -34, -32};

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, 5D_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_INT8, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 1, 2, 3, 16}, {2, 3}, {1, 1, 2, 3, 16}};

    // read data from file for input1
    constexpr uint64_t input1_size = 1 * 1 * 2 * 3 * 16;
    Eigen::half input1[input1_size] = {half_zero};
    for (uint32_t i = 0; i < 16; i++) {
        if (i < 10) {
            input1[0 * 16 + i] = Eigen::half(-12.304688);
            input1[1 * 16 + i] = Eigen::half(-24.906250);
            input1[2 * 16 + i] = Eigen::half(4);
            input1[3 * 16 + i] = Eigen::half(-42.25);
            input1[4 * 16 + i] = Eigen::half(-35.125);
            input1[5 * 16 + i] = Eigen::half(-70);
        }
    }

    // read data from file for input2
    constexpr uint64_t input2_size = 6;
    int8_t input2[input2_size] = {-2, -1, 0, 0, 0, 0};

    constexpr uint64_t output_size = 1 * 1 * 2 * 3 * 16;
    Eigen::half output[output_size] = {half_zero};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    vector<int64_t> use_default_theta = {1, 0, 1, 0, 1, 1};
    vector<float> default_theta = {1.0f, 0.0f, 1.5f, 0.0f};
    CREATE_NODEDEF(shapes, data_types, datas, FORMAT_NC1HWC0, 10, use_default_theta, default_theta);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // read data from file for expect ouput
    Eigen::half output_exp[output_size] = {half_zero};
    for (uint32_t i = 0; i < 16; i++) {
        if (i < 10) {
            output_exp[0 * 16 + i] = Eigen::half(0);
            output_exp[1 * 16 + i] = Eigen::half(-34.0);
            output_exp[2 * 16 + i] = Eigen::half(0);
            output_exp[3 * 16 + i] = Eigen::half(0);
            output_exp[4 * 16 + i] = Eigen::half(-12.302080);
            output_exp[5 * 16 + i] = Eigen::half(-15.499998);
        }
    }

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, 5D_C1_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_FLOAT16, DT_FLOAT16};
    vector<vector<int64_t>> shapes = {{1, 1, 2, 3, 16}, {2, 3}, {1, 1, 2, 3, 16}};

    // read data from file for input1
    constexpr uint64_t input1_size = 1 * 1 * 2 * 3 * 16;
    Eigen::half input1[input1_size] = {half_zero};
    for (uint32_t i = 0; i < 1; i++) {
        if (i < 1) {
            input1[0 * 16 + i] = Eigen::half(-12.304688);
            input1[1 * 16 + i] = Eigen::half(-24.906250);
            input1[2 * 16 + i] = Eigen::half(4);
            input1[3 * 16 + i] = Eigen::half(-42.25);
            input1[4 * 16 + i] = Eigen::half(-35.125);
            input1[5 * 16 + i] = Eigen::half(-70);
        }
    }

    // read data from file for input2
    constexpr uint64_t input2_size = 6;
    Eigen::half input2[input2_size] = {Eigen::half(-2), Eigen::half(-1), Eigen::half(0),
                                       Eigen::half(0),  Eigen::half(0),  Eigen::half(0)};

    constexpr uint64_t output_size = 1 * 1 * 2 * 3 * 16;
    Eigen::half output[output_size] = {half_zero};
    vector<void*> datas = {(void*)input1, (void*)input2, (void*)output};

    vector<int64_t> use_default_theta = {1, 0, 1, 0, 1, 1};
    vector<float> default_theta = {1.0f, 0.0f, 1.5f, 0.0f};
    CREATE_NODEDEF(shapes, data_types, datas, FORMAT_NC1HWC0, 1, use_default_theta, default_theta);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // read data from file for expect ouput
    Eigen::half output_exp[output_size] = {half_zero};
    for (uint32_t i = 0; i < 1; i++) {
        if (i < 1) {
            output_exp[0 * 16 + i] = Eigen::half(0);
            output_exp[1 * 16 + i] = Eigen::half(-34.0);
            output_exp[2 * 16 + i] = Eigen::half(0);
            output_exp[3 * 16 + i] = Eigen::half(0);
            output_exp[4 * 16 + i] = Eigen::half(-12.302080);
            output_exp[5 * 16 + i] = Eigen::half(-15.499998);
        }
    }

    bool compare = CompareResult(output, output_exp, output_size);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, OPTIONAL_THETA_DEFAULT_ONLY_SUCC)
{
    constexpr uint64_t input_size = 1 * 1 * 2 * 3;
    float input[input_size] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    float output[input_size] = {0.0F};

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SpatialTransformer", "SpatialTransformer")
        .Input({"x", DT_FLOAT, {1, 1, 2, 3}, input, FORMAT_NCHW})
        .Output({"y", DT_FLOAT, {1, 1, 2, 3}, output, FORMAT_NCHW})
        .Attr("stn_ori_channel", 1)
        .Attr("use_default_theta", vector<int64_t>{1, 1, 1, 1, 1, 1})
        .Attr("default_theta", vector<float>{1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F});

    string node_def_str;
    node_def->SerializeToString(node_def_str);
    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(node_def.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(ctx.GetInputsSize(), 1U);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
    EXPECT_TRUE(CompareResult(output, input, input_size));
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, OPTIONAL_THETA_REQUIRED_FAIL)
{
    float input[6] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    float output[6] = {0.0F};

    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SpatialTransformer", "SpatialTransformer")
        .Input({"x", DT_FLOAT, {1, 1, 2, 3}, input, FORMAT_NCHW})
        .Output({"y", DT_FLOAT, {1, 1, 2, 3}, output, FORMAT_NCHW})
        .Attr("stn_ori_channel", 1)
        .Attr("use_default_theta", vector<int64_t>{1, 0, 1, 0, 1, 1})
        .Attr("default_theta", vector<float>{1.0F, 0.0F, 0.0F, 0.0F});

    string node_def_str;
    node_def->SerializeToString(node_def_str);
    CpuKernelContext ctx(HOST);
    EXPECT_EQ(ctx.Init(node_def.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, OUTPUT_FORMAT_MISMATCH_FAIL)
{
    vector<int64_t> x_shape = {1, 1, 2, 3};
    vector<int64_t> theta_shape = {2, 3};
    vector<int64_t> y_shape = {1, 1, 2, 3, 16};
    vector<DataType> data_types = {DT_INT8, DT_FLOAT16, DT_INT8};

    int8_t input_x[6] = {-39, -47, -37, 4, -70, -47};
    Eigen::half input_theta[6] = {Eigen::half(-1), Eigen::half(-2), Eigen::half(0),
                                  Eigen::half(0),  Eigen::half(0),  Eigen::half(0)};
    int8_t output[96] = {0};

    vector<int64_t> use_default_theta = {1, 0, 1, 0, 1, 1};
    vector<float> default_theta = {1.0f, 0.0f, 1.5f, 0.0f};
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();
    NodeDefBuilder(node_def.get(), "SpatialTransformer", "SpatialTransformer")
        .Input({"x", data_types[0], x_shape, input_x, FORMAT_NCHW})
        .Input({"theta", data_types[1], theta_shape, input_theta})
        .Output({"y", data_types[2], y_shape, output, FORMAT_NC1HWC0})
        .Attr("stn_ori_channel", 1)
        .Attr("use_default_theta", use_default_theta)
        .Attr("default_theta", default_theta);

    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
TEST_F(TEST_SPATIAL_TRANSFORMER_UT, NCHW_RANK_3_FAIL)
{
    RunRankValidationCase({{1, 1, 6}, {2, 3}, {1, 1, 2, 3}}, FORMAT_NCHW);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, NCHW_RANK_5_FAIL)
{
    RunRankValidationCase({{1, 1, 2, 3, 1}, {2, 3}, {1, 1, 2, 3}}, FORMAT_NCHW);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, NCHW_OUTPUT_RANK_3_FAIL)
{
    RunRankValidationCase({{1, 1, 2, 3}, {2, 3}, {1, 1, 6}}, FORMAT_NCHW);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, NCHW_OUTPUT_RANK_5_FAIL)
{
    RunRankValidationCase({{1, 1, 2, 3}, {2, 3}, {1, 1, 2, 3, 1}}, FORMAT_NCHW);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, NC1HWC0_RANK_4_FAIL)
{
    RunRankValidationCase({{1, 1, 2, 3}, {2, 3}, {1, 1, 2, 3, 16}}, FORMAT_NC1HWC0);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, NC1HWC0_OUTPUT_RANK_4_FAIL)
{
    RunRankValidationCase({{1, 1, 2, 3, 16}, {2, 3}, {1, 1, 2, 3}}, FORMAT_NC1HWC0);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, SHARED_THETA_BATCH_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 1, 2, 3}, {2, 3}, {2, 1, 2, 3}};
    constexpr uint64_t data_size = 2 * 1 * 2 * 3;
    float input_x[data_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    float input_theta[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    float output[data_size] = {0.0f};
    vector<void*> datas = {(void*)input_x, (void*)input_theta, (void*)output};
    vector<int64_t> use_default_theta;
    vector<float> default_theta;
    CREATE_NODEDEF(shapes, data_types, datas, FORMAT_NCHW, 1, use_default_theta, default_theta);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    EXPECT_TRUE(CompareResult(output, input_x, data_size));
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, SHARED_THETA_BATCH_MASKED_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 1, 2, 3}, {2, 3}, {2, 1, 2, 3}};
    constexpr uint64_t data_size = 2 * 1 * 2 * 3;
    float input_x[data_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    float input_theta[6] = {1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    float output[data_size] = {0.0f};
    vector<void*> datas = {(void*)input_x, (void*)input_theta, (void*)output};
    vector<int64_t> use_default_theta = {0, 1, 0, 1, 0, 1};
    vector<float> default_theta = {0.0f, 0.0f, 0.0f};
    CREATE_NODEDEF(shapes, data_types, datas, FORMAT_NCHW, 1, use_default_theta, default_theta);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // Both batches share the same compact theta slice, so the second batch must not continue past it.
    EXPECT_TRUE(CompareResult(output, input_x, data_size));
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, SHARED_THETA_5D_BATCH_SUCC) { RunSharedTheta5DBatchCase(16); }

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, SHARED_THETA_5D_C1_BATCH_SUCC) { RunSharedTheta5DBatchCase(1); }

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, THETA_BATCH_ONE_MISMATCH_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 1, 2, 3}, {1, 2, 3}, {2, 1, 2, 3}};
    constexpr uint64_t data_size = 2 * 1 * 2 * 3;
    float input_x[data_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    float input_theta[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    float output[data_size] = {0.0f};
    vector<void*> datas = {(void*)input_x, (void*)input_theta, (void*)output};
    vector<int64_t> use_default_theta;
    vector<float> default_theta;
    CREATE_NODEDEF(shapes, data_types, datas, FORMAT_NCHW, 1, use_default_theta, default_theta);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, PER_BATCH_THETA_MASKED_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 1, 2, 3}, {2, 2, 3}, {2, 1, 2, 3}};
    constexpr uint64_t data_size = 2 * 1 * 2 * 3;
    float input_x[data_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    float input_theta[12] = {1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    float output[data_size] = {0.0f};
    vector<void*> datas = {(void*)input_x, (void*)input_theta, (void*)output};
    vector<int64_t> use_default_theta = {0, 1, 0, 1, 0, 1};
    vector<float> default_theta = {0.0f, 0.0f, 0.0f};
    CREATE_NODEDEF(shapes, data_types, datas, FORMAT_NCHW, 1, use_default_theta, default_theta);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    // The second batch must start at its own six coefficients and flip its rows.
    float output_exp[data_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 14.0f, 15.0f, 16.0f, 11.0f, 12.0f, 13.0f};
    EXPECT_TRUE(CompareResult(output, output_exp, data_size));
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, UNDERSIZED_THETA_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 1, 2, 3}, {2}, {2, 1, 2, 3}};
    constexpr uint64_t data_size = 2 * 1 * 2 * 3;
    float input_x[data_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    float input_theta[2] = {1.0f, 0.0f};
    float output[data_size] = {0.0f};
    vector<void*> datas = {(void*)input_x, (void*)input_theta, (void*)output};
    vector<int64_t> use_default_theta;
    vector<float> default_theta;
    CREATE_NODEDEF(shapes, data_types, datas, FORMAT_NCHW, 1, use_default_theta, default_theta);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, THETA_RANK_1_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 1, 2, 3}, {6}, {2, 1, 2, 3}};
    constexpr uint64_t data_size = 2 * 1 * 2 * 3;
    float input_x[data_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    float input_theta[6] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    float output[data_size] = {0.0f};
    vector<void*> datas = {(void*)input_x, (void*)input_theta, (void*)output};
    vector<int64_t> use_default_theta;
    vector<float> default_theta;
    CREATE_NODEDEF(shapes, data_types, datas, FORMAT_NCHW, 1, use_default_theta, default_theta);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, THETA_BATCH_MISMATCH_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 1, 2, 3}, {3, 2, 3}, {2, 1, 2, 3}};
    constexpr uint64_t data_size = 2 * 1 * 2 * 3;
    constexpr uint64_t theta_size = 3 * 2 * 3;
    float input_x[data_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    float input_theta[theta_size] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    float output[data_size] = {0.0f};
    vector<void*> datas = {(void*)input_x, (void*)input_theta, (void*)output};
    vector<int64_t> use_default_theta;
    vector<float> default_theta;
    CREATE_NODEDEF(shapes, data_types, datas, FORMAT_NCHW, 1, use_default_theta, default_theta);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SPATIAL_TRANSFORMER_UT, THETA_DIMS_MISMATCH_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{2, 1, 2, 3}, {3, 3}, {2, 1, 2, 3}};
    constexpr uint64_t data_size = 2 * 1 * 2 * 3;
    constexpr uint64_t theta_size = 3 * 3;
    float input_x[data_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    float input_theta[theta_size] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    float output[data_size] = {0.0f};
    vector<void*> datas = {(void*)input_x, (void*)input_theta, (void*)output};
    vector<int64_t> use_default_theta;
    vector<float> default_theta;
    CREATE_NODEDEF(shapes, data_types, datas, FORMAT_NCHW, 1, use_default_theta, default_theta);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
