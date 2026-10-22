/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
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
#include <limits>
#include <random>

using namespace std;
using namespace aicpu;

class TEST_SCALEANDTRANSLATE_UT : public testing::Test {};

template <typename T>
void SetCheckerboardImageInput(int64_t batch_size, int64_t num_row_squares, int64_t num_col_squares,
                               int64_t square_size, int64_t num_channels, std::vector<T>& data)
{
    const int64_t row_size = num_col_squares * square_size * num_channels;
    const int64_t image_size = num_row_squares * square_size * row_size;
    data.resize(batch_size * image_size);
    typedef std::mt19937 RNG_Engine;
    RNG_Engine rng;
    rng.seed(0);
    std::uniform_real_distribution<float> Unifrom_01(0, 1);
    std::vector<float> col(num_channels);
    for (int b = 0; b < batch_size; ++b) {
        for (int y = 0; y < num_row_squares; ++y) {
            for (int x = 0; x < num_col_squares; ++x) {
                for (int n = 0; n < num_channels; ++n) {
                    col[n] = Unifrom_01(rng);
                }
                for (int r = y * square_size; r < (y + 1) * square_size; ++r) {
                    auto it = data.begin() + b * image_size + r * row_size + x * square_size * num_channels;
                    for (int n = 0; n < square_size; ++n) {
                        for (int chan = 0; chan < num_channels; ++chan, ++it) {
                            *it = static_cast<T>(col[chan] * 255.0f);
                        }
                    }
                }
            }
        }
    }
}

#define CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias) \
    auto node_def = CpuKernelUtils::CpuKernelUtils::CreateNodeDef();          \
    NodeDefBuilder(node_def.get(), "ScaleAndTranslate", "ScaleAndTranslate")  \
        .Input({"images", (data_types)[0], (shapes)[0], (datas)[0]})          \
        .Input({"size", (data_types)[1], (shapes)[1], (datas)[1]})            \
        .Input({"scale", (data_types)[2], (shapes)[2], (datas)[2]})           \
        .Input({"translation", (data_types)[3], (shapes)[3], (datas)[3]})     \
        .Output({"y", (data_types)[4], (shapes)[4], (datas)[4]})              \
        .Attr("kernel_type", std::string(kernel_type_str))                    \
        .Attr("antialias", (bool)(antialias));

namespace {
void RunParameterShapeValidationCase(const vector<int64_t>& scale_shape, const vector<int64_t>& translation_shape)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 2, 3, 1}, {2}, scale_shape, translation_shape, {1, 4, 6, 1}};
    float image_data[6] = {138.568253f, 70.984192f, 108.251984f, 215.417908f, 1.203308f, 31.000126f};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.0f, 1.0f};
    float translate_data[2] = {0.0f, 0.0f};
    float output[24] = {0};
    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "box", true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}
} // namespace

TEST_F(TEST_SCALEANDTRANSLATE_UT, SCALE_RANK_FAIL) { RunParameterShapeValidationCase({1, 2}, {2}); }

TEST_F(TEST_SCALEANDTRANSLATE_UT, SCALE_ELEMENT_COUNT_FAIL) { RunParameterShapeValidationCase({1}, {2}); }

TEST_F(TEST_SCALEANDTRANSLATE_UT, EMPTY_SCALE_FAIL) { RunParameterShapeValidationCase({0}, {2}); }

TEST_F(TEST_SCALEANDTRANSLATE_UT, TRANSLATION_RANK_FAIL) { RunParameterShapeValidationCase({2}, {1, 2}); }

TEST_F(TEST_SCALEANDTRANSLATE_UT, TRANSLATION_ELEMENT_COUNT_FAIL) { RunParameterShapeValidationCase({2}, {1}); }

TEST_F(TEST_SCALEANDTRANSLATE_UT, EMPTY_TRANSLATION_FAIL) { RunParameterShapeValidationCase({2}, {0}); }

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_BOX_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    float image_data[6] = {138.568253f, 70.984192f, 108.251984f, 215.417908f, 1.203308f, 31.000126f};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.000000f, 1.000000f};
    float translate_data[2] = {0.000000f, 0.000000f};

    float output[24] = {0};
    float output_exp[24] = {138.568253f, 70.984192f, 108.251984f, 0.000000f, 0.000000f, 0.000000f,
                            215.417908f, 1.203308f,  31.000126f,  0.000000f, 0.000000f, 0.000000f,
                            0.000000f,   0.000000f,  0.000000f,   0.000000f, 0.000000f, 0.000000f,
                            0.000000f,   0.000000f,  0.000000f,   0.000000f, 0.000000f, 0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "box", true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, TINY_POSITIVE_SCALE_WITH_IN_BOUNDS_TRANSLATION_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 2, 2, 1}, {2}, {2}, {2}, {1, 1, 2, 1}};
    float image_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    int32_t size_data[2] = {1, 2};
    float scale_data[2] = {1.0e-30f, 1.0f};
    float translate_data[2] = {0.5f, 0.0f};
    float output[2] = {-1.0f, -1.0f};
    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "lanczos3", true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    EXPECT_NEAR(output[0], 2.0f, 1.0e-4f);
    EXPECT_NEAR(output[1], 3.0f, 1.0e-4f);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, SUBNORMAL_POSITIVE_SCALE_FAIL)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<vector<int64_t>> shapes = {{1, 2, 2, 1}, {2}, {2}, {2}, {1, 1, 2, 1}};
    float image_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    int32_t size_data[2] = {1, 2};
    float scale_data[2] = {std::numeric_limits<float>::denorm_min(), 1.0f};
    float translate_data[2] = {0.5f, 0.0f};
    float output[2] = {-1.0f, -1.0f};
    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "lanczos3", true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_KEYSCUBIC_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    float image_data[6] = {131.681656f, 145.520233f, 7.260928f, 43.738022f, 174.745636f, 212.643707f};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.200000f, 1.500000f};
    float translate_data[2] = {0.100000f, 0.200000f};

    float output[24] = {0};
    float output_exp[24] = {136.860214f, 145.327011f, 140.872299f, 29.942684f,  -22.839226f, 0.000000f,
                            60.625866f,  100.689201f, 167.188705f, 157.434418f, 149.374008f, 0.000000f,
                            18.317068f,  75.915985f,  181.793854f, 228.190155f, 244.949417f, 0.000000f,
                            0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "keyscubic", true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_MITCHELLCUBIC_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    float image_data[6] = {152.408798f, 172.376663f, 76.316841f, 186.338776f, 147.578552f, 206.299500f};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.000000f, 1.000000f};
    float translate_data[2] = {0.300000f, 0.100000f};

    float output[24] = {0};
    float output_exp[24] = {151.321594f, 169.369431f, 83.013969f,  0.000000f, 0.000000f, 0.000000f,
                            176.018616f, 157.613495f, 166.405380f, 0.000000f, 0.000000f, 0.000000f,
                            0.000000f,   0.000000f,   0.000000f,   0.000000f, 0.000000f, 0.000000f,
                            0.000000f,   0.000000f,   0.000000f,   0.000000f, 0.000000f, 0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "mitchellcubic", true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_GAUSSIAN_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    float image_data[6] = {110.188354f, 44.424892f, 43.590641f, 211.046219f, 149.728668f, 117.135208f};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.500000f, 1.000000f};
    float translate_data[2] = {0.000000f, 0.300000f};

    float output[24] = {0};
    float output_exp[24] = {114.176651f, 70.930641f,  49.267105f,  0.000000f, 0.000000f, 0.000000f,
                            158.128662f, 115.779861f, 85.544647f,  0.000000f, 0.000000f, 0.000000f,
                            202.080704f, 160.629074f, 121.822182f, 0.000000f, 0.000000f, 0.000000f,
                            0.000000f,   0.000000f,   0.000000f,   0.000000f, 0.000000f, 0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "gaussian", true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT32_BOX_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    int32_t image_data[6] = {26, 105, 16, 68, 42, 183};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.000000f, 1.000000f};
    float translate_data[2] = {0.000000f, 0.000000f};

    float output[24] = {0};
    float output_exp[24] = {26.000000f, 105.000000f, 16.000000f,  0.000000f, 0.000000f, 0.000000f,
                            68.000000f, 42.000000f,  183.000000f, 0.000000f, 0.000000f, 0.000000f,
                            0.000000f,  0.000000f,   0.000000f,   0.000000f, 0.000000f, 0.000000f,
                            0.000000f,  0.000000f,   0.000000f,   0.000000f, 0.000000f, 0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "box", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT32_KEYSCUBIC_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    int32_t image_data[6] = {0, 41, 0, 97, 100, 245};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.300000f, 1.200000f};
    float translate_data[2] = {0.200000f, 0.100000f};

    float output[24] = {0};
    float output_exp[24] = {-11.912914f, 25.384739f, 7.328219f,   -28.820704f, 0.000000f, 0.000000f,
                            47.052631f,  59.714275f, 99.323532f,  127.973701f, 0.000000f, 0.000000f,
                            106.018173f, 94.043808f, 191.318848f, 284.768097f, 0.000000f, 0.000000f,
                            0.000000f,   0.000000f,  0.000000f,   0.000000f,   0.000000f, 0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "keyscubic", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT32_MITCHELLCUBIC_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    int32_t image_data[6] = {147, 151, 60, 204, 40, 169};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.000000f, 1.500000f};
    float translate_data[2] = {0.100000f, 0.000000f};

    float output[24] = {0};
    float output_exp[24] = {147.859848f, 151.504288f, 134.076508f, 76.278763f,  55.579456f,  0.000000f,
                            198.232605f, 124.231392f, 70.288658f,  137.711273f, 162.962631f, 0.000000f,
                            0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,
                            0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "mitchellcubic", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT32_GAUSSIAN_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    int32_t image_data[6] = {108, 133, 220, 146, 8, 81};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.500000f, 1.300000f};
    float translate_data[2] = {0.000000f, 0.200000f};

    float output[24] = {0};
    float output_exp[24] = {111.103859f, 117.673843f, 148.115021f, 202.018265f, 0.000000f, 0.000000f,
                            124.510078f, 98.750000f,  94.324577f,  142.182220f, 0.000000f, 0.000000f,
                            137.916306f, 79.826157f,  40.534134f,  82.346161f,  0.000000f, 0.000000f,
                            0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, 0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "gaussian", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT32_LANCZOS1_SUCC)
{
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    int32_t image_data[6] = {65, 252, 245, 13, 229, 26};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.000000f, 1.000000f};
    float translate_data[2] = {0.000000f, 0.000000f};

    float output[24] = {0};
    float output_exp[24] = {65.000000f, 252.000000f, 245.000000f, 0.000000f, 0.000000f, 0.000000f,
                            13.000000f, 229.000000f, 26.000000f,  0.000000f, 0.000000f, 0.000000f,
                            0.000000f,  0.000000f,   0.000000f,   0.000000f, 0.000000f, 0.000000f,
                            0.000000f,  0.000000f,   0.000000f,   0.000000f, 0.000000f, 0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "lanczos1", true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT64_LANCZOS1_SUCC)
{
    vector<DataType> data_types = {DT_INT64, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    int64_t image_data[6] = {66, 36, 238, 184, 221, 148};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.200000f, 1.000000f};
    float translate_data[2] = {0.100000f, 0.100000f};

    float output[24] = {0};
    float output_exp[24] = {66.000000f,  36.365852f,  235.536621f, 0.000000f, 0.000000f, 0.000000f,
                            160.399994f, 183.712189f, 166.219543f, 0.000000f, 0.000000f, 0.000000f,
                            184.000000f, 220.548782f, 148.890259f, 0.000000f, 0.000000f, 0.000000f,
                            0.000000f,   0.000000f,   0.000000f,   0.000000f, 0.000000f, 0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "lanczos1", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT16_LANCZOS3_SUCC)
{
    vector<DataType> data_types = {DT_INT16, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    int16_t image_data[6] = {21, 210, 200, 164, 62, 202};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.000000f, 1.200000f};
    float translate_data[2] = {0.000000f, 0.100000f};

    float output[24] = {0};
    float output_exp[24] = {3.307576f,   142.551056f, 228.000000f, 189.905411f, 0.000000f, 0.000000f,
                            177.385513f, 79.621498f,  128.000000f, 229.652969f, 0.000000f, 0.000000f,
                            0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, 0.000000f,
                            0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f, 0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "lanczos3", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_UINT16_LANCZOS3_SUCC)
{
    vector<DataType> data_types = {DT_UINT16, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    uint16_t image_data[6] = {6, 50, 181, 160, 78, 1};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.000000f, 1.000000f};
    float translate_data[2] = {0.200000f, 0.000000f};

    float output[24] = {0};
    float output_exp[24] = {-16.416292f, 45.924309f, 207.200882f, 0.000000f, 0.000000f, 0.000000f,
                            131.879150f, 72.887115f, 33.868526f,  0.000000f, 0.000000f, 0.000000f,
                            0.000000f,   0.000000f,  0.000000f,   0.000000f, 0.000000f, 0.000000f,
                            0.000000f,   0.000000f,  0.000000f,   0.000000f, 0.000000f, 0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "lanczos3", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_LANCZOS3_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    float image_data[6] = {28.281147f, 206.032852f, 17.464172f, 244.885483f, 249.871445f, 92.243668f};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.500000f, 1.500000f};
    float translate_data[2] = {0.000000f, 0.000000f};

    float output[24] = {0};
    float output_exp[24] = {-22.442028f, 112.591362f, 189.377502f, 38.449009f,  -44.437092f, 0.000000f,
                            123.233429f, 198.194443f, 208.764771f, 78.613449f,  11.813992f,  0.000000f,
                            268.908905f, 283.797546f, 228.152069f, 118.777908f, 68.065079f,  0.000000f,
                            0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f,   0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "lanczos3", true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT8_LANCZOS5_SUCC)
{
    vector<DataType> data_types = {DT_INT8, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    int8_t image_data[6] = {-34, 125, -16, -95, 71, -22};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.000000f, 1.000000f};
    float translate_data[2] = {0.100000f, 0.200000f};

    float output[24] = {0};
    float output_exp[24] = {-53.807579f,  115.260719f, 15.897320f, 0.000000f, 0.000000f, 0.000000f,
                            -111.851692f, 53.257767f,  4.182123f,  0.000000f, 0.000000f, 0.000000f,
                            0.000000f,    0.000000f,   0.000000f,  0.000000f, 0.000000f, 0.000000f,
                            0.000000f,    0.000000f,   0.000000f,  0.000000f, 0.000000f, 0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "lanczos5", true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_UINT8_LANCZOS5_SUCC)
{
    vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    uint8_t image_data[6] = {10, 103, 236, 133, 134, 100};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.300000f, 1.000000f};
    float translate_data[2] = {0.000000f, 0.100000f};

    float output[24] = {0};
    float output_exp[24] = {-2.695577f,  76.776100f,  245.176254f, 0.000000f, 0.000000f, 0.000000f,
                            90.168655f,  118.175423f, 146.308121f, 0.000000f, 0.000000f, 0.000000f,
                            174.934570f, 155.964462f, 56.061882f,  0.000000f, 0.000000f, 0.000000f,
                            0.000000f,   0.000000f,   0.000000f,   0.000000f, 0.000000f, 0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "lanczos5", true);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_DOUBLE_TRIANGLE_SUCC)
{
    vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    double image_data[6] = {199.2488598334, 176.6713594530, 247.2630374143,
                            8.2623258913,   249.9158586850, 76.1476041631};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.000000f, 1.000000f};
    float translate_data[2] = {0.000000f, 0.000000f};

    float output[24] = {0};
    float output_exp[24] = {199.248856f, 176.671356f, 247.263031f, 0.000000f, 0.000000f, 0.000000f,
                            8.262326f,   249.915863f, 76.147606f,  0.000000f, 0.000000f, 0.000000f,
                            0.000000f,   0.000000f,   0.000000f,   0.000000f, 0.000000f, 0.000000f,
                            0.000000f,   0.000000f,   0.000000f,   0.000000f, 0.000000f, 0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "triangle", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT16_TRIANGLE_SUCC)
{
    vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    vector<int64_t> in_shape = {1, 2, 3, 1};
    vector<int64_t> out_shape = {1, 4, 6, 1};
    vector<vector<int64_t>> shapes = {in_shape, {2}, {2}, {2}, out_shape};

    Eigen::half image_data[6] = {Eigen::half(59.3750f), Eigen::half(232.5000f), Eigen::half(13.0156f),
                                 Eigen::half(2.5527f),  Eigen::half(79.8750f),  Eigen::half(77.2500f)};
    int32_t size_data[2] = {4, 6};
    float scale_data[2] = {1.200000f, 1.200000f};
    float translate_data[2] = {0.100000f, 0.100000f};

    float output[24] = {0};
    float output_exp[24] = {59.375000f, 174.791656f, 122.757812f, 13.015625f, 0.000000f, 0.000000f,
                            21.493492f, 94.331161f,  93.294266f,  55.838539f, 0.000000f, 0.000000f,
                            2.552734f,  54.100910f,  78.562500f,  77.250000f, 0.000000f, 0.000000f,
                            0.000000f,  0.000000f,   0.000000f,   0.000000f,  0.000000f, 0.000000f};

    vector<void*> datas = {(void*)image_data, (void*)size_data, (void*)scale_data, (void*)translate_data,
                           (void*)output};
    CREATE_NODEDEF(shapes, data_types, datas, "triangle", false);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);

    bool compare = CompareResult(output, output_exp, 24);
    EXPECT_EQ(compare, true);
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_METHOD_BOX_SUCC)
{
    std::vector<float> data;
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    int64_t kBatchSize_exp = 2;
    int64_t kNumRowSquares_exp = 16;
    int64_t kNumColSquares_exp = 13;
    int64_t kSquareSize_exp = 12;
    int64_t kNumChannels_exp = 3;

    SetCheckerboardImageInput<float>(kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp, kSquareSize_exp,
                                     kNumChannels_exp, data);
    vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                                 kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

    const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
    const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
    vector<int64_t> sizeshape = {2};
    const int outputshape = kBatchSize_exp * kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {
        datashape,
        sizeshape,
        {2},
        {2},
        {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}};

    float* data_arr = new float[data.size()];

    std::copy(data.begin(), data.end(), data_arr);

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "box";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] data_arr;
    delete[] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_METHOD_LANCZOS1_SUCC)
{
    std::vector<float> data;
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    int64_t kBatchSize_exp = 4;
    int64_t kNumRowSquares_exp = 20;
    int64_t kNumColSquares_exp = 15;
    int64_t kSquareSize_exp = 13;
    int64_t kNumChannels_exp = 3;

    SetCheckerboardImageInput<float>(kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp, kSquareSize_exp,
                                     kNumChannels_exp, data);
    vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                                 kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

    const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
    const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
    vector<int64_t> sizeshape = {2};
    const int outputshape = kBatchSize_exp * kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {
        datashape,
        sizeshape,
        {2},
        {2},
        {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}};

    float* data_arr = new float[data.size()];

    std::copy(data.begin(), data.end(), data_arr);

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "lanczos1";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] data_arr;
    delete[] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_METHOD_LANCZOS3_SUCC)
{
    std::vector<float> data;
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    int64_t kBatchSize_exp = 4;
    int64_t kNumRowSquares_exp = 20;
    int64_t kNumColSquares_exp = 15;
    int64_t kSquareSize_exp = 13;
    int64_t kNumChannels_exp = 3;

    SetCheckerboardImageInput<float>(kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp, kSquareSize_exp,
                                     kNumChannels_exp, data);
    vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                                 kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

    const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
    const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
    vector<int64_t> sizeshape = {2};
    const int outputshape = kBatchSize_exp * kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {
        datashape,
        sizeshape,
        {2},
        {2},
        {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}};

    float* data_arr = new float[data.size()];

    std::copy(data.begin(), data.end(), data_arr);

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "lanczos3";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] data_arr;
    delete[] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_METHOD_LANCZOS5_SUCC)
{
    std::vector<float> data;
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    int64_t kBatchSize_exp = 4;
    int64_t kNumRowSquares_exp = 20;
    int64_t kNumColSquares_exp = 15;
    int64_t kSquareSize_exp = 13;
    int64_t kNumChannels_exp = 3;

    SetCheckerboardImageInput<float>(kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp, kSquareSize_exp,
                                     kNumChannels_exp, data);
    vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                                 kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

    const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
    const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
    vector<int64_t> sizeshape = {2};
    const int outputshape = kBatchSize_exp * kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {
        datashape,
        sizeshape,
        {2},
        {2},
        {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}};

    float* data_arr = new float[data.size()];

    std::copy(data.begin(), data.end(), data_arr);

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "lanczos5";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] data_arr;
    delete[] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_METHOD_GAUSSIAN_SUCC)
{
    std::vector<float> data;
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    int64_t kBatchSize_exp = 4;
    int64_t kNumRowSquares_exp = 20;
    int64_t kNumColSquares_exp = 15;
    int64_t kSquareSize_exp = 13;
    int64_t kNumChannels_exp = 3;

    SetCheckerboardImageInput<float>(kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp, kSquareSize_exp,
                                     kNumChannels_exp, data);
    vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                                 kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

    const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
    const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
    vector<int64_t> sizeshape = {2};
    const int outputshape = kBatchSize_exp * kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {
        datashape,
        sizeshape,
        {2},
        {2},
        {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}};

    float* data_arr = new float[data.size()];

    std::copy(data.begin(), data.end(), data_arr);

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "gaussian";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] data_arr;
    delete[] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_METHOD_TRIANGLE_SUCC)
{
    std::vector<float> data;
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    int64_t kBatchSize_exp = 4;
    int64_t kNumRowSquares_exp = 20;
    int64_t kNumColSquares_exp = 15;
    int64_t kSquareSize_exp = 13;
    int64_t kNumChannels_exp = 3;

    SetCheckerboardImageInput<float>(kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp, kSquareSize_exp,
                                     kNumChannels_exp, data);
    vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                                 kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

    const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
    const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
    vector<int64_t> sizeshape = {2};
    const int outputshape = kBatchSize_exp * kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {
        datashape,
        sizeshape,
        {2},
        {2},
        {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}};

    float* data_arr = new float[data.size()];

    std::copy(data.begin(), data.end(), data_arr);

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "triangle";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] data_arr;
    delete[] output;
}
TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_METHOD_KEYSCUBIC_SUCC)
{
    std::vector<float> data;
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    int64_t kBatchSize_exp = 4;
    int64_t kNumRowSquares_exp = 20;
    int64_t kNumColSquares_exp = 15;
    int64_t kSquareSize_exp = 13;
    int64_t kNumChannels_exp = 3;

    SetCheckerboardImageInput<float>(kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp, kSquareSize_exp,
                                     kNumChannels_exp, data);
    vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                                 kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

    const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
    const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
    vector<int64_t> sizeshape = {2};
    const int outputshape = kBatchSize_exp * kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {
        datashape,
        sizeshape,
        {2},
        {2},
        {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}};

    float* data_arr = new float[data.size()];

    std::copy(data.begin(), data.end(), data_arr);

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "keyscubic";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] data_arr;
    delete[] output;
}
TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT_METHOD_MITCHELLCUBIC_SUCC)
{
    std::vector<float> data;
    vector<DataType> data_types = {DT_FLOAT, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    int64_t kBatchSize_exp = 4;
    int64_t kNumRowSquares_exp = 20;
    int64_t kNumColSquares_exp = 15;
    int64_t kSquareSize_exp = 13;
    int64_t kNumChannels_exp = 3;

    SetCheckerboardImageInput<float>(kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp, kSquareSize_exp,
                                     kNumChannels_exp, data);
    vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                                 kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

    const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
    const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
    vector<int64_t> sizeshape = {2};
    const int outputshape = kBatchSize_exp * kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {
        datashape,
        sizeshape,
        {2},
        {2},
        {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}};

    float* data_arr = new float[data.size()];

    std::copy(data.begin(), data.end(), data_arr);

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "mitchellcubic";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] data_arr;
    delete[] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT8_METHOD_BOX_SUCC)
{
    std::vector<int8_t> data;
    vector<DataType> data_types = {DT_INT8, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    int64_t kBatchSize_exp = 2;
    int64_t kNumRowSquares_exp = 16;
    int64_t kNumColSquares_exp = 13;
    int64_t kSquareSize_exp = 12;
    int64_t kNumChannels_exp = 3;

    SetCheckerboardImageInput<int8_t>(kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp, kSquareSize_exp,
                                      kNumChannels_exp, data);
    vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                                 kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

    const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
    const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
    vector<int64_t> sizeshape = {2};
    const int outputshape = kBatchSize_exp * kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {
        datashape,
        sizeshape,
        {2},
        {2},
        {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}};

    int8_t* data_arr = new int8_t[data.size()];

    std::copy(data.begin(), data.end(), data_arr);

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "box";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] data_arr;
    delete[] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT16_METHOD_BOX_SUCC)
{
    std::vector<int16_t> data;
    vector<DataType> data_types = {DT_INT16, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    int64_t kBatchSize_exp = 2;
    int64_t kNumRowSquares_exp = 16;
    int64_t kNumColSquares_exp = 13;
    int64_t kSquareSize_exp = 12;
    int64_t kNumChannels_exp = 3;

    SetCheckerboardImageInput<int16_t>(kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp, kSquareSize_exp,
                                       kNumChannels_exp, data);
    vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                                 kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

    const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
    const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
    vector<int64_t> sizeshape = {2};
    const int outputshape = kBatchSize_exp * kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {
        datashape,
        sizeshape,
        {2},
        {2},
        {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}};

    int16_t* data_arr = new int16_t[data.size()];

    std::copy(data.begin(), data.end(), data_arr);

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "box";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] data_arr;
    delete[] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT32_METHOD_BOX_SUCC)
{
    std::vector<int32_t> data;
    vector<DataType> data_types = {DT_INT32, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    int64_t kBatchSize_exp = 2;
    int64_t kNumRowSquares_exp = 16;
    int64_t kNumColSquares_exp = 13;
    int64_t kSquareSize_exp = 12;
    int64_t kNumChannels_exp = 3;

    SetCheckerboardImageInput<int32_t>(kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp, kSquareSize_exp,
                                       kNumChannels_exp, data);
    vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                                 kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

    const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
    const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
    vector<int64_t> sizeshape = {2};
    const int outputshape = kBatchSize_exp * kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {
        datashape,
        sizeshape,
        {2},
        {2},
        {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}};

    int32_t* data_arr = new int32_t[data.size()];

    std::copy(data.begin(), data.end(), data_arr);

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "box";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] data_arr;
    delete[] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_UINT8_METHOD_BOX_SUCC)
{
    std::vector<uint8_t> data;
    vector<DataType> data_types = {DT_UINT8, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    int64_t kBatchSize_exp = 2;
    int64_t kNumRowSquares_exp = 16;
    int64_t kNumColSquares_exp = 13;
    int64_t kSquareSize_exp = 12;
    int64_t kNumChannels_exp = 3;

    SetCheckerboardImageInput<uint8_t>(kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp, kSquareSize_exp,
                                       kNumChannels_exp, data);
    vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                                 kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

    const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
    const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
    vector<int64_t> sizeshape = {2};
    const int outputshape = kBatchSize_exp * kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {
        datashape,
        sizeshape,
        {2},
        {2},
        {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}};

    uint8_t* data_arr = new uint8_t[data.size()];

    std::copy(data.begin(), data.end(), data_arr);

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "box";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] data_arr;
    delete[] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_UINT16_METHOD_BOX_SUCC)
{
    std::vector<uint16_t> data;
    vector<DataType> data_types = {DT_UINT16, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    int64_t kBatchSize_exp = 2;
    int64_t kNumRowSquares_exp = 16;
    int64_t kNumColSquares_exp = 13;
    int64_t kSquareSize_exp = 12;
    int64_t kNumChannels_exp = 3;

    SetCheckerboardImageInput<uint16_t>(kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp, kSquareSize_exp,
                                        kNumChannels_exp, data);
    vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                                 kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

    const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
    const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
    vector<int64_t> sizeshape = {2};
    const int outputshape = kBatchSize_exp * kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {
        datashape,
        sizeshape,
        {2},
        {2},
        {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}};

    uint16_t* data_arr = new uint16_t[data.size()];

    std::copy(data.begin(), data.end(), data_arr);

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "box";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] data_arr;
    delete[] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_INT64_METHOD_BOX_SUCC)
{
    std::vector<int64_t> data;
    vector<DataType> data_types = {DT_INT64, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    int64_t kBatchSize_exp = 2;
    int64_t kNumRowSquares_exp = 16;
    int64_t kNumColSquares_exp = 13;
    int64_t kSquareSize_exp = 12;
    int64_t kNumChannels_exp = 3;

    SetCheckerboardImageInput<int64_t>(kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp, kSquareSize_exp,
                                       kNumChannels_exp, data);
    vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                                 kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

    const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
    const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
    vector<int64_t> sizeshape = {2};
    const int outputshape = kBatchSize_exp * kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {
        datashape,
        sizeshape,
        {2},
        {2},
        {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}};

    int64_t* data_arr = new int64_t[data.size()];

    std::copy(data.begin(), data.end(), data_arr);

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "box";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] data_arr;
    delete[] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_FLOAT16_METHOD_BOX_SUCC)
{
    std::vector<Eigen::half> data;
    vector<DataType> data_types = {DT_FLOAT16, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    int64_t kBatchSize_exp = 2;
    int64_t kNumRowSquares_exp = 16;
    int64_t kNumColSquares_exp = 13;
    int64_t kSquareSize_exp = 12;
    int64_t kNumChannels_exp = 3;

    SetCheckerboardImageInput<Eigen::half>(kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp, kSquareSize_exp,
                                           kNumChannels_exp, data);
    vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                                 kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

    const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
    const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
    vector<int64_t> sizeshape = {2};
    const int outputshape = kBatchSize_exp * kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {
        datashape,
        sizeshape,
        {2},
        {2},
        {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}};

    Eigen::half* data_arr = new Eigen::half[data.size()];

    std::copy(data.begin(), data.end(), data_arr);

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "box";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] data_arr;
    delete[] output;
}

TEST_F(TEST_SCALEANDTRANSLATE_UT, DATA_TYPE_DOUBLE_METHOD_BOX_SUCC)
{
    std::vector<double> data;
    vector<DataType> data_types = {DT_DOUBLE, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};
    int64_t kBatchSize_exp = 2;
    int64_t kNumRowSquares_exp = 16;
    int64_t kNumColSquares_exp = 13;
    int64_t kSquareSize_exp = 12;
    int64_t kNumChannels_exp = 3;

    SetCheckerboardImageInput<double>(kBatchSize_exp, kNumRowSquares_exp, kNumColSquares_exp, kSquareSize_exp,
                                      kNumChannels_exp, data);
    vector<int64_t> datashape = {kBatchSize_exp, kNumRowSquares_exp * kSquareSize_exp,
                                 kNumColSquares_exp * kSquareSize_exp, kNumChannels_exp};

    const int kOutputImageHeight_exp = kNumRowSquares_exp * kSquareSize_exp;
    const int kOutputImageWidth_exp = kNumColSquares_exp * kSquareSize_exp;

    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {kOutputImageHeight_exp, kOutputImageWidth_exp};
    vector<int64_t> sizeshape = {2};
    const int outputshape = kBatchSize_exp * kOutputImageHeight_exp * kOutputImageWidth_exp * kNumChannels_exp;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {
        datashape,
        sizeshape,
        {2},
        {2},
        {kBatchSize_exp, kOutputImageHeight_exp, kOutputImageWidth_exp, kNumChannels_exp}};

    double* data_arr = new double[data.size()];

    std::copy(data.begin(), data.end(), data_arr);

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "box";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_OK);
    delete[] data_arr;
    delete[] output;
}

// exception instance
TEST_F(TEST_SCALEANDTRANSLATE_UT, INPUT_DATA_TYPE_EXCEPTION)
{
    std::vector<uint64_t> data;
    vector<DataType> data_types = {DT_UINT64, DT_INT32, DT_FLOAT, DT_FLOAT, DT_FLOAT};

    uint64_t data_arr[4] = {(uint64_t)1};
    vector<int64_t> datashape = {1, 2, 2, 1};
    float scale_data[2] = {0.5, 0.5};

    float translate_data[2] = {0.5, 0.5};
    int size_data[2] = {4, 4};
    vector<int64_t> sizeshape = {2};
    const int outputshape = 16;
    float* output = new float[outputshape];

    vector<vector<int64_t>> shapes = {datashape, sizeshape, {2}, {2}, {1, 4, 4, 1}};

    vector<void*> datas = {(void*)data_arr, (void*)size_data, (void*)scale_data, (void*)translate_data, (void*)output};
    std::string kernel_type_str = "box";
    bool antialias = false;
    CREATE_NODEDEF(shapes, data_types, datas, kernel_type_str, antialias);
    RUN_KERNEL(node_def, HOST, KERNEL_STATUS_PARAM_INVALID);
    delete[] output;
}
