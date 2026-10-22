/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "utils/aicpu_read_file.h"
#undef private
#undef protected
#include <cmath>
#include <fstream>
#include <limits>
#include <vector>
#include "Eigen/Core"

using namespace std;
using namespace aicpu;

const std::string ktestcaseFilePath = "../../../../image/crop_and_resize/tests/ut/op_kernel_aicpu/";

class TEST_CropAndResize_UTest : public testing::Test {};

constexpr int BATCH_SIZE = 1;
constexpr int NUM_BOXES = 5;
constexpr int IMAGE_HEIGHT = 256;
constexpr int IMAGE_WIDTH = 256;
constexpr int CHANNELS = 3;
float inputs[BATCH_SIZE][IMAGE_HEIGHT][IMAGE_WIDTH][CHANNELS] = {0};
float boxes[NUM_BOXES][4] = {0};
float y[5][24][24][3] = {0};
float expected_y[5][24][24][3] = {0};

TEST_F(TEST_CropAndResize_UTest, CropAndResizeKernel_Success)
{
    // raw data
    int32_t crop_size[2] = {24, 24};
    int32_t box_index[5] = {0};

    char path[256] = {};
    if (!getcwd(path, 256)) {
        cout << "Get path fail" << endl;
    }
    cout << "path is " << path << endl;

    std::ifstream in_image(ktestcaseFilePath + "data/image_data");
    if (!in_image.is_open()) {
        std::cout << "open image_data errorm, path is " << ktestcaseFilePath + "data/image_data" << std::endl;
        exit(1);
    }

    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < IMAGE_HEIGHT; j++) {
            for (int t = 0; t < IMAGE_WIDTH; t++) {
                for (int s = 0; s < CHANNELS; s++) {
                    float value;
                    in_image >> value;
                    inputs[i][j][t][s] = value;
                }
            }
        }
    }
    in_image.close();

    std::ifstream in_boxes(ktestcaseFilePath + "data/boxes_data");
    if (!in_boxes.is_open()) {
        std::cout << "open boxes_data error" << std::endl;
        exit(1);
    }
    for (int i = 0; i < NUM_BOXES; i++) {
        for (int j = 0; j < 4; j++) {
            float value;
            in_boxes >> value;
            boxes[i][j] = value;
        }
    }
    in_boxes.close();

    std::ifstream in_expected(ktestcaseFilePath + "data/expected_output_data");
    if (!in_expected.is_open()) {
        std::cout << "open expected_output_data error." << std::endl;
        exit(1);
    }

    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 24; j++) {
            for (int t = 0; t < 24; t++) {
                for (int s = 0; s < 3; s++) {
                    float value;
                    in_expected >> value;
                    expected_y[i][j][t][s] = value;
                }
            }
        }
    }
    in_expected.close();

    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("CropAndResize");

    // set attr
    auto methodAttr = CpuKernelUtils::CreateAttrValue();
    methodAttr->SetString("bilinear");
    nodeDef->AddAttrs("method", methodAttr.get());

    auto extrapolation_value = CpuKernelUtils::CreateAttrValue();
    extrapolation_value->SetFloat(0);
    nodeDef->AddAttrs("extrapolation_value", extrapolation_value.get());

    // image
    // set input
    auto inputTensor0 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor0, nullptr);
    auto aicpuShape0 = inputTensor0->GetTensorShape();
    std::vector<int64_t> shapes0 = {BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS};
    aicpuShape0->SetDimSizes(shapes0);
    inputTensor0->SetDataType(DT_FLOAT);
    inputTensor0->SetData(inputs);
    inputTensor0->SetDataSize(sizeof(inputs));
    // boxes
    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);
    auto aicpuShape1 = inputTensor1->GetTensorShape();
    std::vector<int64_t> shapes1 = {NUM_BOXES, 4};
    aicpuShape1->SetDimSizes(shapes1);
    inputTensor1->SetDataType(DT_FLOAT);
    inputTensor1->SetData(boxes);
    inputTensor1->SetDataSize(sizeof(boxes));
    // box_index
    auto inputTensor2 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor2, nullptr);
    auto aicpuShape2 = inputTensor2->GetTensorShape();
    std::vector<int64_t> shapes2 = {NUM_BOXES};
    aicpuShape2->SetDimSizes(shapes2);
    inputTensor2->SetDataType(DT_INT32);
    inputTensor2->SetData(box_index);
    inputTensor2->SetDataSize(sizeof(box_index));
    // crop_size
    auto inputTensor3 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor3, nullptr);
    auto aicpuShape3 = inputTensor3->GetTensorShape();
    std::vector<int64_t> shapes3 = {2};
    aicpuShape3->SetDimSizes(shapes3);
    inputTensor3->SetDataType(DT_INT32);
    inputTensor3->SetData(crop_size);
    inputTensor3->SetDataSize(sizeof(crop_size));

    // set output
    auto outputTensor1 = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor1, nullptr);
    auto aicpuShape4 = outputTensor1->GetTensorShape();
    std::vector<int64_t> shapes4 = {5, 24, 24, 3};
    aicpuShape4->SetDimSizes(shapes4);
    outputTensor1->SetDataType(DT_FLOAT);
    outputTensor1->SetData(y);
    outputTensor1->SetDataSize(sizeof(y));

    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    if (ret != 0U) {
        exit(1);
    }

    float eps = 0.0001;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 24; j++) {
            for (int t = 0; t < 24; t++) {
                for (int s = 0; s < 3; s++) {
                    EXPECT_LT(std::abs(y[i][j][t][s] - expected_y[i][j][t][s]), eps);
                }
            }
        }
    }
}

TEST_F(TEST_CropAndResize_UTest, CropAndResizeKernel_V2_Success)
{
    // raw data
    int32_t crop_size[2] = {24, 24};
    int32_t box_index[5] = {0};

    std::ifstream in_image(ktestcaseFilePath + "data/image_data");
    if (!in_image.is_open()) {
        std::cout << "open image_data error" << std::endl;
        exit(1);
    }

    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < IMAGE_HEIGHT; j++) {
            for (int t = 0; t < IMAGE_WIDTH; t++) {
                for (int s = 0; s < CHANNELS; s++) {
                    float value;
                    in_image >> value;
                    inputs[i][j][t][s] = value;
                }
            }
        }
    }
    in_image.close();

    std::ifstream in_boxes(ktestcaseFilePath + "data/boxes_data");
    if (!in_boxes.is_open()) {
        std::cout << "open boxes_data error" << std::endl;
        exit(1);
    }
    for (int i = 0; i < NUM_BOXES; i++) {
        for (int j = 0; j < 4; j++) {
            float value;
            in_boxes >> value;
            boxes[i][j] = value;
        }
    }
    in_boxes.close();

    std::ifstream in_expected(ktestcaseFilePath + "data/expected_output_data_v2");
    if (!in_expected.is_open()) {
        std::cout << "open expected_output_data error" << std::endl;
        exit(1);
    }
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 24; j++) {
            for (int t = 0; t < 24; t++) {
                for (int s = 0; s < 3; s++) {
                    float value;
                    in_expected >> value;
                    expected_y[i][j][t][s] = value;
                }
            }
        }
    }
    in_expected.close();

    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("CropAndResize");

    // set attr
    auto method = CpuKernelUtils::CreateAttrValue();
    method->SetString("bilinear_v2");
    nodeDef->AddAttrs("method", method.get());

    auto extrapolation_value = CpuKernelUtils::CreateAttrValue();
    extrapolation_value->SetFloat(0);
    nodeDef->AddAttrs("extrapolation_value", extrapolation_value.get());

    // set input
    // image
    auto inputTensor0 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor0, nullptr);
    auto aicpuShape0 = inputTensor0->GetTensorShape();
    std::vector<int64_t> shapes0 = {BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS};
    aicpuShape0->SetDimSizes(shapes0);
    inputTensor0->SetDataType(DT_FLOAT);
    inputTensor0->SetData(inputs);
    inputTensor0->SetDataSize(sizeof(inputs));
    // boxes
    auto inputTensor1 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor1, nullptr);
    auto aicpuShape1 = inputTensor1->GetTensorShape();
    std::vector<int64_t> shapes1 = {NUM_BOXES, 4};
    aicpuShape1->SetDimSizes(shapes1);
    inputTensor1->SetDataType(DT_FLOAT);
    inputTensor1->SetData(boxes);
    inputTensor1->SetDataSize(sizeof(boxes));
    // box_index
    auto inputTensor2 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor2, nullptr);
    auto aicpuShape2 = inputTensor2->GetTensorShape();
    std::vector<int64_t> shapes2 = {NUM_BOXES};
    aicpuShape2->SetDimSizes(shapes2);
    inputTensor2->SetDataType(DT_INT32);
    inputTensor2->SetData(box_index);
    inputTensor2->SetDataSize(sizeof(box_index));
    // crop_size
    auto inputTensor3 = nodeDef->AddInputs();
    EXPECT_NE(inputTensor3, nullptr);
    auto aicpuShape3 = inputTensor3->GetTensorShape();
    std::vector<int64_t> shapes3 = {2};
    aicpuShape3->SetDimSizes(shapes3);
    inputTensor3->SetDataType(DT_INT32);
    inputTensor3->SetData(crop_size);
    inputTensor3->SetDataSize(sizeof(crop_size));

    // set output
    auto outputTensor1 = nodeDef->AddOutputs();
    EXPECT_NE(outputTensor1, nullptr);
    auto aicpuShape4 = outputTensor1->GetTensorShape();
    std::vector<int64_t> shapes4 = {5, 24, 24, 3};
    aicpuShape4->SetDimSizes(shapes4);
    outputTensor1->SetDataType(DT_FLOAT);
    outputTensor1->SetData(y);
    outputTensor1->SetDataSize(sizeof(y));

    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
    if (ret != 0U) {
        exit(1);
    }

    float eps = 0.0001;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 24; j++) {
            for (int t = 0; t < 24; t++) {
                for (int s = 0; s < 3; s++) {
                    EXPECT_LT(std::abs(y[i][j][t][s] - expected_y[i][j][t][s]), eps);
                }
            }
        }
    }
}

TEST_F(TEST_CropAndResize_UTest, CropAndResizeKernel_CropSizeInvalid)
{
    constexpr int BATCH_SIZE = 1;
    constexpr int NUM_BOXES = 1;
    constexpr int IMAGE_HEIGHT = 4;
    constexpr int IMAGE_WIDTH = 4;
    constexpr int CHANNELS = 3;

    float inputs[BATCH_SIZE][IMAGE_HEIGHT][IMAGE_WIDTH][CHANNELS] = {0};
    float boxes[NUM_BOXES][4] = {{0.0f, 0.0f, 1.0f, 1.0f}};
    int32_t box_index[NUM_BOXES] = {0};
    int32_t crop_size[2] = {0, 24};
    float y[NUM_BOXES][24][24][CHANNELS] = {0};

    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("CropAndResize");

    auto method = CpuKernelUtils::CreateAttrValue();
    method->SetString("bilinear");
    nodeDef->AddAttrs("method", method.get());

    auto extrapolation_value = CpuKernelUtils::CreateAttrValue();
    extrapolation_value->SetFloat(0);
    nodeDef->AddAttrs("extrapolation_value", extrapolation_value.get());

    auto inputTensor0 = nodeDef->AddInputs();
    auto aicpuShape0 = inputTensor0->GetTensorShape();
    std::vector<int64_t> shapes0 = {BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS};
    aicpuShape0->SetDimSizes(shapes0);
    inputTensor0->SetDataType(DT_FLOAT);
    inputTensor0->SetData(inputs);
    inputTensor0->SetDataSize(sizeof(inputs));

    auto inputTensor1 = nodeDef->AddInputs();
    auto aicpuShape1 = inputTensor1->GetTensorShape();
    std::vector<int64_t> shapes1 = {NUM_BOXES, 4};
    aicpuShape1->SetDimSizes(shapes1);
    inputTensor1->SetDataType(DT_FLOAT);
    inputTensor1->SetData(boxes);
    inputTensor1->SetDataSize(sizeof(boxes));

    auto inputTensor2 = nodeDef->AddInputs();
    auto aicpuShape2 = inputTensor2->GetTensorShape();
    std::vector<int64_t> shapes2 = {NUM_BOXES};
    aicpuShape2->SetDimSizes(shapes2);
    inputTensor2->SetDataType(DT_INT32);
    inputTensor2->SetData(box_index);
    inputTensor2->SetDataSize(sizeof(box_index));

    auto inputTensor3 = nodeDef->AddInputs();
    auto aicpuShape3 = inputTensor3->GetTensorShape();
    std::vector<int64_t> shapes3 = {2};
    aicpuShape3->SetDimSizes(shapes3);
    inputTensor3->SetDataType(DT_INT32);
    inputTensor3->SetData(crop_size);
    inputTensor3->SetDataSize(sizeof(crop_size));

    auto outputTensor1 = nodeDef->AddOutputs();
    auto aicpuShape4 = outputTensor1->GetTensorShape();
    std::vector<int64_t> shapes4 = {NUM_BOXES, 24, 24, CHANNELS};
    aicpuShape4->SetDimSizes(shapes4);
    outputTensor1->SetDataType(DT_FLOAT);
    outputTensor1->SetData(y);
    outputTensor1->SetDataSize(sizeof(y));

    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CropAndResize_UTest, CropAndResizeKernel_CropSizeDtypeInvalid)
{
    constexpr int BATCH_SIZE = 1;
    constexpr int NUM_BOXES = 1;
    constexpr int IMAGE_HEIGHT = 1;
    constexpr int IMAGE_WIDTH = 1;
    constexpr int CHANNELS = 3;

    float inputs[BATCH_SIZE][IMAGE_HEIGHT][IMAGE_WIDTH][CHANNELS] = {{{{1.0f, 2.0f, 3.0f}}}};
    float boxes[NUM_BOXES][4] = {{0.0f, 0.0f, 1.0f, 1.0f}};
    int32_t box_index[NUM_BOXES] = {0};
    // The smallest positive float has the raw int32 bit pattern 1. Without the
    // dtype check, the old kernel interpreted this as a valid 1x1 crop.
    float crop_size[2] = {std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::denorm_min()};
    float y[NUM_BOXES][1][1][CHANNELS] = {0};

    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("CropAndResize");

    auto method = CpuKernelUtils::CreateAttrValue();
    method->SetString("bilinear");
    nodeDef->AddAttrs("method", method.get());

    auto extrapolation_value = CpuKernelUtils::CreateAttrValue();
    extrapolation_value->SetFloat(0);
    nodeDef->AddAttrs("extrapolation_value", extrapolation_value.get());

    auto inputTensor0 = nodeDef->AddInputs();
    auto aicpuShape0 = inputTensor0->GetTensorShape();
    aicpuShape0->SetDimSizes({BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS});
    inputTensor0->SetDataType(DT_FLOAT);
    inputTensor0->SetData(inputs);
    inputTensor0->SetDataSize(sizeof(inputs));

    auto inputTensor1 = nodeDef->AddInputs();
    auto aicpuShape1 = inputTensor1->GetTensorShape();
    aicpuShape1->SetDimSizes({NUM_BOXES, 4});
    inputTensor1->SetDataType(DT_FLOAT);
    inputTensor1->SetData(boxes);
    inputTensor1->SetDataSize(sizeof(boxes));

    auto inputTensor2 = nodeDef->AddInputs();
    auto aicpuShape2 = inputTensor2->GetTensorShape();
    aicpuShape2->SetDimSizes({NUM_BOXES});
    inputTensor2->SetDataType(DT_INT32);
    inputTensor2->SetData(box_index);
    inputTensor2->SetDataSize(sizeof(box_index));

    auto inputTensor3 = nodeDef->AddInputs();
    auto aicpuShape3 = inputTensor3->GetTensorShape();
    aicpuShape3->SetDimSizes({2});
    inputTensor3->SetDataType(DT_FLOAT);
    inputTensor3->SetData(crop_size);
    inputTensor3->SetDataSize(sizeof(crop_size));

    auto outputTensor1 = nodeDef->AddOutputs();
    auto aicpuShape4 = outputTensor1->GetTensorShape();
    aicpuShape4->SetDimSizes({NUM_BOXES, 1, 1, CHANNELS});
    outputTensor1->SetDataType(DT_FLOAT);
    outputTensor1->SetData(y);
    outputTensor1->SetDataSize(sizeof(y));

    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CropAndResize_UTest, CropAndResizeKernel_BoxIndexInvalid)
{
    constexpr int BATCH_SIZE = 1;
    constexpr int NUM_BOXES = 1;
    constexpr int IMAGE_HEIGHT = 4;
    constexpr int IMAGE_WIDTH = 4;
    constexpr int CHANNELS = 3;

    float inputs[BATCH_SIZE][IMAGE_HEIGHT][IMAGE_WIDTH][CHANNELS] = {0};
    float boxes[NUM_BOXES][4] = {{0.0f, 0.0f, 1.0f, 1.0f}};
    int32_t box_index[NUM_BOXES] = {5};
    int32_t crop_size[2] = {2, 2};
    float y[NUM_BOXES][2][2][CHANNELS] = {0};

    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("CropAndResize");

    auto method = CpuKernelUtils::CreateAttrValue();
    method->SetString("bilinear");
    nodeDef->AddAttrs("method", method.get());

    auto extrapolation_value = CpuKernelUtils::CreateAttrValue();
    extrapolation_value->SetFloat(0);
    nodeDef->AddAttrs("extrapolation_value", extrapolation_value.get());

    auto inputTensor0 = nodeDef->AddInputs();
    auto aicpuShape0 = inputTensor0->GetTensorShape();
    std::vector<int64_t> shapes0 = {BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS};
    aicpuShape0->SetDimSizes(shapes0);
    inputTensor0->SetDataType(DT_FLOAT);
    inputTensor0->SetData(inputs);
    inputTensor0->SetDataSize(sizeof(inputs));

    auto inputTensor1 = nodeDef->AddInputs();
    auto aicpuShape1 = inputTensor1->GetTensorShape();
    std::vector<int64_t> shapes1 = {NUM_BOXES, 4};
    aicpuShape1->SetDimSizes(shapes1);
    inputTensor1->SetDataType(DT_FLOAT);
    inputTensor1->SetData(boxes);
    inputTensor1->SetDataSize(sizeof(boxes));

    auto inputTensor2 = nodeDef->AddInputs();
    auto aicpuShape2 = inputTensor2->GetTensorShape();
    std::vector<int64_t> shapes2 = {NUM_BOXES};
    aicpuShape2->SetDimSizes(shapes2);
    inputTensor2->SetDataType(DT_INT32);
    inputTensor2->SetData(box_index);
    inputTensor2->SetDataSize(sizeof(box_index));

    auto inputTensor3 = nodeDef->AddInputs();
    auto aicpuShape3 = inputTensor3->GetTensorShape();
    std::vector<int64_t> shapes3 = {2};
    aicpuShape3->SetDimSizes(shapes3);
    inputTensor3->SetDataType(DT_INT32);
    inputTensor3->SetData(crop_size);
    inputTensor3->SetDataSize(sizeof(crop_size));

    auto outputTensor1 = nodeDef->AddOutputs();
    auto aicpuShape4 = outputTensor1->GetTensorShape();
    std::vector<int64_t> shapes4 = {NUM_BOXES, 2, 2, CHANNELS};
    aicpuShape4->SetDimSizes(shapes4);
    outputTensor1->SetDataType(DT_FLOAT);
    outputTensor1->SetData(y);
    outputTensor1->SetDataSize(sizeof(y));

    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_PARAM_INVALID);
}

TEST_F(TEST_CropAndResize_UTest, CropAndResizeKernel_Nearest_Success)
{
    constexpr int BATCH_SIZE = 1;
    constexpr int NUM_BOXES = 1;
    constexpr int IMAGE_HEIGHT = 4;
    constexpr int IMAGE_WIDTH = 4;
    constexpr int CHANNELS = 3;

    float inputs[BATCH_SIZE][IMAGE_HEIGHT][IMAGE_WIDTH][CHANNELS] = {
        {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}},
         {{13.0f, 14.0f, 15.0f}, {16.0f, 17.0f, 18.0f}, {19.0f, 20.0f, 21.0f}, {22.0f, 23.0f, 24.0f}},
         {{25.0f, 26.0f, 27.0f}, {28.0f, 29.0f, 30.0f}, {31.0f, 32.0f, 33.0f}, {34.0f, 35.0f, 36.0f}},
         {{37.0f, 38.0f, 39.0f}, {40.0f, 41.0f, 42.0f}, {43.0f, 44.0f, 45.0f}, {46.0f, 47.0f, 48.0f}}}};
    float boxes[NUM_BOXES][4] = {{0.0f, 0.0f, 1.0f, 1.0f}};
    int32_t box_index[NUM_BOXES] = {0};
    int32_t crop_size[2] = {2, 2};
    float y[NUM_BOXES][2][2][CHANNELS] = {0};

    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("CropAndResize");

    auto method = CpuKernelUtils::CreateAttrValue();
    method->SetString("nearest");
    nodeDef->AddAttrs("method", method.get());

    auto extrapolation_value = CpuKernelUtils::CreateAttrValue();
    extrapolation_value->SetFloat(0);
    nodeDef->AddAttrs("extrapolation_value", extrapolation_value.get());

    auto inputTensor0 = nodeDef->AddInputs();
    auto aicpuShape0 = inputTensor0->GetTensorShape();
    std::vector<int64_t> shapes0 = {BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS};
    aicpuShape0->SetDimSizes(shapes0);
    inputTensor0->SetDataType(DT_FLOAT);
    inputTensor0->SetData(inputs);
    inputTensor0->SetDataSize(sizeof(inputs));

    auto inputTensor1 = nodeDef->AddInputs();
    auto aicpuShape1 = inputTensor1->GetTensorShape();
    std::vector<int64_t> shapes1 = {NUM_BOXES, 4};
    aicpuShape1->SetDimSizes(shapes1);
    inputTensor1->SetDataType(DT_FLOAT);
    inputTensor1->SetData(boxes);
    inputTensor1->SetDataSize(sizeof(boxes));

    auto inputTensor2 = nodeDef->AddInputs();
    auto aicpuShape2 = inputTensor2->GetTensorShape();
    std::vector<int64_t> shapes2 = {NUM_BOXES};
    aicpuShape2->SetDimSizes(shapes2);
    inputTensor2->SetDataType(DT_INT32);
    inputTensor2->SetData(box_index);
    inputTensor2->SetDataSize(sizeof(box_index));

    auto inputTensor3 = nodeDef->AddInputs();
    auto aicpuShape3 = inputTensor3->GetTensorShape();
    std::vector<int64_t> shapes3 = {2};
    aicpuShape3->SetDimSizes(shapes3);
    inputTensor3->SetDataType(DT_INT32);
    inputTensor3->SetData(crop_size);
    inputTensor3->SetDataSize(sizeof(crop_size));

    auto outputTensor1 = nodeDef->AddOutputs();
    auto aicpuShape4 = outputTensor1->GetTensorShape();
    std::vector<int64_t> shapes4 = {NUM_BOXES, 2, 2, CHANNELS};
    aicpuShape4->SetDimSizes(shapes4);
    outputTensor1->SetDataType(DT_FLOAT);
    outputTensor1->SetData(y);
    outputTensor1->SetDataSize(sizeof(y));

    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_CropAndResize_UTest, CropAndResizeKernel_ExtrapolationValue_Success)
{
    constexpr int BATCH_SIZE = 1;
    constexpr int NUM_BOXES = 1;
    constexpr int IMAGE_HEIGHT = 4;
    constexpr int IMAGE_WIDTH = 4;
    constexpr int CHANNELS = 3;

    float inputs[BATCH_SIZE][IMAGE_HEIGHT][IMAGE_WIDTH][CHANNELS] = {
        {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}},
         {{13.0f, 14.0f, 15.0f}, {16.0f, 17.0f, 18.0f}, {19.0f, 20.0f, 21.0f}, {22.0f, 23.0f, 24.0f}},
         {{25.0f, 26.0f, 27.0f}, {28.0f, 29.0f, 30.0f}, {31.0f, 32.0f, 33.0f}, {34.0f, 35.0f, 36.0f}},
         {{37.0f, 38.0f, 39.0f}, {40.0f, 41.0f, 42.0f}, {43.0f, 44.0f, 45.0f}, {46.0f, 47.0f, 48.0f}}}};
    float boxes[NUM_BOXES][4] = {{-0.5f, -0.5f, 1.5f, 1.5f}};
    int32_t box_index[NUM_BOXES] = {0};
    int32_t crop_size[2] = {2, 2};
    float y[NUM_BOXES][2][2][CHANNELS] = {0};

    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("CropAndResize");

    auto method = CpuKernelUtils::CreateAttrValue();
    method->SetString("bilinear");
    nodeDef->AddAttrs("method", method.get());

    auto extrapolation_value = CpuKernelUtils::CreateAttrValue();
    extrapolation_value->SetFloat(-1.0f);
    nodeDef->AddAttrs("extrapolation_value", extrapolation_value.get());

    auto inputTensor0 = nodeDef->AddInputs();
    auto aicpuShape0 = inputTensor0->GetTensorShape();
    std::vector<int64_t> shapes0 = {BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS};
    aicpuShape0->SetDimSizes(shapes0);
    inputTensor0->SetDataType(DT_FLOAT);
    inputTensor0->SetData(inputs);
    inputTensor0->SetDataSize(sizeof(inputs));

    auto inputTensor1 = nodeDef->AddInputs();
    auto aicpuShape1 = inputTensor1->GetTensorShape();
    std::vector<int64_t> shapes1 = {NUM_BOXES, 4};
    aicpuShape1->SetDimSizes(shapes1);
    inputTensor1->SetDataType(DT_FLOAT);
    inputTensor1->SetData(boxes);
    inputTensor1->SetDataSize(sizeof(boxes));

    auto inputTensor2 = nodeDef->AddInputs();
    auto aicpuShape2 = inputTensor2->GetTensorShape();
    std::vector<int64_t> shapes2 = {NUM_BOXES};
    aicpuShape2->SetDimSizes(shapes2);
    inputTensor2->SetDataType(DT_INT32);
    inputTensor2->SetData(box_index);
    inputTensor2->SetDataSize(sizeof(box_index));

    auto inputTensor3 = nodeDef->AddInputs();
    auto aicpuShape3 = inputTensor3->GetTensorShape();
    std::vector<int64_t> shapes3 = {2};
    aicpuShape3->SetDimSizes(shapes3);
    inputTensor3->SetDataType(DT_INT32);
    inputTensor3->SetData(crop_size);
    inputTensor3->SetDataSize(sizeof(crop_size));

    auto outputTensor1 = nodeDef->AddOutputs();
    auto aicpuShape4 = outputTensor1->GetTensorShape();
    std::vector<int64_t> shapes4 = {NUM_BOXES, 2, 2, CHANNELS};
    aicpuShape4->SetDimSizes(shapes4);
    outputTensor1->SetDataType(DT_FLOAT);
    outputTensor1->SetData(y);
    outputTensor1->SetDataSize(sizeof(y));

    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_CropAndResize_UTest, CropAndResizeKernel_BilinearV2_SmallBox)
{
    constexpr int BATCH_SIZE = 1;
    constexpr int NUM_BOXES = 1;
    constexpr int IMAGE_HEIGHT = 4;
    constexpr int IMAGE_WIDTH = 4;
    constexpr int CHANNELS = 3;

    float inputs[BATCH_SIZE][IMAGE_HEIGHT][IMAGE_WIDTH][CHANNELS] = {
        {{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}, {7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}},
         {{13.0f, 14.0f, 15.0f}, {16.0f, 17.0f, 18.0f}, {19.0f, 20.0f, 21.0f}, {22.0f, 23.0f, 24.0f}},
         {{25.0f, 26.0f, 27.0f}, {28.0f, 29.0f, 30.0f}, {31.0f, 32.0f, 33.0f}, {34.0f, 35.0f, 36.0f}},
         {{37.0f, 38.0f, 39.0f}, {40.0f, 41.0f, 42.0f}, {43.0f, 44.0f, 45.0f}, {46.0f, 47.0f, 48.0f}}}};
    float boxes[NUM_BOXES][4] = {{0.25f, 0.25f, 0.5f, 0.5f}};
    int32_t box_index[NUM_BOXES] = {0};
    int32_t crop_size[2] = {2, 2};
    float y[NUM_BOXES][2][2][CHANNELS] = {0};

    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("CropAndResize");

    auto method = CpuKernelUtils::CreateAttrValue();
    method->SetString("bilinear_v2");
    nodeDef->AddAttrs("method", method.get());

    auto extrapolation_value = CpuKernelUtils::CreateAttrValue();
    extrapolation_value->SetFloat(0);
    nodeDef->AddAttrs("extrapolation_value", extrapolation_value.get());

    auto inputTensor0 = nodeDef->AddInputs();
    auto aicpuShape0 = inputTensor0->GetTensorShape();
    std::vector<int64_t> shapes0 = {BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS};
    aicpuShape0->SetDimSizes(shapes0);
    inputTensor0->SetDataType(DT_FLOAT);
    inputTensor0->SetData(inputs);
    inputTensor0->SetDataSize(sizeof(inputs));

    auto inputTensor1 = nodeDef->AddInputs();
    auto aicpuShape1 = inputTensor1->GetTensorShape();
    std::vector<int64_t> shapes1 = {NUM_BOXES, 4};
    aicpuShape1->SetDimSizes(shapes1);
    inputTensor1->SetDataType(DT_FLOAT);
    inputTensor1->SetData(boxes);
    inputTensor1->SetDataSize(sizeof(boxes));

    auto inputTensor2 = nodeDef->AddInputs();
    auto aicpuShape2 = inputTensor2->GetTensorShape();
    std::vector<int64_t> shapes2 = {NUM_BOXES};
    aicpuShape2->SetDimSizes(shapes2);
    inputTensor2->SetDataType(DT_INT32);
    inputTensor2->SetData(box_index);
    inputTensor2->SetDataSize(sizeof(box_index));

    auto inputTensor3 = nodeDef->AddInputs();
    auto aicpuShape3 = inputTensor3->GetTensorShape();
    std::vector<int64_t> shapes3 = {2};
    aicpuShape3->SetDimSizes(shapes3);
    inputTensor3->SetDataType(DT_INT32);
    inputTensor3->SetData(crop_size);
    inputTensor3->SetDataSize(sizeof(crop_size));

    auto outputTensor1 = nodeDef->AddOutputs();
    auto aicpuShape4 = outputTensor1->GetTensorShape();
    std::vector<int64_t> shapes4 = {NUM_BOXES, 2, 2, CHANNELS};
    aicpuShape4->SetDimSizes(shapes4);
    outputTensor1->SetDataType(DT_FLOAT);
    outputTensor1->SetData(y);
    outputTensor1->SetDataSize(sizeof(y));

    CpuKernelContext ctx(DEVICE);
    EXPECT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    uint32_t ret = CpuKernelRegister::Instance().RunCpuKernel(ctx);
    EXPECT_EQ(ret, KERNEL_STATUS_OK);
}

TEST_F(TEST_CropAndResize_UTest, CropAndResizeKernel_BilinearV2_FullFrame)
{
    float inputs[1][4][4][1] = {{{{1.0f}, {2.0f}, {3.0f}, {4.0f}},
                                 {{5.0f}, {6.0f}, {7.0f}, {8.0f}},
                                 {{9.0f}, {10.0f}, {11.0f}, {12.0f}},
                                 {{13.0f}, {14.0f}, {15.0f}, {16.0f}}}};
    float boxes[1][4] = {{0.0f, 0.0f, 1.0f, 1.0f}};
    int32_t box_index[1] = {0};
    int32_t crop_size[2] = {2, 2};
    float output[1][2][2][1] = {};

    auto nodeDef = CpuKernelUtils::CreateNodeDef();
    nodeDef->SetOpType("CropAndResize");

    auto method = CpuKernelUtils::CreateAttrValue();
    method->SetString("bilinear_v2");
    nodeDef->AddAttrs("method", method.get());
    auto extrapolation_value = CpuKernelUtils::CreateAttrValue();
    extrapolation_value->SetFloat(-1.0f);
    nodeDef->AddAttrs("extrapolation_value", extrapolation_value.get());

    auto input = nodeDef->AddInputs();
    input->GetTensorShape()->SetDimSizes({1, 4, 4, 1});
    input->SetDataType(DT_FLOAT);
    input->SetData(inputs);
    input->SetDataSize(sizeof(inputs));

    auto boxesInput = nodeDef->AddInputs();
    boxesInput->GetTensorShape()->SetDimSizes({1, 4});
    boxesInput->SetDataType(DT_FLOAT);
    boxesInput->SetData(boxes);
    boxesInput->SetDataSize(sizeof(boxes));

    auto boxIndexInput = nodeDef->AddInputs();
    boxIndexInput->GetTensorShape()->SetDimSizes({1});
    boxIndexInput->SetDataType(DT_INT32);
    boxIndexInput->SetData(box_index);
    boxIndexInput->SetDataSize(sizeof(box_index));

    auto cropSizeInput = nodeDef->AddInputs();
    cropSizeInput->GetTensorShape()->SetDimSizes({2});
    cropSizeInput->SetDataType(DT_INT32);
    cropSizeInput->SetData(crop_size);
    cropSizeInput->SetDataSize(sizeof(crop_size));

    auto outputTensor = nodeDef->AddOutputs();
    outputTensor->GetTensorShape()->SetDimSizes({1, 2, 2, 1});
    outputTensor->SetDataType(DT_FLOAT);
    outputTensor->SetData(output);
    outputTensor->SetDataSize(sizeof(output));

    CpuKernelContext ctx(DEVICE);
    ASSERT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
    EXPECT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);

    EXPECT_FLOAT_EQ(output[0][0][0][0], 3.5f);
    EXPECT_FLOAT_EQ(output[0][0][1][0], 5.5f);
    EXPECT_FLOAT_EQ(output[0][1][0][0], 11.5f);
    EXPECT_FLOAT_EQ(output[0][1][1][0], 13.5f);
}

TEST_F(TEST_CropAndResize_UTest, CropAndResizeKernel_BilinearV2_IntegerBoundaryExtrapolation)
{
    constexpr int32_t kCropSize = 4096;
    constexpr int64_t kImageLongSide = 16385;
    constexpr int32_t kBoundaryOutput = 1024;
    const float beyondOne = std::nextafter(1.0f, 2.0f);

    for (bool vertical : {true, false}) {
        int64_t imageHeight = vertical ? kImageLongSide : 1;
        int64_t imageWidth = vertical ? 1 : kImageLongSide;
        std::vector<float> image(static_cast<size_t>(imageHeight * imageWidth), 7.0f);
        std::vector<float> output(kCropSize, 0.0f);
        float boxes[1][4] = {};
        if (vertical) {
            boxes[0][0] = 1.0f;
            boxes[0][2] = beyondOne;
        } else {
            boxes[0][1] = 1.0f;
            boxes[0][3] = beyondOne;
        }
        int32_t box_index[1] = {0};
        int32_t crop_size[2] = {vertical ? kCropSize : 1, vertical ? 1 : kCropSize};

        auto nodeDef = CpuKernelUtils::CreateNodeDef();
        nodeDef->SetOpType("CropAndResize");

        auto method = CpuKernelUtils::CreateAttrValue();
        method->SetString("bilinear_v2");
        nodeDef->AddAttrs("method", method.get());
        auto extrapolationValue = CpuKernelUtils::CreateAttrValue();
        extrapolationValue->SetFloat(-1.0f);
        nodeDef->AddAttrs("extrapolation_value", extrapolationValue.get());

        auto input = nodeDef->AddInputs();
        input->GetTensorShape()->SetDimSizes({1, imageHeight, imageWidth, 1});
        input->SetDataType(DT_FLOAT);
        input->SetData(image.data());
        input->SetDataSize(image.size() * sizeof(float));

        auto boxesInput = nodeDef->AddInputs();
        boxesInput->GetTensorShape()->SetDimSizes({1, 4});
        boxesInput->SetDataType(DT_FLOAT);
        boxesInput->SetData(boxes);
        boxesInput->SetDataSize(sizeof(boxes));

        auto boxIndexInput = nodeDef->AddInputs();
        boxIndexInput->GetTensorShape()->SetDimSizes({1});
        boxIndexInput->SetDataType(DT_INT32);
        boxIndexInput->SetData(box_index);
        boxIndexInput->SetDataSize(sizeof(box_index));

        auto cropSizeInput = nodeDef->AddInputs();
        cropSizeInput->GetTensorShape()->SetDimSizes({2});
        cropSizeInput->SetDataType(DT_INT32);
        cropSizeInput->SetData(crop_size);
        cropSizeInput->SetDataSize(sizeof(crop_size));

        auto outputTensor = nodeDef->AddOutputs();
        outputTensor->GetTensorShape()->SetDimSizes({1, crop_size[0], crop_size[1], 1});
        outputTensor->SetDataType(DT_FLOAT);
        outputTensor->SetData(output.data());
        outputTensor->SetDataSize(output.size() * sizeof(float));

        CpuKernelContext ctx(DEVICE);
        ASSERT_EQ(ctx.Init(nodeDef.get()), KERNEL_STATUS_OK);
        ASSERT_EQ(CpuKernelRegister::Instance().RunCpuKernel(ctx), KERNEL_STATUS_OK);
        EXPECT_FLOAT_EQ(output[kBoundaryOutput], -1.0f);
    }
}
