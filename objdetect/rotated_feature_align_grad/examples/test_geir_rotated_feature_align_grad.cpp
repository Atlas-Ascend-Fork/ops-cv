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
#include <fstream>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <map>
#include <random>
#include "cassert"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "experiment_ops.h"

using namespace ge;
using std::map;
using std::string;
using std::vector;

// bboxes 布局：末维 BBOX_COMPONENT_NUM 个分量，依次为 (y, x, w, h, angle)
namespace {
constexpr size_t BBOX_COMPONENT_NUM = 5;
constexpr size_t BBOX_IDX_Y = 0;
constexpr size_t BBOX_IDX_X = 1;
constexpr size_t BBOX_IDX_W = 2;
constexpr size_t BBOX_IDX_H = 3;
constexpr size_t BBOX_IDX_ANGLE = 4;
// 随机 bboxes 数据生成范围
constexpr int32_t COORD_RANGE = 10;     // y/x 坐标上界
constexpr int32_t BBOX_SIZE_MIN = 2;    // w/h 下界
constexpr int32_t BBOX_SIZE_RANGE = 8;  // w/h 范围跨度
constexpr float ANGLE_RANGE = 3.14f;    // 角度上界（近似 π）
constexpr size_t TIME_STR_BUF_LEN = 64; // 时间字符串缓冲区长度
constexpr int64_t PRINT_COUNT_MAX = 10; // 结果打印的最大元素个数
constexpr uint32_t RANDOM_SEED = 42;    // 固定种子，保证 example 可复现

std::mt19937& ExampleRng()
{
    static std::mt19937 rng(RANDOM_SEED);
    return rng;
}
} // namespace

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[TIME_STR_BUF_LEN];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return string(tmp);
}

uint32_t GetDataTypeSize(DataType dt)
{
    if (dt == ge::DT_FLOAT)
        return sizeof(float);
    if (dt == ge::DT_FLOAT16)
        return sizeof(uint16_t);
    if (dt == ge::DT_INT32)
        return sizeof(int32_t);
    if (dt == ge::DT_INT64)
        return sizeof(int64_t);
    if (dt == ge::DT_INT8)
        return sizeof(int8_t);
    return sizeof(float);
}

int32_t GenRandomDataFloat32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    uint32_t data_len = static_cast<uint32_t>(size * sizeof(float));
    float* pData = new (std::nothrow) float[size];
    if (pData == nullptr) {
        return FAILED;
    }
    std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        pData[i] = unitDist(ExampleRng());
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t GenBboxDataFloat32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= static_cast<size_t>(shapes[i]);
    }
    uint32_t data_len = static_cast<uint32_t>(size * sizeof(float));
    float* pData = new (std::nothrow) float[size];
    if (pData == nullptr) {
        return FAILED;
    }
    size_t numBoxes = size / BBOX_COMPONENT_NUM;
    std::uniform_int_distribution<int32_t> coordDist(0, COORD_RANGE - 1);
    std::uniform_int_distribution<int32_t> sizeDist(0, BBOX_SIZE_RANGE - 1);
    std::uniform_real_distribution<float> angleDist(0.0f, ANGLE_RANGE);
    for (size_t i = 0; i < numBoxes; ++i) {
        pData[i * BBOX_COMPONENT_NUM + BBOX_IDX_Y] = static_cast<float>(coordDist(ExampleRng()));
        pData[i * BBOX_COMPONENT_NUM + BBOX_IDX_X] = static_cast<float>(coordDist(ExampleRng()));
        pData[i * BBOX_COMPONENT_NUM + BBOX_IDX_W] = static_cast<float>(sizeDist(ExampleRng()) + BBOX_SIZE_MIN);
        pData[i * BBOX_COMPONENT_NUM + BBOX_IDX_H] = static_cast<float>(sizeDist(ExampleRng()) + BBOX_SIZE_MIN);
        pData[i * BBOX_COMPONENT_NUM + BBOX_IDX_ANGLE] = angleDist(ExampleRng());
    }
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    FILE* fp = fopen(bin_file.c_str(), "w");
    if (fp == nullptr)
        return FAILED;
    fwrite(inputData, sizeof(uint8_t), data_size, fp);
    fclose(fp);
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    auto add1 = op::RotatedFeatureAlignGrad("rfa_grad");

    vector<int64_t> dyShape = {2, 4, 4, 3};
    vector<int64_t> bboxesShape = {2, 4, 4, 5};
    vector<int64_t> dxShape = {2, 4, 4, 3};

    auto placeholder1 = op::Data("placeholder1").set_attr_index(0);
    TensorDesc desc1 = TensorDesc(ge::Shape(dyShape), FORMAT_ND, inDtype);
    desc1.SetPlacement(ge::kPlacementHost);
    desc1.SetFormat(FORMAT_ND);
    Tensor tensor1;
    ret = GenRandomDataFloat32(dyShape, tensor1, desc1);
    if (ret != SUCCESS) {
        printf("Gen dy data failed\n");
        return FAILED;
    }
    placeholder1.update_input_desc_x(desc1);
    placeholder1.update_output_desc_y(desc1);
    input.push_back(tensor1);
    graph.AddOp(placeholder1);
    add1.set_input_dy(placeholder1);
    add1.update_input_desc_dy(desc1);
    inputs.push_back(placeholder1);

    auto placeholder2 = op::Data("placeholder2").set_attr_index(1);
    TensorDesc desc2 = TensorDesc(ge::Shape(bboxesShape), FORMAT_ND, inDtype);
    desc2.SetPlacement(ge::kPlacementHost);
    desc2.SetFormat(FORMAT_ND);
    Tensor tensor2;
    ret = GenBboxDataFloat32(bboxesShape, tensor2, desc2);
    if (ret != SUCCESS) {
        printf("Gen bboxes data failed\n");
        return FAILED;
    }
    placeholder2.update_input_desc_x(desc2);
    placeholder2.update_output_desc_y(desc2);
    input.push_back(tensor2);
    graph.AddOp(placeholder2);
    add1.set_input_bboxes(placeholder2);
    add1.update_input_desc_bboxes(desc2);
    inputs.push_back(placeholder2);

    TensorDesc outDesc = TensorDesc(ge::Shape(dxShape), FORMAT_ND, inDtype);
    add1.update_output_desc_dx(outDesc);

    float spatial_scale = 0.25f;
    int64_t points = 1;
    add1.set_attr_spatial_scale(spatial_scale);
    add1.set_attr_points(points);

    outputs.push_back(add1);
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "rfa_grad_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: GEInitialize failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: GEInitialize success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype = DT_FLOAT;
    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: CreateOppInGraph failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    printf("%s - INFO - [XIR]: Create session\n", GetTime().c_str());
    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create session failed\n", GetTime().c_str());
        GEFinalize();
        return FAILED;
    }

    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, build_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: AddGraph success\n", GetTime().c_str());

    printf("%s - INFO - [XIR]: RunGraph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: RunGraph failed, ret=%d\n", GetTime().c_str(), ret);
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: RunGraph success\n", GetTime().c_str());

    int output_num = static_cast<int>(output.size());
    for (int i = 0; i < output_num; i++) {
        int64_t output_shape_size = output[i].GetTensorDesc().GetShape().GetShapeSize();
        printf("output[%d] shape_size=%ld, dtype=%d\n", i, output_shape_size, output[i].GetTensorDesc().GetDataType());
        uint32_t data_size = static_cast<uint32_t>(output_shape_size) *
                             GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        string output_file = "./rfa_grad_output_" + std::to_string(i) + ".bin";
        WriteDataToFile(output_file, data_size, output[i].GetData());
        float* resultData = reinterpret_cast<float*>(output[i].GetData());
        int64_t printCount = output_shape_size < PRINT_COUNT_MAX ? output_shape_size : PRINT_COUNT_MAX;
        for (int64_t j = 0; j < printCount; j++) {
            printf("output[%ld] = %f\n", j, resultData[j]);
        }
    }

    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - WARN - [XIR]: GEFinalize failed\n", GetTime().c_str());
    }
    printf("%s - INFO - [XIR]: Done\n", GetTime().c_str());
    return SUCCESS;
}
