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
#include <cassert>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../../op_graph/image_projective_transform_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

uint32_t GetDataTypeSize(DataType dt)
{
    uint32_t dilation = 1;
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        dilation = fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_BF16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_UINT16) {
        dilation = twoByte;
    } else if (dt == ge::DT_INT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_UINT32) {
        dilation = fourByte;
    } else if (dt == ge::DT_INT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_UINT64) {
        dilation = eightByte;
    } else if (dt == ge::DT_INT8) {
        dilation = oneByte;
    } else if (dt == ge::DT_UINT8) {
        dilation = oneByte;
    }
    return dilation;
}

int32_t GenDataFloat(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, float* value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    float* pData = new (std::nothrow) float[size];
    if (pData == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value[i];
    }
    uint32_t data_len = size * sizeof(float);
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    delete[] pData;
    return SUCCESS;
}

int32_t GenDataInt32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, int32_t* value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    int32_t* pData = new (std::nothrow) int32_t[size];
    if (pData == nullptr) {
        return FAILED;
    }
    for (size_t i = 0; i < size; ++i) {
        *(pData + i) = value[i];
    }
    uint32_t data_len = size * sizeof(int32_t);
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    delete[] pData;
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto add1 = op::ImageProjectiveTransform("image_projective_transform");

    // images: NHWC float32, shape {1, 4, 4, 1}
    std::vector<int64_t> imagesShape = {1, 4, 4, 1};
    // transforms: ND float32, shape {1, 8}, identity projective matrix [1,0,0,0,1,0,0,0]
    std::vector<int64_t> transformsShape = {1, 8};
    // output_shape: ND int32, shape {2}, [height, width] = [4, 4]
    std::vector<int64_t> outputShapeShape = {2};
    // transformed_images: NHWC float32, shape {1, 4, 4, 1}
    std::vector<int64_t> outShape = {1, 4, 4, 1};

    // input 0: images (NHWC)
    auto placeholderImages = op::Data("placeholderImages").set_attr_index(0);
    TensorDesc imagesDesc = TensorDesc(ge::Shape(imagesShape), FORMAT_NHWC, inDtype);
    imagesDesc.SetPlacement(ge::kPlacementHost);
    imagesDesc.SetFormat(FORMAT_NHWC);
    Tensor tensorImages;
    float imagesData[16];
    for (int i = 0; i < 16; i++) {
        imagesData[i] = 1.0f;
    }
    ret = GenDataFloat(imagesShape, tensorImages, imagesDesc, imagesData);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate images data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholderImages.update_input_desc_x(imagesDesc);
    input.push_back(tensorImages);
    graph.AddOp(placeholderImages);
    add1.set_input_images(placeholderImages);
    add1.update_input_desc_images(imagesDesc);
    inputs.push_back(placeholderImages);

    // input 1: transforms (ND)
    auto placeholderTransforms = op::Data("placeholderTransforms").set_attr_index(1);
    TensorDesc transformsDesc = TensorDesc(ge::Shape(transformsShape), FORMAT_ND, DT_FLOAT);
    transformsDesc.SetPlacement(ge::kPlacementHost);
    transformsDesc.SetFormat(FORMAT_ND);
    Tensor tensorTransforms;
    float transformsData[8] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    ret = GenDataFloat(transformsShape, tensorTransforms, transformsDesc, transformsData);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate transforms data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholderTransforms.update_input_desc_x(transformsDesc);
    input.push_back(tensorTransforms);
    graph.AddOp(placeholderTransforms);
    add1.set_input_transforms(placeholderTransforms);
    add1.update_input_desc_transforms(transformsDesc);
    inputs.push_back(placeholderTransforms);

    // input 2: output_shape (ND)
    auto placeholderOutputShape = op::Data("placeholderOutputShape").set_attr_index(2);
    TensorDesc outputShapeDesc = TensorDesc(ge::Shape(outputShapeShape), FORMAT_ND, DT_INT32);
    outputShapeDesc.SetPlacement(ge::kPlacementHost);
    outputShapeDesc.SetFormat(FORMAT_ND);
    Tensor tensorOutputShape;
    int32_t outputShapeData[2] = {4, 4};
    ret = GenDataInt32(outputShapeShape, tensorOutputShape, outputShapeDesc, outputShapeData);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Generate output_shape data failed\n", GetTime().c_str());
        return FAILED;
    }
    placeholderOutputShape.update_input_desc_x(outputShapeDesc);
    input.push_back(tensorOutputShape);
    graph.AddOp(placeholderOutputShape);
    add1.set_input_output_shape(placeholderOutputShape);
    add1.update_input_desc_output_shape(outputShapeDesc);
    inputs.push_back(placeholderOutputShape);

    // attributes
    add1.set_attr_interpolation("BILINEAR");
    add1.set_attr_fill_mode("CONSTANT");

    // output: transformed_images (NHWC)
    TensorDesc transformedImagesDesc = TensorDesc(ge::Shape(outShape), FORMAT_NHWC, inDtype);
    add1.update_output_desc_transformed_images(transformedImagesDesc);

    outputs.push_back(add1);
    // 添加完毕
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;

    printf("%s - INFO - [XIR]: Start to initialize ge using ge global options\n", GetTime().c_str());
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Initialize ge using ge global options success\n", GetTime().c_str());

    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    DataType inDtype = DT_FLOAT;
    std::cout << inDtype << std::endl;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        ge::GEFinalize();
        return FAILED;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    std::map<AscendString, AscendString> build_options = {};
    printf("%s - INFO - [XIR]: Start to create ir session using build options\n", GetTime().c_str());
    ge::Session* session = new Session(build_options);

    if (session == nullptr) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Create ir session using build options success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: Start to add compute graph to ir session\n", GetTime().c_str());

    std::map<AscendString, AscendString> graph_options = {};
    uint32_t graph_id = 0;
    ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    printf("%s - INFO - [XIR]: Start to run ir compute graph\n", GetTime().c_str());
    std::vector<ge::Tensor> output;
    ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Session run ir compute graph success\n", GetTime().c_str());

    int input_num = input.size();
    for (int i = 0; i < input_num; i++) {
        std::cout << "input " << i << " dtype :  " << input[i].GetTensorDesc().GetDataType() << std::endl;
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        float* output_data_i = (float*)output[i].GetData();
        for (int64_t j = 0; j < output_shape; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, output_data_i[j]);
        }
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Start to finalize ir graph session\n", GetTime().c_str());
    delete session;
    ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        printf("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    printf("%s - INFO - [XIR]: Finalize ir graph session success\n", GetTime().c_str());
    return SUCCESS;
}
