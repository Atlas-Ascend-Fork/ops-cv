/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
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
#include "ge_ir_build.h"

#include "nn_other.h"
#include "../op_graph/spatial_transformer_proto.h"

#define FAILED -1
#define SUCCESS 0

#include "graph/operator.h"
#include "graph/operator_reg.h"
namespace ge {

REG_OP(Data).INPUT(x, TensorType::ALL()).OUTPUT(y, TensorType::ALL()).ATTR(index, Int, 0).OP_END_FACTORY_REG(Data)
}

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT(inputIndex, inputName, inputDtype, inputShape, val)                                               \
    vector<int64_t> placeholder##inputIndex##_shape = inputShape;                                                   \
    auto placeholder##inputIndex = op::Data("placeholder" + inputIndex).set_attr_index(0);                          \
    TensorDesc placeholder##inputIndex##_desc = TensorDesc(ge::Shape(placeholder##inputIndex##_shape), FORMAT_NCHW, \
                                                           inputDtype);                                             \
    placeholder##inputIndex##_desc.SetPlacement(ge::kPlacementHost);                                                \
    placeholder##inputIndex##_desc.SetFormat(FORMAT_NCHW);                                                          \
    Tensor tensor_placeholder##inputIndex;                                                                          \
    ret = GenOnesDataFloat32(placeholder##inputIndex##_shape, tensor_placeholder##inputIndex,                       \
                             placeholder##inputIndex##_desc, val);                                                  \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##inputIndex.update_input_desc_x(placeholder##inputIndex##_desc);                                    \
    input.push_back(tensor_placeholder##inputIndex);                                                                \
    graph.AddOp(placeholder##inputIndex);                                                                           \
    add1.set_input_##inputName(placeholder##inputIndex);                                                            \
    inputs.push_back(placeholder##inputIndex)

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                         \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_NCHW, outputDtype); \
    add1.update_output_desc_##outputName(outputName##outputIndex##_desc)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

#define ADD_INPUT_ATTR(attrName, attrValue) add1.set_attr_##attrName(attrValue)

string GetTime()
{
    time_t timep;
    time(&timep);
    char tmp[64];
    strftime(tmp, sizeof(tmp), "%Y-%m-%d %H:%M:%S,000", localtime(&timep));
    return tmp;
}

int32_t GenOnesDataFloat32(vector<int64_t> shapes, Tensor& input_tensor, TensorDesc& input_tensor_desc, float* value)
{
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }

    float* pData = new (std::nothrow) float[size];
    for (uint32_t i = 0; i < size; ++i) {
        *(pData + i) = value[i];
    }

    uint32_t data_len = size * sizeof(float);
    input_tensor = Tensor(input_tensor_desc, reinterpret_cast<uint8_t*>(pData), data_len);
    delete[] pData;

    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype1, DataType inDtype2, std::vector<ge::Tensor>& input,
                     std::vector<Operator>& inputs, std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;
    // 自定义代码：添加单算子定义到图中
    auto add1 = op::SpatialTransformer("SpatialTransformer");
    std::vector<std::vector<int64_t>> shapes = {{1, 1, 2, 3}, {2, 3}, {1, 1, 2, 3}};
    float x_data[6] = {-39.0, -47.0, -37.0, 4.0, -70.0, -47.0};
    float theta_data[6] = {-1.0, -2.0, 0.0, 0.0, 0.0, 0.0};

    ADD_INPUT(1, x, inDtype1, shapes[0], x_data);
    ADD_INPUT(2, theta, inDtype2, shapes[1], theta_data);
    ADD_OUTPUT(3, y, inDtype1, shapes[2]);

    // 添加属性
    vector<int64_t> output_size = {-1, -1};
    vector<float> default_theta = {1.0, 0.0, 1.5, 0.0};
    bool align_corners = false;
    vector<int64_t> use_default_theta = {1, 0, 1, 0, 1, 1};

    ADD_INPUT_ATTR(output_size, output_size);
    ADD_INPUT_ATTR(default_theta, default_theta);
    ADD_INPUT_ATTR(align_corners, align_corners);
    ADD_INPUT_ATTR(use_default_theta, use_default_theta);

    outputs.push_back(add1);
    // 添加完毕
    return SUCCESS;
}

bool InitEnv()
{
    std::map<AscendString, AscendString> global_options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(global_options);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - INFO - [XIR]: Initialize ge using ge global options failed\n", GetTime().c_str());
        return false;
    }
    return true;
}

bool CreateAndConfigGraph(Graph& graph, std::vector<ge::Tensor>& input)
{
    std::vector<Operator> inputs{};
    std::vector<Operator> outputs{};

    Status ret = CreateOppInGraph(DT_FLOAT, DT_FLOAT, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
        return false;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }
    return true;
}

bool AddGraphToSession(ge::Session* session, Graph& graph, uint32_t graph_id)
{
    std::map<AscendString, AscendString> graph_options = {};
    Status ret = session->AddGraph(graph_id, graph, graph_options);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - INFO - [XIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        ge::GEFinalize();
        return false;
    }

    return true;
}

bool DumpAndRunGraph(ge::Session* session, Graph& graph, std::vector<ge::Tensor>& input,
                     std::vector<ge::Tensor>& output, uint32_t graph_id)
{
    std::string file_path = "./dump";
    aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());

    Status ret = session->RunGraph(graph_id, input, output);
    if (ret != SUCCESS) {
        LOG_PRINT("%s - INFO - [XIR]: Run graph failed\n", GetTime().c_str());
        ge::AscendString error_msg = ge::GEGetErrorMsgV2();
        std::string error_str(error_msg.GetString());
        ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
        std::string warning_str(warning_msg.GetString());
        std::cout << "Warning message: " << warning_str << std::endl;
        delete session;
        ge::GEFinalize();
        return false;
    }
    return true;
}

void ProcessOutputData(std::vector<ge::Tensor>& output)
{
    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        int64_t shape_size = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "output: " << i << " dtype: " << output[i].GetTensorDesc().GetDataType()
                  << " shape_size: " << shape_size << std::endl;
        float* output_data_i = (float*)output[i].GetData();
        for (int64_t j = 0; j < shape_size; j++) {
            LOG_PRINT("result[%ld] is: %f\n", j, output_data_i[j]);
        }
    }
}

int FinalizeRes()
{
    Status ret = ge::GEFinalize();
    if (ret != SUCCESS) {
        LOG_PRINT("%s - INFO - [XIR]: Finalize ir graph session failed\n", GetTime().c_str());
        return FAILED;
    }
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    LOG_PRINT("=== SpatialTransformer GEIR Test Start ===\n");
    // 初始化环境
    if (!InitEnv()) {
        return FAILED;
    }

    // 创建计算图
    const char* graph_name = "tc_ge_irrun_test";
    Graph graph(graph_name);
    std::vector<ge::Tensor> input;
    if (!CreateAndConfigGraph(graph, input)) {
        LOG_PRINT("ERROR: CreateAndConfigGraph failed\n");
        return FAILED;
    }

    // 创建会话并添加图
    std::map<AscendString, AscendString> build_options = {};
    ge::Session* session = new Session(build_options);
    if (session == nullptr) {
        LOG_PRINT("ERROR: Failed to create session\n");
        ge::GEFinalize();
        return FAILED;
    }

    uint32_t graph_id = 0;
    if (!AddGraphToSession(session, graph, graph_id)) {
        LOG_PRINT("ERROR: AddGraphToSession failed\n");
        return FAILED;
    }

    // 执行图
    std::vector<ge::Tensor> output;
    if (!DumpAndRunGraph(session, graph, input, output, graph_id)) {
        LOG_PRINT("ERROR: DumpAndRunGraph failed\n");
        return FAILED;
    }

    // 处理输入输出数据
    ProcessOutputData(output);

    // 清理资源
    LOG_PRINT("=== Test completed successfully ===\n");
    return FinalizeRes();
}
