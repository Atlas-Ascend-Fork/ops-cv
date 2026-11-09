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
#include <string.h>
#include <stdint.h>
#include <cstring>
#include <vector>
#include <string>
#include <map>
#include "assert.h"

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"

#include "../../op_graph/rotated_box_encode_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

#define ADD_INPUT(intputIndex, intputName, intputDtype, inputShape, inputValues)                                    \
    vector<int64_t> placeholder##intputIndex##_shape = inputShape;                                                  \
    auto placeholder##intputIndex = op::Data("placeholder" + intputIndex).set_attr_index(0);                        \
    TensorDesc placeholder##intputIndex##_desc = TensorDesc(ge::Shape(placeholder##intputIndex##_shape), FORMAT_ND, \
                                                            intputDtype);                                           \
    placeholder##intputIndex##_desc.SetPlacement(ge::kPlacementHost);                                               \
    placeholder##intputIndex##_desc.SetFormat(FORMAT_ND);                                                           \
    Tensor tensor_placeholder##intputIndex;                                                                         \
    ret = FillData(placeholder##intputIndex##_shape, inputValues, tensor_placeholder##intputIndex,                  \
                   placeholder##intputIndex##_desc, intputDtype);                                                   \
    if (ret != SUCCESS) {                                                                                           \
        printf("%s - ERROR - [XIR]: Generate input data failed\n", GetTime().c_str());                              \
        return FAILED;                                                                                              \
    }                                                                                                               \
    placeholder##intputIndex.update_input_desc_x(placeholder##intputIndex##_desc);                                  \
    input.push_back(tensor_placeholder##intputIndex);                                                               \
    graph.AddOp(placeholder##intputIndex);                                                                          \
    add1.set_input_##intputName(placeholder##intputIndex);                                                          \
    inputs.push_back(placeholder##intputIndex);

#define ADD_OUTPUT(outputIndex, outputName, outputDtype, outputShape)                                       \
    TensorDesc outputName##outputIndex##_desc = TensorDesc(ge::Shape(outputShape), FORMAT_ND, outputDtype); \
    add1.update_output_desc_##outputName(outputName##outputIndex##_desc);

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
    uint32_t oneByte = 1;
    uint32_t twoByte = 2;
    uint32_t fourByte = 4;
    uint32_t eightByte = 8;

    if (dt == ge::DT_FLOAT) {
        return fourByte;
    } else if (dt == ge::DT_FLOAT16) {
        return twoByte;
    } else if (dt == ge::DT_INT16) {
        return twoByte;
    } else if (dt == ge::DT_UINT16) {
        return twoByte;
    } else if (dt == ge::DT_INT32) {
        return fourByte;
    } else if (dt == ge::DT_UINT32) {
        return fourByte;
    } else if (dt == ge::DT_INT64) {
        return eightByte;
    } else if (dt == ge::DT_UINT64) {
        return eightByte;
    } else if (dt == ge::DT_INT8) {
        return oneByte;
    }
    return fourByte;
}

uint16_t FloatToHalf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000U);
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFFU) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFU;
    if (((bits >> 23) & 0xFFU) == 0xFFU) {
        return static_cast<uint16_t>(sign | 0x7C00U | (mantissa ? 0x0200U : 0U));
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    if (exponent <= 0) {
        if (exponent < -10) {
            return sign;
        }
        mantissa = (mantissa | 0x800000U) >> (1 - exponent);
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

// Fill a tensor with a repeating float pattern using the real element type.
// The previous helper wrote int32 values into a DT_FLOAT buffer, which turned
// every input into a subnormal float and made the sample output identically 0.
int32_t FillData(vector<int64_t> shapes, const vector<float>& values, Tensor& input_tensor,
                 TensorDesc& input_tensor_desc, DataType data_type)
{
    if (values.empty()) {
        return FAILED;
    }
    input_tensor_desc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (uint32_t i = 0; i < shapes.size(); i++) {
        if (shapes[i] < 0) {
            return FAILED;
        }
        size *= static_cast<size_t>(shapes[i]);
    }
    const uint32_t elem_bytes = GetDataTypeSize(data_type);
    if (data_type != ge::DT_FLOAT && data_type != ge::DT_FLOAT16) {
        return FAILED;
    }
    vector<uint8_t> buffer(size * elem_bytes, 0);
    for (size_t i = 0; i < size; ++i) {
        const float value = values[i % values.size()];
        if (data_type == ge::DT_FLOAT) {
            reinterpret_cast<float*>(buffer.data())[i] = value;
        } else {
            reinterpret_cast<uint16_t*>(buffer.data())[i] = FloatToHalf(value);
        }
    }
    input_tensor = Tensor(input_tensor_desc, buffer.data(), buffer.size());
    return SUCCESS;
}

int32_t WriteDataToFile(string bin_file, uint64_t data_size, uint8_t* inputData)
{
    if (inputData == nullptr) {
        return FAILED;
    }
    FILE* fp = fopen(bin_file.c_str(), "wb");
    if (fp == nullptr) {
        printf("ERROR: cannot open %s for writing\n", bin_file.c_str());
        return FAILED;
    }
    const size_t written = fwrite(inputData, sizeof(uint8_t), data_size, fp);
    const int close_ret = fclose(fp);
    if (written != data_size || close_ret != 0) {
        printf("ERROR: cannot write %s\n", bin_file.c_str());
        return FAILED;
    }
    return SUCCESS;
}

int CreateOppInGraph(DataType inDtype, std::vector<ge::Tensor>& input, std::vector<Operator>& inputs,
                     std::vector<Operator>& outputs, Graph& graph)
{
    Status ret = SUCCESS;

    auto add1 = op::RotatedBoxEncode("rotated_box_encode_1");

    // anchor_box: (B, 5, N) = (2, 5, 8)
    const int64_t batch = 2;
    const int64_t boxes = 8;
    std::vector<int64_t> boxShape = {batch, 5, boxes};
    // Distinct anchor/gt boxes so the sample exercises a non-zero encoding.
    // A (B, 5, N) tensor is laid out as [b][c][n], so index = (b * 5 + c) * N + n.
    std::vector<float> anchorValues(static_cast<size_t>(batch * 5 * boxes), 0.0f);
    std::vector<float> gtValues(anchorValues.size(), 0.0f);
    auto boxIndex = [boxes](int64_t b, int64_t c, int64_t n) { return static_cast<size_t>((b * 5 + c) * boxes + n); };
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t n = 0; n < boxes; ++n) {
            const float offset = static_cast<float>(b * boxes + n);
            anchorValues[boxIndex(b, 0, n)] = 1.0f + offset;
            anchorValues[boxIndex(b, 1, n)] = 1.0f;
            anchorValues[boxIndex(b, 2, n)] = 3.0f + offset;
            anchorValues[boxIndex(b, 3, n)] = 4.0f;
            anchorValues[boxIndex(b, 4, n)] = 10.0f + offset;
            gtValues[boxIndex(b, 0, n)] = 1.2f + offset;
            gtValues[boxIndex(b, 1, n)] = 1.1f;
            gtValues[boxIndex(b, 2, n)] = 3.4f + offset;
            gtValues[boxIndex(b, 3, n)] = 4.2f;
            gtValues[boxIndex(b, 4, n)] = 12.0f + offset;
        }
    }
    ADD_INPUT(1, anchor_box, inDtype, boxShape, anchorValues);

    // gt_box: (B, 5, N) = (2, 5, 8)
    ADD_INPUT(2, gt_box, inDtype, boxShape, gtValues);

    // weight: ListFloat attribute (length 5)
    std::vector<float> weightVec = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    add1.SetAttr("weight", weightVec);

    // y 输出: (B, 5, N) = (2, 5, 8)
    ADD_OUTPUT(1, y, inDtype, boxShape);

    outputs.push_back(add1);

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

    if (argc > 1) {
        std::cout << argv[1] << std::endl;
    }

    DataType inDtype = DT_FLOAT;

    std::cout << "Input dtype: " << inDtype << std::endl;

    ret = CreateOppInGraph(inDtype, input, inputs, outputs, graph);
    if (ret != SUCCESS) {
        printf("%s - ERROR - [XIR]: Create ir session using build options failed\n", GetTime().c_str());
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
        printf("%s - ERROR - [XIR]: AddGraph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return FAILED;
    }

    printf("%s - INFO - [XIR]: Session add ir compute graph to ir session success\n", GetTime().c_str());
    printf("%s - INFO - [XIR]: dump graph to txt\n", GetTime().c_str());
    std::string file_path = "./dump";
    ret = aclgrphDumpGraph(graph, file_path.c_str(), file_path.length());
    if (ret != SUCCESS) {
        printf("%s - WARNING - [XIR]: dump graph failed, continue\n", GetTime().c_str());
    }
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
        string input_file = "./tc_ge_irrun_test_npu_input_" + std::to_string(i) + ".bin";
        uint8_t* input_data_i = input[i].GetData();
        int64_t input_shape = input[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th input, input shape size =" << input_shape << std::endl;
        uint32_t data_size = input_shape * GetDataTypeSize(input[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile((const char*)input_file.c_str(), data_size, input_data_i) != SUCCESS) {
            printf("%s - ERROR - [XIR]: Write input file failed\n", GetTime().c_str());
            delete session;
            GEFinalize();
            return FAILED;
        }
    }

    int output_num = output.size();
    for (int i = 0; i < output_num; i++) {
        std::cout << "output " << i << " dtype :  " << output[i].GetTensorDesc().GetDataType() << std::endl;
        string output_file = "./tc_ge_irrun_test_npu_output_" + std::to_string(i) + ".bin";
        uint8_t* output_data_i = output[i].GetData();
        int64_t output_shape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        std::cout << "this is " << i << "th output, output shape size =" << output_shape << std::endl;
        uint32_t data_size = output_shape * GetDataTypeSize(output[i].GetTensorDesc().GetDataType());
        if (WriteDataToFile((const char*)output_file.c_str(), data_size, output_data_i) != SUCCESS) {
            printf("%s - ERROR - [XIR]: Write output file failed\n", GetTime().c_str());
            delete session;
            GEFinalize();
            return FAILED;
        }
    }

    ge::AscendString error_msg = ge::GEGetErrorMsgV2();
    std::string error_str(error_msg.GetString());
    std::cout << "Error message: " << error_str << std::endl;
    ge::AscendString warning_msg = ge::GEGetWarningMsgV2();
    std::string warning_str(warning_msg.GetString());
    std::cout << "Warning message: " << warning_str << std::endl;
    printf("%s - INFO - [XIR]: Graph execution completed; outputs were written to ./tc_ge_irrun_test_npu_output_*.bin "
           "(no numeric comparison is performed by this sample)\n",
           GetTime().c_str());
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
