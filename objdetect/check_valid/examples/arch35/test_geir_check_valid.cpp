/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_api.h"
#include "array_ops.h"
#include "ge_ir_build.h"
#include "../../op_graph/check_valid_proto.h"

namespace {
constexpr int kSuccess = 0;
constexpr int kFailed = 1;

void ReportGeError(const char* operation, ge::Status status)
{
    const auto message = ge::GEGetErrorMsgV2();
    std::cerr << operation << " failed, status=" << status << ": "
              << (message.GetString() == nullptr ? "" : message.GetString()) << std::endl;
}

int WriteDataToFile(const std::string& file, size_t size, const uint8_t* data)
{
    FILE* fp = std::fopen(file.c_str(), "wb");
    if (fp == nullptr) {
        std::perror(file.c_str());
        return kFailed;
    }
    const size_t written = std::fwrite(data, 1, size, fp);
    const int closeStatus = std::fclose(fp);
    if (written != size || closeStatus != 0) {
        std::cerr << "Write " << file << " failed: wrote " << written << "/" << size << " bytes, fclose=" << closeStatus
                  << std::endl;
        return kFailed;
    }
    return kSuccess;
}

int CreateOppInGraph(std::vector<ge::Tensor>& input, ge::Graph& graph)
{
    const std::vector<float> bboxValues = {0.0f, 0.0f, 2.0f, 2.0f, -1.0f, 0.0f, 2.0f, 2.0f};
    const std::vector<float> metasValues = {2.0f, 3.0f, 1.5f};
    ge::TensorDesc bboxDesc(ge::Shape({2, 4}), ge::FORMAT_ND, ge::DT_FLOAT);
    ge::TensorDesc metasDesc(ge::Shape({3}), ge::FORMAT_ND, ge::DT_FLOAT);
    bboxDesc.SetPlacement(ge::kPlacementHost);
    metasDesc.SetPlacement(ge::kPlacementHost);
    input.emplace_back(bboxDesc, reinterpret_cast<const uint8_t*>(bboxValues.data()),
                       bboxValues.size() * sizeof(float));
    input.emplace_back(metasDesc, reinterpret_cast<const uint8_t*>(metasValues.data()),
                       metasValues.size() * sizeof(float));

    auto bbox = ge::op::Data("bbox_tensor").set_attr_index(0);
    auto metas = ge::op::Data("img_metas").set_attr_index(1);
    bbox.update_input_desc_x(bboxDesc);
    metas.update_input_desc_x(metasDesc);
    auto checkValid = ge::op::CheckValid("check_valid_1");
    checkValid.set_input_bbox_tensor(bbox).set_input_img_metas(metas);
    checkValid.update_output_desc_valid_tensor(ge::TensorDesc(ge::Shape({2, 1}), ge::FORMAT_ND, ge::DT_INT8));
    if (graph.AddOp(bbox) != ge::GRAPH_SUCCESS || graph.AddOp(metas) != ge::GRAPH_SUCCESS) {
        std::cerr << "Add input Data nodes failed" << std::endl;
        return kFailed;
    }
    graph.SetInputs({bbox, metas}).SetOutputs({checkValid});
    return kSuccess;
}

int CheckOutput(const std::vector<ge::Tensor>& output)
{
    // Fixed example mask, confirmed with the supplied golden.py during validation.
    const std::vector<int8_t> expected = {1, 0};
    if (output.size() != 1) {
        std::cerr << "Expected one output, got " << output.size() << std::endl;
        return kFailed;
    }
    const auto& tensor = output[0];
    if (tensor.GetTensorDesc().GetDataType() != ge::DT_INT8 ||
        tensor.GetTensorDesc().GetShape().GetDims() != std::vector<int64_t>({2, 1}) ||
        tensor.GetSize() != expected.size()) {
        std::cerr << "Expected INT8 output with shape (2,1) and 2 bytes" << std::endl;
        return kFailed;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        const int actual = static_cast<int8_t>(tensor.GetData()[i]);
        if (actual != expected[i]) {
            std::cerr << "Output[" << i << "]: expected " << static_cast<int>(expected[i]) << ", got " << actual
                      << std::endl;
            return kFailed;
        }
    }
    std::cout << "Output matches expected mask: [1, 0]" << std::endl;
    return kSuccess;
}

int RunExample()
{
    ge::Graph graph("tc_ge_irrun_test");
    std::vector<ge::Tensor> input;
    if (CreateOppInGraph(input, graph) != kSuccess) {
        return kFailed;
    }

    const std::map<ge::AscendString, ge::AscendString> options;
    ge::Session session(options);
    const uint32_t graphId = 0;
    auto status = session.AddGraph(graphId, graph, options);
    if (status != ge::SUCCESS) {
        ReportGeError("AddGraph", status);
        return kFailed;
    }
    const std::string dumpPath = "./dump";
    status = ge::aclgrphDumpGraph(graph, dumpPath.c_str(), dumpPath.size());
    if (status != ge::GRAPH_SUCCESS) {
        ReportGeError("DumpGraph", status);
        return kFailed;
    }

    std::vector<ge::Tensor> output;
    status = session.RunGraph(graphId, input, output);
    if (status != ge::SUCCESS) {
        ReportGeError("RunGraph", status);
        return kFailed;
    }
    for (size_t i = 0; i < input.size(); ++i) {
        const auto file = "./tc_ge_irrun_test_npu_input_" + std::to_string(i) + ".bin";
        if (WriteDataToFile(file, input[i].GetSize(), input[i].GetData()) != kSuccess) {
            return kFailed;
        }
    }
    for (size_t i = 0; i < output.size(); ++i) {
        const auto file = "./tc_ge_irrun_test_npu_output_" + std::to_string(i) + ".bin";
        if (WriteDataToFile(file, output[i].GetSize(), output[i].GetData()) != kSuccess) {
            return kFailed;
        }
    }
    return CheckOutput(output);
}
} // namespace

int main(int argc, char* argv[])
{
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [device_id]" << std::endl;
        return kFailed;
    }
    const char* configuredDevice = std::getenv("ASCEND_DEVICE_ID");
    const char* deviceId = argc == 2 ? argv[1] : (configuredDevice == nullptr ? "0" : configuredDevice);
    const std::map<ge::AscendString, ge::AscendString> options = {{"ge.exec.deviceId", deviceId},
                                                                  {"ge.graphRunMode", "1"}};
    auto status = ge::GEInitialize(options);
    if (status != ge::SUCCESS) {
        ReportGeError("GEInitialize", status);
        return kFailed;
    }

    int result = kFailed;
    try {
        // RunExample owns the Session, so it is destroyed before GEFinalize on every return path.
        result = RunExample();
    } catch (const std::exception& error) {
        std::cerr << "Example failed: " << error.what() << std::endl;
    }
    status = ge::GEFinalize();
    if (status != ge::SUCCESS) {
        ReportGeError("GEFinalize", status);
        return kFailed;
    }
    if (result == kSuccess) {
        std::cout << "CheckValid GE example passed" << std::endl;
    }
    return result;
}
