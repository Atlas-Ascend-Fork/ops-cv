/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph.h"
#include "tensor.h"
#include "ge_api.h"
#include "ge_ir_build.h"
#include "ops_proto_legacy.h"
#include "../../op_graph/points_in_polygons_proto.h"

namespace {
constexpr int kSuccess = 0;
constexpr int kFailed = 1;

bool WriteDataToFile(const std::string& path, const ge::Tensor& tensor)
{
    const size_t size = tensor.GetSize();
    const uint8_t* data = tensor.GetData();
    if (size != 0 && data == nullptr) {
        std::cerr << "Missing tensor data for " << path << std::endl;
        return false;
    }
    FILE* file = std::fopen(path.c_str(), "wb");
    if (file == nullptr) {
        std::cerr << "Cannot open " << path << ": " << std::strerror(errno) << std::endl;
        return false;
    }
    const bool written = size == 0 || std::fwrite(data, 1, size, file) == size;
    const int closed = std::fclose(file);
    if (!written || closed != 0) {
        std::cerr << "Cannot write tensor to " << path << std::endl;
        return false;
    }
    return true;
}

bool CheckOutput(const std::vector<ge::Tensor>& outputs)
{
    // (0,0) is inside square 0; (1,0) is on its edge; (2,0) is outside both;
    // (4,4) is inside square 1. Boundary points are outside.
    const std::vector<float> expected = {1, 0, 0, 0, 0, 0, 0, 1};
    if (outputs.size() != 1 || outputs[0].GetTensorDesc().GetDataType() != ge::DT_FLOAT ||
        outputs[0].GetTensorDesc().GetShape().GetDims() != std::vector<int64_t>{4, 2} ||
        outputs[0].GetSize() != expected.size() * sizeof(float) || outputs[0].GetData() == nullptr) {
        std::cerr << "Unexpected output count, dtype, shape or byte size" << std::endl;
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        float value;
        std::memcpy(&value, outputs[0].GetData() + i * sizeof(float), sizeof(float));
        if (value != expected[i]) {
            std::cerr << "Output mismatch at " << i << ": expected " << expected[i] << ", got " << value << std::endl;
            return false;
        }
    }
    return true;
}

int RunExample()
{
    ge::Graph graph("points_in_polygons_example");
    auto points = ge::op::Data("points_data").set_attr_index(0);
    auto polygons = ge::op::Data("polygons_data").set_attr_index(1);
    ge::TensorDesc pointsDesc(ge::Shape({4, 2}), ge::FORMAT_ND, ge::DT_FLOAT);
    ge::TensorDesc polygonsDesc(ge::Shape({8, 2}), ge::FORMAT_ND, ge::DT_FLOAT);
    pointsDesc.SetPlacement(ge::kPlacementHost);
    polygonsDesc.SetPlacement(ge::kPlacementHost);
    points.update_input_desc_x(pointsDesc);
    points.update_output_desc_y(pointsDesc);
    polygons.update_input_desc_x(polygonsDesc);
    polygons.update_output_desc_y(polygonsDesc);

    const std::vector<float> pointValues = {0, 0, 1, 0, 2, 0, 4, 4};
    // Two squares: [-1,1] x [-1,1] and [3,5] x [3,5]. One polygon per column.
    const std::vector<float> polygonValues = {-1, 3, -1, 3, 1, 5, -1, 3, 1, 5, 1, 5, -1, 3, 1, 5};
    // Tensor copies the float32 buffers; the vectors release their own storage.
    std::vector<ge::Tensor> inputs{ge::Tensor(pointsDesc, reinterpret_cast<const uint8_t*>(pointValues.data()),
                                              pointValues.size() * sizeof(float)),
                                   ge::Tensor(polygonsDesc, reinterpret_cast<const uint8_t*>(polygonValues.data()),
                                              polygonValues.size() * sizeof(float))};

    auto membership = ge::op::PointsInPolygons("membership");
    membership.set_input_points(points).set_input_polygons(polygons);
    membership.update_input_desc_points(pointsDesc);
    membership.update_input_desc_polygons(polygonsDesc);
    membership.update_output_desc_output(ge::TensorDesc(ge::Shape({4, 2}), ge::FORMAT_ND, ge::DT_FLOAT));
    graph.AddOp(points);
    graph.AddOp(polygons);
    graph.AddOp(membership);
    graph.SetInputs({points, polygons}).SetOutputs({membership});

    const std::string dumpPath = "./dump";
    if (aclgrphDumpGraph(graph, dumpPath.c_str(), dumpPath.size()) != ge::GRAPH_SUCCESS) {
        std::cerr << "Cannot dump the example graph" << std::endl;
        return kFailed;
    }
    std::map<ge::AscendString, ge::AscendString> buildOptions;
    ge::Session session(buildOptions);
    if (session.AddGraph(0, graph) != ge::GRAPH_SUCCESS) {
        std::cerr << "AddGraph failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        return kFailed;
    }
    std::vector<ge::Tensor> outputs;
    if (session.RunGraph(0, inputs, outputs) != ge::GRAPH_SUCCESS) {
        std::cerr << "RunGraph failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        return kFailed;
    }
    if (!CheckOutput(outputs)) {
        return kFailed;
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!WriteDataToFile("tc_ge_irrun_test_0008_npu_input_" + std::to_string(i) + ".bin", inputs[i])) {
            return kFailed;
        }
    }
    if (!WriteDataToFile("tc_ge_irrun_test_0008_npu_output_0.bin", outputs[0])) {
        return kFailed;
    }
    std::cout << "Precision is ok: all 8 results match the expected geometry" << std::endl;
    return kSuccess;
}
} // namespace

int main()
{
    std::map<ge::AscendString, ge::AscendString> options{{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (ge::GEInitialize(options) != ge::GRAPH_SUCCESS) {
        std::cerr << "GEInitialize failed: " << ge::GEGetErrorMsgV2().GetString() << std::endl;
        return kFailed;
    }
    int result = kFailed;
    try {
        result = RunExample();
    } catch (const std::exception& error) {
        std::cerr << "Example failed: " << error.what() << std::endl;
    }
    // RunExample's Session is destroyed before GE is finalized on every path.
    if (ge::GEFinalize() != ge::GRAPH_SUCCESS) {
        std::cerr << "GEFinalize failed" << std::endl;
        return kFailed;
    }
    return result;
}
