/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "array_ops.h"
#include "ge_api.h"
#include "ge_api_types.h"
#include "ge_error_codes.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"

#include "../../op_graph/batch_multi_class_non_max_suppression_proto.h"

#define FAILED -1
#define SUCCESS 0

using namespace ge;
using std::map;
using std::string;
using std::vector;

namespace {
constexpr float kTolerance = 1e-5F;

TensorDesc MakeTensorDesc(const vector<int64_t>& shape, DataType dataType)
{
    TensorDesc desc(Shape(shape), FORMAT_ND, dataType);
    desc.SetPlacement(kPlacementHost);
    desc.SetFormat(FORMAT_ND);
    desc.SetRealDimCnt(shape.size());
    return desc;
}

template <typename T>
Tensor MakeTensor(const vector<int64_t>& shape, vector<T>& data, DataType dataType)
{
    TensorDesc desc = MakeTensorDesc(shape, dataType);
    return Tensor(desc, reinterpret_cast<uint8_t*>(data.data()), data.size() * sizeof(T));
}

template <typename T>
op::Data AddInput(const string& name, int32_t index, const vector<int64_t>& shape, vector<T>& data, DataType dataType,
                  vector<Tensor>& input, Graph& graph)
{
    auto placeholder = op::Data(name.c_str()).set_attr_index(index);
    TensorDesc desc = MakeTensorDesc(shape, dataType);
    placeholder.update_input_desc_x(desc);
    placeholder.update_output_desc_y(desc);
    input.push_back(MakeTensor(shape, data, dataType));
    graph.AddOp(placeholder);
    return placeholder;
}

bool VerifyOutput(const vector<Tensor>& output)
{
    const vector<int64_t> expectedElements = {16, 4, 4, 1};
    const vector<DataType> expectedDataTypes = {DT_FLOAT, DT_FLOAT, DT_FLOAT, DT_INT32};
    if (output.size() != expectedElements.size()) {
        std::cerr << "Unexpected output count: " << output.size() << std::endl;
        return false;
    }
    for (size_t index = 0; index < output.size(); ++index) {
        if (output[index].GetTensorDesc().GetDataType() != expectedDataTypes[index] ||
            output[index].GetTensorDesc().GetShape().GetShapeSize() != expectedElements[index] ||
            output[index].GetData() == nullptr) {
            std::cerr << "Output " << index << " has an unexpected descriptor" << std::endl;
            return false;
        }
    }

    const vector<float> expectedBoxes = {0.00F, 0.00F, 0.50F, 0.50F, 0.05F, 0.05F, 0.55F, 0.55F,
                                         0.55F, 0.55F, 1.00F, 1.00F, 0.00F, 0.00F, 0.00F, 0.00F};
    const vector<float> expectedScores = {0.95F, 0.85F, 0.80F, 0.00F};
    const vector<float> expectedClasses = {0.00F, 1.00F, 0.00F, 0.00F};
    const auto* boxes = reinterpret_cast<const float*>(output[0].GetData());
    const auto* scores = reinterpret_cast<const float*>(output[1].GetData());
    const auto* classes = reinterpret_cast<const float*>(output[2].GetData());
    const auto* num = reinterpret_cast<const int32_t*>(output[3].GetData());
    for (size_t index = 0; index < expectedBoxes.size(); ++index) {
        if (std::fabs(boxes[index] - expectedBoxes[index]) > kTolerance) {
            std::cerr << "nmsed_boxes mismatch at " << index << std::endl;
            return false;
        }
    }
    for (size_t index = 0; index < expectedScores.size(); ++index) {
        if (std::fabs(scores[index] - expectedScores[index]) > kTolerance ||
            std::fabs(classes[index] - expectedClasses[index]) > kTolerance) {
            std::cerr << "nmsed_scores or nmsed_classes mismatch at " << index << std::endl;
            return false;
        }
    }
    if (num[0] != 3) {
        std::cerr << "nmsed_num mismatch: " << num[0] << std::endl;
        return false;
    }
    return true;
}
} // namespace

int CreateOppInGraph(vector<Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs, Graph& graph)
{
    // boxes: [B, N, q, 4], scores: [B, N, C].  q=1 means boxes are shared by all classes.
    static vector<float> boxesData = {0.00F, 0.00F, 0.50F, 0.50F, 0.05F, 0.05F, 0.55F, 0.55F,
                                      0.55F, 0.55F, 1.00F, 1.00F, 0.10F, 0.55F, 0.45F, 1.00F};
    static vector<float> scoresData = {0.95F, 0.20F, 0.90F, 0.85F, 0.80F, 0.10F, 0.99F, 0.98F};
    static vector<float> clipWindowData = {0.00F, 0.00F, 1.00F, 1.00F};
    static vector<int32_t> numValidBoxesData = {3};

    auto boxes = AddInput("boxes", 0, {1, 4, 1, 4}, boxesData, DT_FLOAT, input, graph);
    auto scores = AddInput("scores", 1, {1, 4, 2}, scoresData, DT_FLOAT, input, graph);
    auto clipWindow = AddInput("clip_window", 2, {1, 4}, clipWindowData, DT_FLOAT, input, graph);
    auto numValidBoxes = AddInput("num_valid_boxes", 3, {1}, numValidBoxesData, DT_INT32, input, graph);

    auto nms = op::BatchMultiClassNonMaxSuppression("batch_multi_class_non_max_suppression");
    nms.set_input_boxes(boxes);
    nms.set_input_scores(scores);
    nms.set_input_clip_window(clipWindow);
    nms.set_input_num_valid_boxes(numValidBoxes);
    nms.set_attr_score_threshold(0.20F);
    nms.set_attr_iou_threshold(0.50F);
    nms.set_attr_max_size_per_class(2);
    nms.set_attr_max_total_size(4);
    nms.set_attr_change_coordinate_frame(false);
    nms.set_attr_transpose_box(false);
    nms.update_output_desc_nmsed_boxes(MakeTensorDesc({1, 4, 4}, DT_FLOAT));
    nms.update_output_desc_nmsed_scores(MakeTensorDesc({1, 4}, DT_FLOAT));
    nms.update_output_desc_nmsed_classes(MakeTensorDesc({1, 4}, DT_FLOAT));
    nms.update_output_desc_nmsed_num(MakeTensorDesc({1}, DT_INT32));

    inputs = {boxes, scores, clipWindow, numValidBoxes};
    outputs = {nms};
    return SUCCESS;
}

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (GEInitialize(globalOptions) != SUCCESS) {
        std::cerr << "GEInitialize failed" << std::endl;
        return FAILED;
    }

    Graph graph("batch_multi_class_non_max_suppression_graph");
    vector<Tensor> input;
    vector<Operator> inputs;
    vector<Operator> outputs;
    int ret = CreateOppInGraph(input, inputs, outputs, graph);
    if (ret == SUCCESS) {
        graph.SetInputs(inputs).SetOutputs(outputs);
        map<AscendString, AscendString> buildOptions = {};
        map<AscendString, AscendString> graphOptions = {};
        Session session(buildOptions);
        ret = session.AddGraph(0, graph, graphOptions);
        vector<Tensor> output;
        if (ret == SUCCESS) {
            ret = session.RunGraph(0, input, output);
        }
        if (ret == SUCCESS && !VerifyOutput(output)) {
            ret = FAILED;
        }
    }

    const Status finalizeRet = GEFinalize();
    if (ret != SUCCESS || finalizeRet != SUCCESS) {
        std::cerr << "BatchMultiClassNonMaxSuppression GE IR example failed" << std::endl;
        return FAILED;
    }
    std::cout << "BatchMultiClassNonMaxSuppression GE IR example passed" << std::endl;
    return SUCCESS;
}
