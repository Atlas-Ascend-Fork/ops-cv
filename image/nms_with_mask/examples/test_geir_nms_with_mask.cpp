/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <chrono>
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdint.h>
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
#include "../op_graph/nms_with_mask_proto.h"

constexpr int32_t kFailed = -1;
constexpr int32_t kSuccess = 0;

using namespace ge;
using std::map;
using std::string;
using std::vector;

static string GetTime()
{
    auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    struct tm tmPt;
    localtime_r(&tt, &tmPt);
    char buf[64] = {0};
    (void)snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d", tmPt.tm_year + 1900, tmPt.tm_mon + 1,
                   tmPt.tm_mday, tmPt.tm_hour, tmPt.tm_min, tmPt.tm_sec);
    return string(buf);
}

static int32_t GenDataFloat(vector<int64_t> shapes, Tensor& inputTensor, TensorDesc& inputTensorDesc)
{
    inputTensorDesc.SetRealDimCnt(shapes.size());
    size_t size = 1;
    for (size_t i = 0; i < shapes.size(); i++) {
        size *= shapes[i];
    }
    float* pData = new (std::nothrow) float[size];
    if (pData == nullptr) {
        return kFailed;
    }
    // 输入格式 [N, 5]：每行 [x1, y1, x2, y2, score]
    // 偶数行大框互相重叠，奇数行小框在大框内部，IoU > threshold 的会被 NMS 抑制
    constexpr int64_t NUM_COLS = 5;
    for (size_t row = 0; row < shapes[0]; row++) {
        size_t base = row * NUM_COLS;
        if (row % 2 == 0) {
            pData[base + 0] = 0.0f;
            pData[base + 1] = 0.0f;
            pData[base + 2] = 10.0f;
            pData[base + 3] = 10.0f;
        } else {
            pData[base + 0] = 1.0f;
            pData[base + 1] = 1.0f;
            pData[base + 2] = 9.0f;
            pData[base + 3] = 9.0f;
        }
        pData[base + 4] = static_cast<float>(shapes[0] - row) / static_cast<float>(shapes[0]);
    }
    uint32_t dataLen = size * sizeof(float);
    inputTensor = Tensor(inputTensorDesc, (uint8_t*)pData, dataLen);
    delete[] pData;
    return kSuccess;
}

static int32_t WriteDataToFile(string binFile, uint64_t dataSize, uint8_t* inputData)
{
    FILE* fp = fopen(binFile.c_str(), "w");
    if (fp == nullptr) {
        return kFailed;
    }
    fwrite(inputData, sizeof(uint8_t), dataSize, fp);
    fclose(fp);
    return kSuccess;
}

static int CreateOppInGraph(vector<ge::Tensor>& input, vector<Operator>& inputs, vector<Operator>& outputs,
                            Graph& graph)
{
    Status ret = kSuccess;
    auto nmsWithMaskOp = op::NMSWithMask("nms_with_mask");
    nmsWithMaskOp.set_attr_iou_threshold(0.5f);

    vector<int64_t> boxScoresShape = {8, 5};

    auto placeholderBoxScores = op::Data("placeholderBoxScores").set_attr_index(0);
    TensorDesc boxScoresDesc = TensorDesc(ge::Shape(boxScoresShape), FORMAT_ND, DT_FLOAT);
    boxScoresDesc.SetPlacement(ge::kPlacementHost);
    boxScoresDesc.SetFormat(FORMAT_ND);
    Tensor tensorBoxScores;
    ret = GenDataFloat(boxScoresShape, tensorBoxScores, boxScoresDesc);
    if (ret != kSuccess) {
        printf("%s - ERROR - [GEIR]: Generate box_scores data failed\n", GetTime().c_str());
        return kFailed;
    }
    input.push_back(tensorBoxScores);
    placeholderBoxScores.update_input_desc_x(boxScoresDesc);
    nmsWithMaskOp.set_input_box_scores(placeholderBoxScores);
    nmsWithMaskOp.update_input_desc_box_scores(boxScoresDesc);

    TensorDesc selectedBoxesDesc = TensorDesc(ge::Shape(vector<int64_t>{8, 5}), FORMAT_ND, DT_FLOAT);
    nmsWithMaskOp.update_output_desc_selected_boxes(selectedBoxesDesc);
    vector<int64_t> selectedNumShape = {8};
    TensorDesc selectedIdxDesc = TensorDesc(ge::Shape(selectedNumShape), FORMAT_ND, DT_INT32);
    nmsWithMaskOp.update_output_desc_selected_idx(selectedIdxDesc);
    TensorDesc selectedMaskDesc = TensorDesc(ge::Shape(selectedNumShape), FORMAT_ND, DT_UINT8);
    nmsWithMaskOp.update_output_desc_selected_mask(selectedMaskDesc);

    inputs.push_back(placeholderBoxScores);
    outputs.push_back(nmsWithMaskOp);
    return kSuccess;
}

int main(int argc, char* argv[])
{
    const char* graphName = "tc_ge_irrun_test";
    Graph graph(graphName);
    vector<ge::Tensor> input;

    printf("%s - INFO - [GEIR]: Start to initialize ge\n", GetTime().c_str());
    map<AscendString, AscendString> globalOptions = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    Status ret = ge::GEInitialize(globalOptions);
    if (ret != kSuccess) {
        printf("%s - ERROR - [GEIR]: Initialize ge failed\n", GetTime().c_str());
        return kFailed;
    }

    vector<Operator> inputs{};
    vector<Operator> outputs{};

    ret = CreateOppInGraph(input, inputs, outputs, graph);
    if (ret != kSuccess) {
        printf("%s - ERROR - [GEIR]: Create op in graph failed\n", GetTime().c_str());
        GEFinalize();
        return kFailed;
    }

    if (!inputs.empty() && !outputs.empty()) {
        graph.SetInputs(inputs).SetOutputs(outputs);
    }

    map<AscendString, AscendString> buildOptions = {};
    ge::Session* session = new Session(buildOptions);
    if (session == nullptr) {
        printf("%s - ERROR - [GEIR]: Create session failed\n", GetTime().c_str());
        GEFinalize();
        return kFailed;
    }

    uint32_t graphId = 0;
    ret = session->AddGraph(graphId, graph, buildOptions);
    if (ret != kSuccess) {
        printf("%s - ERROR - [GEIR]: Add graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return kFailed;
    }

    printf("%s - INFO - [GEIR]: Start to run graph\n", GetTime().c_str());
    vector<ge::Tensor> output;
    ret = session->RunGraph(graphId, input, output);
    if (ret != kSuccess) {
        printf("%s - ERROR - [GEIR]: Run graph failed\n", GetTime().c_str());
        delete session;
        GEFinalize();
        return kFailed;
    }
    printf("%s - INFO - [GEIR]: Run graph success\n", GetTime().c_str());

    for (size_t i = 0; i < output.size(); i++) {
        int64_t outputShape = output[i].GetTensorDesc().GetShape().GetShapeSize();
        DataType outputDtype = output[i].GetTensorDesc().GetDataType();
        printf("%s - INFO - [GEIR]: output[%zu] shape size = %ld, dtype = %d\n", GetTime().c_str(), i, outputShape,
               static_cast<int32_t>(outputDtype));
        uint32_t dataSize = outputShape * sizeof(float);
        if (outputDtype == DT_INT32) {
            dataSize = outputShape * sizeof(int32_t);
        } else if (outputDtype == DT_UINT8) {
            dataSize = outputShape * sizeof(uint8_t);
        }
        string outputFile = "./tc_ge_irrun_test_npu_output_" + std::to_string(i) + ".bin";
        WriteDataToFile(outputFile, dataSize, output[i].GetData());
        if (outputDtype == DT_FLOAT) {
            float* resultData = (float*)output[i].GetData();
            for (int64_t j = 0; j < outputShape; j++) {
                printf("result[%ld] = %f\n", j, resultData[j]);
            }
        } else if (outputDtype == DT_INT32) {
            int32_t* resultData = (int32_t*)output[i].GetData();
            for (int64_t j = 0; j < outputShape; j++) {
                printf("result[%ld] = %d\n", j, resultData[j]);
            }
        } else {
            uint8_t* resultData = (uint8_t*)output[i].GetData();
            for (int64_t j = 0; j < outputShape; j++) {
                printf("result[%ld] = %u\n", j, resultData[j]);
            }
        }
    }

    delete session;
    GEFinalize();
    printf("%s - INFO - [GEIR]: Done\n", GetTime().c_str());
    return kSuccess;
}
