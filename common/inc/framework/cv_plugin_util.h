/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CV_COMMON_CV_PLUGIN_UTIL_H
#define CV_COMMON_CV_PLUGIN_UTIL_H

#include <string>
#include <vector>

#include "graph/operator.h"
#include "log/log.h"
#include "register/register.h"

namespace domi {
// ONNX AttributeProto.AttributeType (FLOAT=1, INT=2). Shared by ONNX plugins parsing
// GE "attribute" JSON so callers do not re-define the same magic numbers.
constexpr int ATTR_TYPE_FLOAT = 1;
constexpr int ATTR_TYPE_INT = 2;

inline Status StrToFloat(const std::string& value, float& result)
{
    try {
        size_t pos = 0;
        result = std::stof(value, &pos);
        if (pos != value.size()) {
            return FAILED;
        }
    } catch (...) {
        return FAILED;
    }
    return SUCCESS;
}

// ONNX parser fills Operator attr "input"/"output" with the NodeProto I/O name lists
// before ParseParamsByOperatorFn runs. Prefer those lengths over GetInputsSize() /
// GetOutputsSize(): the latter only count already-registered GE ports and are
// typically 0 during ParseParams. Same role as node->input_size()/output_size()
// under the old ParseParamsFn path.
inline uint32_t ResolveOnnxInputNum(const ge::Operator& op_src)
{
    std::vector<ge::AscendString> onnx_inputs;
    if (op_src.GetAttr("input", onnx_inputs) == ge::GRAPH_SUCCESS && !onnx_inputs.empty()) {
        return static_cast<uint32_t>(onnx_inputs.size());
    }
    return static_cast<uint32_t>(op_src.GetInputsSize());
}

inline uint32_t ResolveOnnxOutputNum(const ge::Operator& op_src)
{
    std::vector<ge::AscendString> onnx_outputs;
    if (op_src.GetAttr("output", onnx_outputs) == ge::GRAPH_SUCCESS && !onnx_outputs.empty()) {
        return static_cast<uint32_t>(onnx_outputs.size());
    }
    return static_cast<uint32_t>(op_src.GetOutputsSize());
}

template <typename T>
inline std::string GetOpName(const T& op)
{
    ge::AscendString op_ascend_name;
    ge::graphStatus ret = op.GetName(op_ascend_name);
    if (ret != ge::GRAPH_SUCCESS) {
        std::string op_name = "None";
        return op_name;
    }
    return op_ascend_name.GetString();
}

inline Status ChangeFormatFromOnnx(ge::Operator& op, const int idx, ge::Format format, bool is_input)
{
    if (is_input) {
        ge::TensorDesc org_tensor = op.GetInputDesc(idx);
        org_tensor.SetOriginFormat(format);
        org_tensor.SetFormat(format);
        auto ret = op.UpdateInputDesc(idx, org_tensor);
        if (ret != ge::GRAPH_SUCCESS) {
            OP_LOGE(GetOpName(op).c_str(), "change input format failed.");
            return FAILED;
        }
    } else {
        ge::TensorDesc org_tensor_y = op.GetOutputDesc(idx);
        org_tensor_y.SetOriginFormat(format);
        org_tensor_y.SetFormat(format);
        auto ret_y = op.UpdateOutputDesc(idx, org_tensor_y);
        if (ret_y != ge::GRAPH_SUCCESS) {
            OP_LOGE(GetOpName(op).c_str(), "change output format failed.");
            return FAILED;
        }
    }
    return SUCCESS;
}

template <typename T>
inline ge::Tensor Vec2Tensor(std::vector<T>& vals, const std::vector<int64_t>& dims, ge::DataType dtype,
                             ge::Format format = ge::FORMAT_ND)
{
    ge::Shape shape(dims);
    ge::TensorDesc desc(shape, format, dtype);
    ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(vals.data()), vals.size() * sizeof(T));
    return tensor;
}

template <typename T>
inline ge::Tensor CreateScalar(T val, ge::DataType dtype, ge::Format format = ge::FORMAT_ND)
{
    std::vector<int64_t> dims_scalar = {};
    ge::Shape shape(dims_scalar);
    ge::TensorDesc desc(shape, format, dtype);
    ge::Tensor tensor(desc, reinterpret_cast<uint8_t*>(&val), sizeof(T));
    return tensor;
}
} // namespace domi

#endif // CV_COMMON_CV_PLUGIN_UTIL_H
