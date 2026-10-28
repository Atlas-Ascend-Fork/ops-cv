/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file crop_and_resize_def.cpp
 * \brief Operator definition of crop_and_resize
 *
 * 4 dtype combinations:
 *   #0: x=FP32, boxes=FP32, y=FP32
 *   #1: x=FP16, boxes=FP32, y=FP32
 *   #2: x=FP16, boxes=FP16, y=FP16
 *   #3: x=FP32, boxes=FP16, y=FP16
 * box_index/crop_size=INT32。所有候选 format=ND，NHWC/NCHW 输入均由 ND 候选
 * 承接，format 由 infershape/tiling/check 按实际 desc format 分支解析。
 *
 * 构建注意：本文件注释会被构建脚本按行抓取拼接进 shell 命令，
 * 注释中勿出现注册宏名与未闭合括号（否则 CMake configure 报语法错误）。
 */

#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "crop_and_resize_constraints.h"

#include <algorithm>
#include <string>
#include <vector>

namespace ops {

struct CropAndResizeCheckInput {
    ge::DataType xDtype = ge::DT_UNDEFINED;
    ge::DataType boxesDtype = ge::DT_UNDEFINED;
    ge::DataType boxIndexDtype = ge::DT_UNDEFINED;
    // x 实际 format（GetPrimaryFormat 归一化）：NCHW → dims=(N,C,H,W)；ND/NHWC → dims=(N,H,W,C)
    ge::Format xFormat = ge::FORMAT_ND;
    std::vector<int64_t> xDims;     // 空 = x 为 unknown rank([-2])，由读取处归一化
    std::vector<int64_t> boxesDims; // 空 = boxes 为 unknown rank([-2])，由读取处归一化
    // box_index 不携带 dims（无 shape 约束），仅保留 unknown rank 标志
    bool boxIndexUnknownRank = false;
    std::string method; // 空 = 属性缺省或读取失败，均按默认 bilinear 处理
    bool cropSizeHasValue = false;
    int32_t cropH = 0;
    int32_t cropW = 0;
};

static bool CheckCropAndResizeConstraints(const CropAndResizeCheckInput& in, std::string& rejectDetail)
{
    auto rejected = [&rejectDetail](const std::string& detail) {
        rejectDetail = detail;
        return false;
    };

    // 动态 shape 一律 fallback AiCpu（对齐 TBE check_supported）：x/boxes 任一维为
    // -1，或任一输入为 -2 时拒绝认领。AiCpu kernel 运行时读取全部 shape/值，天然
    // 支持动态。box_index 仅拦 -2（tiling 需读其 shape），boxCoords 动态不检查
    // （均对齐 TBE）。
    auto hasUnknownDim = [](const std::vector<int64_t>& dims) {
        return std::find(dims.begin(), dims.end(), ge::UNKNOWN_DIM) != dims.end();
    };
    if (in.xDims.empty() || in.boxesDims.empty() || in.boxIndexUnknownRank || hasUnknownDim(in.xDims) ||
        in.boxesDims[0] == ge::UNKNOWN_DIM) {
        return rejected("dynamic shape (-1/-2) is not supported by aicore, fallback to aicpu");
    }

    // dtype：x/boxes 限 f32/f16，box_index 限 int32（AiCpu 侧 x 共收 9 种 dtype，
    // 其中 f64/uint8/int8/uint16/int16/int32/int64 这 7 种仅 AiCpu 可承接，故此处拒绝即 fallback）
    if (in.xDtype != ge::DT_FLOAT && in.xDtype != ge::DT_FLOAT16) {
        return rejected("x dtype not in {float32, float16}");
    }
    if (in.boxesDtype != ge::DT_FLOAT && in.boxesDtype != ge::DT_FLOAT16) {
        return rejected("boxes dtype not in {float32, float16}");
    }
    if (in.boxIndexDtype != ge::DT_INT32) {
        return rejected("box_index dtype must be int32");
    }

    // method 插值方式：SIMT kernel 仅实现 bilinear
    if (!in.method.empty() && in.method != "bilinear") {
        return rejected("method=" + in.method + ", only bilinear");
    }

    // x shape: 4D。dims 按 xFormat 分支解析（对齐 TBE）：NCHW → C=dims[1],H=dims[2],W=dims[3]；ND/NHWC →
    // H=dims[1],W=dims[2],C=dims[3]
    if (in.xDims.size() != static_cast<size_t>(X_DIM)) {
        return rejected("x must be 4D (N,H,W,C)");
    }
    int64_t h = 0;
    int64_t w = 0;
    int64_t c = 0;
    std::string depthDesc; // depth 所在 dim 位置，随 format 变化
    if (in.xFormat == ge::FORMAT_NCHW) {
        c = in.xDims[1];
        h = in.xDims[2];
        w = in.xDims[3];
        depthDesc = "x.shape[1]";
    } else if (in.xFormat == ge::FORMAT_ND || in.xFormat == ge::FORMAT_NHWC) {
        h = in.xDims[1];
        w = in.xDims[2];
        c = in.xDims[3];
        depthDesc = "x.shape[3]";
    } else {
        return rejected("x format must be ND/NHWC/NCHW, got " + ge::TypeUtils::FormatToSerialString(in.xFormat));
    }
    if (c < DEPTH_MIN || c > DEPTH_MAX) {
        return rejected("depth (" + depthDesc + ") must be in [" + std::to_string(DEPTH_MIN) + ", " +
                        std::to_string(DEPTH_MAX) + "], got " + std::to_string(c));
    }
    // h/w 为 0/负维时对 AiCpu 同样非法，由 tiling 阶段 all-dims-positive 拒绝，此处不重复拦
    if (h > 0 && w > 0) {
        int64_t hw = h * w;
        int64_t hwMax = (in.xDtype == ge::DT_FLOAT) ? HW_FP32_MAX : HW_MAX;
        if (hw > hwMax) {
            return rejected("H*W must be <= " + std::to_string(hwMax));
        }
    }

    // boxes shape: 2D (num_boxes, 4)
    if (in.boxesDims.size() != static_cast<size_t>(BOXES_DIM)) {
        return rejected("boxes must be 2D (num_boxes, 4)");
    }
    int64_t numBoxes = in.boxesDims[0];
    if (numBoxes <= NUM_BOXES_MIN || numBoxes > NUM_BOXES_MAX) {
        return rejected("num_boxes (boxes.shape[0]) must be in (" + std::to_string(NUM_BOXES_MIN) + ", " +
                        std::to_string(NUM_BOXES_MAX) + "], got " + std::to_string(numBoxes) +
                        ", performance is better on aicpu outside this range");
    }
    int64_t boxCoords = in.boxesDims[1];
    if (boxCoords != ge::UNKNOWN_DIM && boxCoords != BOX_COORDS) {
        return rejected("boxes.shape[1] must be " + std::to_string(BOX_COORDS));
    }

    // crop_size 值：正值 + max(crop_h, crop_w) <= 16 + crop 面积上限。
    // 无值（非 const，V1 读不到）时静默跳过，由 tiling 阶段兜底。
    if (in.cropSizeHasValue) {
        if (in.cropH <= 0 || in.cropW <= 0) {
            return rejected("crop_h/crop_w must be positive, got [" + std::to_string(in.cropH) + ", " +
                            std::to_string(in.cropW) + "]");
        }
        if (in.cropH > CROP_DIM_MAX || in.cropW > CROP_DIM_MAX) {
            return rejected("max(crop_h, crop_w) must be <= " + std::to_string(CROP_DIM_MAX) + ", got [" +
                            std::to_string(in.cropH) + ", " + std::to_string(in.cropW) + "]");
        }
        int64_t cropArea = static_cast<int64_t>(in.cropH) * in.cropW;
        if (cropArea > CROP_AREA_MAX) {
            return rejected("crop_h*crop_w must be <= " + std::to_string(CROP_AREA_MAX) + ", got " +
                            std::to_string(cropArea));
        }
    }
    return true;
}

// ge::TensorDesc 侧 unknown rank 判定；Ops::Base::IsUnknownRank 仅收 gert::Shape，
// ge::Operator 上下文无 gert 类型，仓内同场景惯例为 GetDims 与 UNKNOWN_RANK 比对
static bool IsUnknownRank(const ge::TensorDesc& desc) { return desc.GetShape().GetDims() == ge::UNKNOWN_RANK; }

static ge::graphStatus CheckIfAICoreSupported(const ge::Operator& op, ge::AscendString& result)
{
    CropAndResizeCheckInput in;
    // GetInputDescByName 按值返回 TensorDesc，各取一次复用
    ge::TensorDesc xDesc = op.GetInputDescByName("x");
    ge::TensorDesc boxesDesc = op.GetInputDescByName("boxes");
    ge::TensorDesc boxIndexDesc = op.GetInputDescByName("box_index");
    in.xDtype = xDesc.GetDataType();
    in.boxesDtype = boxesDesc.GetDataType();
    in.boxIndexDtype = boxIndexDesc.GetDataType();
    // x format：V1 TensorDesc::GetFormat 直接返回 ge::Format，GetPrimaryFormat 剥离 sub-format 位
    in.xFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc.GetFormat()));
    in.boxIndexUnknownRank = IsUnknownRank(boxIndexDesc);
    std::vector<int64_t> xDims = xDesc.GetShape().GetDims();
    std::vector<int64_t> boxesDims = boxesDesc.GetShape().GetDims();
    // unknown rank([-2]) 归一化：dims 清空（见结构体注释）
    in.xDims = IsUnknownRank(xDesc) ? std::vector<int64_t>{} : std::move(xDims);
    in.boxesDims = IsUnknownRank(boxesDesc) ? std::vector<int64_t>{} : std::move(boxesDims);
    (void)op.GetAttr("method", in.method);
    // crop_size 值（V1 独有能力：const 时经 GetInputConstData 读前驱 Constant 节点 weights）
    ge::Tensor cropTensor;
    if (op.GetInputConstData("crop_size", cropTensor) == ge::GRAPH_SUCCESS && cropTensor.GetData() != nullptr &&
        cropTensor.GetSize() == CROP_SIZE_LEN * sizeof(int32_t)) {
        const int32_t* cropData = reinterpret_cast<const int32_t*>(cropTensor.GetData());
        in.cropSizeHasValue = true;
        in.cropH = cropData[0];
        in.cropW = cropData[1];
    }

    std::string rejectDetail;
    if (!CheckCropAndResizeConstraints(in, rejectDetail)) {
        // reason 带性能路由语义，提示该拒绝为路由行为而非错误。
        // dynamicCompileStatic=True（对齐库上惯例，如 reverse_sequence/dynamic_stitch）：
        // 拒绝后 FE 继续轮询 AiCpu；若 AiCpu 也拒绝（如动态 + boxes=FP16），
        // GE 运行时编译（jit）模式下仍可按实参动态编译 AiCore 兜底。
        std::string json = "{\"isSupported\": \"False\", \"dynamicCompileStatic\": \"True\", \"reason\": "
                           "\"AiCore path of CropAndResize on ascend950 does not support this input (" +
                           rejectDetail + "), performance/compatibility is better on AiCpu, fallback expected.\"}";
        result = ge::AscendString(json.c_str());
        return ge::GRAPH_FAILED;
    }
    result = ge::AscendString(
        "{\"isSupported\": \"True\", \"dynamicCompileStatic\": \"True\", \"reason\": \"AICore CheckSupport Passed.\"}");
    return ge::GRAPH_SUCCESS;
}

class CropAndResize : public OpDef {
public:
    explicit CropAndResize(const char* name) : OpDef(name)
    {
        // 输入 x: 4D (N, H, W, C) ND/NHWC 或 (N, C, H, W) NCHW；NCHW 由 ND 候选承接（format
        // 由下游 infershape/tiling/check 按实际 desc format 分支解析，避免 FE 动态 shape 候选推导插转置）
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 输入 boxes: 2D (num_boxes, 4), float16/float32
        // NaN 检查已移至 kernel 运行时，tiling 不再读取 boxes 值，无需 ValueDepend
        this->Input("boxes")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 输入 box_index: 1D (num_boxes,), int32
        this->Input("box_index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 输入 crop_size: 1D (2,), int32
        // ValueDepend(OPTIONAL): infershape/tiling 需读取 crop_size 值
        this->Input("crop_size")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .ValueDepend(OPTIONAL)
            .AutoContiguous();

        // 输出 y: (num_boxes, crop_h, crop_w, C) ND/NHWC 或 (num_boxes, C, crop_h, crop_w) NCHW
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .AutoContiguous();

        // 属性: extrapolation_value (Float, 默认 0)
        this->Attr("extrapolation_value").AttrType(OPTIONAL).Float(0);

        // 属性: method (String, 默认 "bilinear", 仅支持 bilinear)
        this->Attr("method").AttrType(OPTIONAL).String("bilinear");

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            // true: 引擎分配阶段回调 CheckIfAICoreSupported，违反约束时拒绝并
            // fallback 到 AiCpu
            .NeedCheckSupportFlag(true)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "crop_and_resize_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
        this->AICore().AddConfig("ascend350", aicoreConfig);
        this->AICore().SetCheckSupport(CheckIfAICoreSupported);
    }
};

OP_ADD(CropAndResize);
} // namespace ops
