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
 * \file aipp_reshape_fusion_pass.cpp
 * \brief Compile-time images writeback for Aipp.
 */

#include "aipp_reshape_fusion_pass.h"

#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

#include "ge/ge_api_error_codes.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "register/register_custom_pass.h"
#include "version/ge-compiler_version.h"
using namespace ge;
using namespace fusion;

namespace {
const char* kPassName = "AippReshapeFusionPass";
const char* kOpTypeAipp = "Aipp";
const char* kAttrHasInferedVerified = "has_infered_verified";
const char* kAttrInputDims = "input_dims";
const char* kAttrAippConfigPath = "aipp_config_path";
const char* kAttrAippMode = "aipp_mode";
const char* kAttrMaxSrcImgSize = "max_src_image_size";
const char* kAttrCrop = "crop";
const char* kAttrCropSizeH = "crop_size_h";
const char* kAttrCropSizeW = "crop_size_w";
const char* kAttrResize = "resize";
const char* kAttrResizeOutputH = "resize_output_h";
const char* kAttrResizeOutputW = "resize_output_w";
const char* kAttrSrcImageSizeH = "src_image_size_h";
const char* kAttrSrcImageSizeW = "src_image_size_w";
const char* kAttrPadding = "padding";
const char* kAttrLeftPaddingSize = "left_padding_size";
const char* kAttrRightPaddingSize = "right_padding_size";
const char* kAttrTopPaddingSize = "top_padding_size";
const char* kAttrBottomPaddingSize = "bottom_padding_size";
const char* kAttrInputFormat = "input_format";
const char* kValueAippModeDynamic = "dynamic";
const char* kValueAippModeStatic = "static";
const char* kFormatYuv420spU8 = "YUV420SP_U8";
const char* kFormatXrgb8888U8 = "XRGB8888_U8";
const char* kFormatRgb888U8 = "RGB888_U8";
const char* kFormatArgb8888U8 = "ARGB8888_U8";
const char* kFormatYuyvU8 = "YUYV_U8";
const char* kFormatYuv422spU8 = "YUV422SP_U8";
const char* kFormatAyuv444U8 = "AYUV444_U8";
const char* kFormatYuv400U8 = "YUV400_U8";
const char* kFormatNc1hwc0diFp16 = "NC1HWC0DI_FP16";
const char* kFormatNc1hwc0diS8 = "NC1HWC0DI_S8";
const char* kFormatRaw10 = "RAW10";
const char* kFormatRaw12 = "RAW12";
const char* kFormatRaw16 = "RAW16";
const char* kFormatRaw24 = "RAW24";
const char* kFormatRgb16 = "RGB16";
const char* kFormatRgb20 = "RGB20";
const char* kFormatRgb24 = "RGB24";
const char* kFormatRgb8Ir = "RGB8_IR";
const char* kFormatRgb16Ir = "RGB16_IR";
const char* kFormatRgb24Ir = "RGB24_IR";

constexpr int32_t kImagesInputIndex = 0;
constexpr int32_t kFeaturesOutputIndex = 0;
constexpr int64_t kChannel1 = 1;
constexpr int64_t kChannel2 = 2;
constexpr int64_t kChannel3 = 3;
constexpr int64_t kChannel4 = 4;
constexpr int64_t kNum2 = 2;
constexpr int64_t kNum3 = 3;
constexpr int64_t kNum4 = 4;
constexpr int64_t kNum5 = 5;
constexpr int64_t kUnknownRankDim = -2;
constexpr size_t kUnknownRankDimCount = 1;
constexpr size_t kUnknownRankIndex = 0;
constexpr int64_t kHasInferedVerifiedValue = 1;
constexpr int64_t kDynamicCompileBatch = 1;
constexpr int64_t kDynamicDimRangeMin = 1;

constexpr size_t kBatchDim = 0;
constexpr size_t kNchwDimH = 2;
constexpr size_t kNchwDimW = 3;
constexpr size_t kNhwcDimH = 1;
constexpr size_t kNhwcDimW = 2;
constexpr size_t kC04DimC1 = 1;
constexpr size_t kC04DimH = 2;
constexpr size_t kC04DimW = 3;
constexpr size_t kC04DimC0 = 4;

struct AippParamsCfg {
    std::string aippMode;
    std::string inputFormat;
    int64_t maxSrcImageSize = 0;
    bool crop = false;
    int64_t cropSizeH = 0;
    int64_t cropSizeW = 0;
    bool resize = false;
    int64_t resizeOutputH = 0;
    int64_t resizeOutputW = 0;
    int64_t srcImageSizeH = 0;
    int64_t srcImageSizeW = 0;
    bool padding = false;
    int64_t leftPaddingSize = 0;
    int64_t rightPaddingSize = 0;
    int64_t topPaddingSize = 0;
    int64_t bottomPaddingSize = 0;
};

struct ImagesLayout {
    int64_t batch = 0;
    int64_t height = 0;
    int64_t width = 0;
    int64_t c1 = 0;
    int64_t c0 = 0;
    Format format = FORMAT_RESERVED;
};

std::string GetNodeName(const GNode& node)
{
    AscendString name;
    if (node.GetName(name) != GRAPH_SUCCESS || name.GetString() == nullptr) {
        return kOpTypeAipp;
    }
    return name.GetString();
}

Format ToPrimaryFormat(Format format) { return static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(format))); }

bool IsUnknownRankDims(const std::vector<int64_t>& dims)
{
    return dims.size() == kUnknownRankDimCount && dims[kUnknownRankIndex] == kUnknownRankDim;
}

template <class T>
graphStatus GetAippAttrValue(const nlohmann::json& cfgJson, const char* attrName, T& value)
{
    if (cfgJson.find(attrName) == cfgJson.end()) {
        return GRAPH_FAILED;
    }
    value = cfgJson[attrName];
    return GRAPH_SUCCESS;
}

graphStatus FillStaticCfg(const nlohmann::json& cfgJson, AippParamsCfg& params)
{
    (void)GetAippAttrValue(cfgJson, kAttrInputFormat, params.inputFormat);
    (void)GetAippAttrValue(cfgJson, kAttrCrop, params.crop);
    if (params.crop) {
        (void)GetAippAttrValue(cfgJson, kAttrCropSizeH, params.cropSizeH);
        (void)GetAippAttrValue(cfgJson, kAttrCropSizeW, params.cropSizeW);
    }
    (void)GetAippAttrValue(cfgJson, kAttrResize, params.resize);
    if (params.resize) {
        (void)GetAippAttrValue(cfgJson, kAttrResizeOutputH, params.resizeOutputH);
        (void)GetAippAttrValue(cfgJson, kAttrResizeOutputW, params.resizeOutputW);
    }
    (void)GetAippAttrValue(cfgJson, kAttrSrcImageSizeH, params.srcImageSizeH);
    (void)GetAippAttrValue(cfgJson, kAttrSrcImageSizeW, params.srcImageSizeW);
    (void)GetAippAttrValue(cfgJson, kAttrPadding, params.padding);
    if (params.padding) {
        (void)GetAippAttrValue(cfgJson, kAttrLeftPaddingSize, params.leftPaddingSize);
        (void)GetAippAttrValue(cfgJson, kAttrRightPaddingSize, params.rightPaddingSize);
        (void)GetAippAttrValue(cfgJson, kAttrTopPaddingSize, params.topPaddingSize);
        (void)GetAippAttrValue(cfgJson, kAttrBottomPaddingSize, params.bottomPaddingSize);
    }
    return GRAPH_SUCCESS;
}

graphStatus ParseAippCfg(const char* nodeName, const std::string& cfgJsonStr, AippParamsCfg& params)
{
    try {
        nlohmann::json cfgJson = nlohmann::json::parse(cfgJsonStr);
        OP_CHECK_IF(!cfgJson.is_object(),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "aipp_config_path", cfgJsonStr.c_str(),
                                                          "aipp_config_path must be a json object"),
                    return GRAPH_FAILED);
        if (GetAippAttrValue(cfgJson, kAttrAippMode, params.aippMode) != GRAPH_SUCCESS) {
            OP_LOGE_WITH_INVALID_INPUT(nodeName, "aipp_mode");
            return GRAPH_FAILED;
        }
        if (params.aippMode == kValueAippModeDynamic) {
            (void)GetAippAttrValue(cfgJson, kAttrMaxSrcImgSize, params.maxSrcImageSize);
            return GRAPH_SUCCESS;
        }
        OP_CHECK_IF(params.aippMode != kValueAippModeStatic,
                    OP_LOGE_FOR_INVALID_VALUE(nodeName, "aipp_mode", params.aippMode.c_str(), "static or dynamic"),
                    return GRAPH_FAILED);
        return FillStaticCfg(cfgJson, params);
    } catch (...) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "aipp_config_path", "invalid json",
                                              "failed to parse aipp_config_path");
        return GRAPH_FAILED;
    }
}

int64_t GetChannel(const AippParamsCfg& params)
{
    if (params.inputFormat == kFormatXrgb8888U8 || params.inputFormat == kFormatArgb8888U8 ||
        params.inputFormat == kFormatAyuv444U8 || params.inputFormat == kFormatRgb8Ir ||
        params.inputFormat == kFormatRgb16Ir || params.inputFormat == kFormatRgb24Ir) {
        return kChannel4;
    }
    if (params.inputFormat == kFormatYuv400U8 || params.inputFormat == kFormatRaw10 ||
        params.inputFormat == kFormatRaw12 || params.inputFormat == kFormatRaw16 ||
        params.inputFormat == kFormatRaw24) {
        return kChannel1;
    }
    return kChannel3;
}

int64_t PreferCfgSize(int64_t cfgSize, int64_t layoutSize) { return cfgSize ? cfgSize : layoutSize; }

void SetShapeAndOrigin(TensorDesc& desc, const Shape& shape)
{
    desc.SetShape(shape);
    desc.SetOriginShape(shape);
}

void SetFormatAndOrigin(TensorDesc& desc, Format format)
{
    desc.SetFormat(format);
    desc.SetOriginFormat(format);
}

struct SrcGeom {
    int64_t batch = 0;
    int64_t c1 = 0;
    int64_t srcH = 0;
    int64_t srcW = 0;
};

int64_t BatchHwMul(const SrcGeom& geom, int64_t factor) { return geom.batch * geom.srcH * geom.srcW * factor; }

struct SrcMeasure {
    bool matched = false;
    DataType dtype = DT_UNDEFINED;
    int64_t size = 0;
};

SrcMeasure MeasureInterleavedSrc(const AippParamsCfg& params, const SrcGeom& geom)
{
    const int64_t batch = geom.batch;
    const int64_t c1 = geom.c1;
    const int64_t srcH = geom.srcH;
    const int64_t srcW = geom.srcW;
    if (params.inputFormat == kFormatYuv420spU8) {
        return {true, DT_UINT8, batch * kNum3 * srcH * srcW / kNum2};
    }
    if (params.inputFormat == kFormatXrgb8888U8 || params.inputFormat == kFormatArgb8888U8 ||
        params.inputFormat == kFormatAyuv444U8) {
        return {true, DT_UINT8, batch * kChannel4 * srcH * srcW};
    }
    if (params.inputFormat == kFormatRgb888U8) {
        return {true, DT_UINT8, batch * kChannel3 * srcH * srcW};
    }
    if (params.inputFormat == kFormatYuv400U8) {
        return {true, DT_UINT8, batch * srcH * srcW};
    }
    if (params.inputFormat == kFormatNc1hwc0diFp16) {
        return {true, DT_FLOAT16, batch * c1 * srcH * srcW * kChannel4 * kNum2};
    }
    if (params.inputFormat == kFormatNc1hwc0diS8) {
        return {true, DT_INT8, batch * c1 * srcH * srcW * kChannel4};
    }
    if (params.inputFormat == kFormatYuyvU8 || params.inputFormat == kFormatYuv422spU8) {
        return {true, DT_UINT8, batch * kChannel2 * srcH * srcW};
    }
    return {};
}

SrcMeasure MeasureRawIrSrc(const AippParamsCfg& params, const SrcGeom& geom)
{
    if (params.inputFormat == kFormatRaw10 || params.inputFormat == kFormatRaw12 ||
        params.inputFormat == kFormatRaw16) {
        return {true, DT_UINT16, BatchHwMul(geom, kChannel2)};
    }
    if (params.inputFormat == kFormatRaw24) {
        return {true, DT_UINT32, BatchHwMul(geom, kChannel4)};
    }
    if (params.inputFormat == kFormatRgb16) {
        return {true, DT_UINT16, BatchHwMul(geom, kChannel3)};
    }
    if (params.inputFormat == kFormatRgb20 || params.inputFormat == kFormatRgb24) {
        return {true, DT_UINT32, BatchHwMul(geom, kChannel3)};
    }
    if (params.inputFormat == kFormatRgb8Ir) {
        return {true, DT_UINT8, BatchHwMul(geom, kChannel4)};
    }
    if (params.inputFormat == kFormatRgb16Ir) {
        return {true, DT_UINT16, BatchHwMul(geom, kChannel4)};
    }
    if (params.inputFormat == kFormatRgb24Ir) {
        return {true, DT_UINT32, BatchHwMul(geom, kChannel4)};
    }
    return {};
}

int64_t GetSrcImageSizeDtype(const AippParamsCfg& params, const ImagesLayout& layout, DataType& srcImgDtype)
{
    const SrcGeom geom{layout.batch, layout.c1, PreferCfgSize(params.srcImageSizeH, layout.height),
                       PreferCfgSize(params.srcImageSizeW, layout.width)};
    SrcMeasure measure = MeasureInterleavedSrc(params, geom);
    if (!measure.matched) {
        measure = MeasureRawIrSrc(params, geom);
    }
    if (!measure.matched) {
        const char* inputFormat = params.inputFormat.empty() ? "empty" : params.inputFormat.c_str();
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("Aipp", "input_format", inputFormat, "input_format is undefined");
        return 0;
    }
    srcImgDtype = measure.dtype;
    return measure.size;
}

std::vector<int64_t> GetAclInputDims(const AippParamsCfg& params, int64_t batch, int64_t srcH, int64_t srcW)
{
    int64_t height = srcH;
    int64_t channel = kChannel3;
    if (params.inputFormat == kFormatYuv420spU8) {
        height = srcH * kNum3 / kNum2;
        channel = kChannel1;
    } else if (params.inputFormat == kFormatArgb8888U8 || params.inputFormat == kFormatAyuv444U8 ||
               params.inputFormat == kFormatXrgb8888U8) {
        channel = kChannel4;
    } else if (params.inputFormat == kFormatRgb888U8) {
        channel = kChannel3;
    } else if (params.inputFormat == kFormatYuv400U8) {
        channel = kChannel1;
    } else if (params.inputFormat == kFormatYuyvU8) {
        channel = kChannel2;
    } else if (params.inputFormat == kFormatYuv422spU8) {
        height = srcH * kNum2;
        channel = kChannel1;
    } else if (params.inputFormat == kFormatRaw10 || params.inputFormat == kFormatRaw12 ||
               params.inputFormat == kFormatRaw16 || params.inputFormat == kFormatRaw24) {
        channel = kChannel1;
    } else if (params.inputFormat == kFormatRgb16 || params.inputFormat == kFormatRgb20 ||
               params.inputFormat == kFormatRgb24) {
        channel = kChannel3;
    } else if (params.inputFormat == kFormatRgb8Ir || params.inputFormat == kFormatRgb16Ir ||
               params.inputFormat == kFormatRgb24Ir) {
        channel = kChannel4;
    } else {
        const char* inputFormat = params.inputFormat.empty() ? "empty" : params.inputFormat.c_str();
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("Aipp", "input_format", inputFormat, "input_format is undefined");
    }
    return {batch, height, srcW, channel};
}

graphStatus ParseImagesLayout(const char* nodeName, const TensorDesc& imagesDesc, ImagesLayout& layout)
{
    layout.format = ToPrimaryFormat(imagesDesc.GetFormat());
    const auto imagesShape = imagesDesc.GetShape().GetDims();
    const auto dimNum = static_cast<int64_t>(imagesDesc.GetShape().GetDimNum());
    if (((layout.format == FORMAT_NCHW || layout.format == FORMAT_NHWC) && dimNum < kNum4) ||
        (layout.format == FORMAT_NC1HWC0_C04 && dimNum < kNum5)) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            nodeName, "images", std::to_string(dimNum).c_str(),
            "NCHW and NHWC need at least 4 dimensions, NC1HWC0_C04 needs at least 5 dimensions");
        return GRAPH_FAILED;
    }
    if (layout.format == FORMAT_NCHW) {
        layout.batch = imagesShape[kBatchDim];
        layout.height = imagesShape[kNchwDimH];
        layout.width = imagesShape[kNchwDimW];
        return GRAPH_SUCCESS;
    }
    if (layout.format == FORMAT_NHWC) {
        layout.batch = imagesShape[kBatchDim];
        layout.height = imagesShape[kNhwcDimH];
        layout.width = imagesShape[kNhwcDimW];
        return GRAPH_SUCCESS;
    }
    if (layout.format == FORMAT_NC1HWC0_C04) {
        layout.batch = imagesShape[kBatchDim];
        layout.c1 = imagesShape[kC04DimC1];
        layout.height = imagesShape[kC04DimH];
        layout.width = imagesShape[kC04DimW];
        layout.c0 = imagesShape[kC04DimC0];
        return GRAPH_SUCCESS;
    }
    const ge::AscendString formatStr = ge::TypeUtils::FormatToAscendString(layout.format);
    const char* actualFormat = (formatStr.GetString() == nullptr) ? "NONE" : formatStr.GetString();
    OP_LOGE_FOR_INVALID_FORMAT(nodeName, "images", actualFormat, "NCHW, NHWC or NC1HWC0_C04");
    return GRAPH_FAILED;
}

graphStatus WriteFeaturesDescFromImages(GNode& node, const TensorDesc& imagesDesc, int64_t featuresSize)
{
    const std::string nodeName = GetNodeName(node);
    TensorDesc featuresDesc;
    OP_CHECK_IF(node.GetOutputDesc(kFeaturesOutputIndex, featuresDesc) != GRAPH_SUCCESS,
                OP_LOGE(nodeName.c_str(), "GetOutputDesc features failed."), return GRAPH_FAILED);
    featuresDesc.SetFormat(imagesDesc.GetFormat());
    featuresDesc.SetOriginFormat(imagesDesc.GetOriginFormat());
    featuresDesc.SetSize(featuresSize);
    OP_CHECK_IF(node.UpdateOutputDesc(kFeaturesOutputIndex, featuresDesc) != GRAPH_SUCCESS,
                OP_LOGE(nodeName.c_str(), "UpdateOutputDesc features failed."), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

graphStatus SetHasInferedVerified(GNode& node)
{
    AscendString attrName(kAttrHasInferedVerified);
    int64_t verified = kHasInferedVerifiedValue;
    const std::string nodeName = GetNodeName(node);
    OP_CHECK_IF(node.SetAttr(attrName, verified) != GRAPH_SUCCESS,
                OP_LOGE(nodeName.c_str(), "SetAttr has_infered_verified failed."), return GRAPH_FAILED);
    return GRAPH_SUCCESS;
}

void FillStaticPackedShape(const ImagesLayout& layout, int64_t srcH, int64_t srcW, int64_t realChannel,
                           TensorDesc& imagesDesc)
{
    if (layout.format == FORMAT_NCHW || layout.format == FORMAT_NHWC) {
        SetFormatAndOrigin(imagesDesc, FORMAT_NHWC);
        SetShapeAndOrigin(imagesDesc, Shape({layout.batch, srcH, srcW, realChannel}));
        return;
    }
    SetShapeAndOrigin(imagesDesc, Shape({layout.batch, layout.c1, srcH, srcW, layout.c0}));
}

graphStatus WriteBackStatic(GNode& node, const AippParamsCfg& params)
{
    const std::string nodeName = GetNodeName(node);
    TensorDesc imagesDesc;
    OP_CHECK_IF(node.GetInputDesc(kImagesInputIndex, imagesDesc) != GRAPH_SUCCESS,
                OP_LOGE(nodeName.c_str(), "GetInputDesc images failed."), return GRAPH_FAILED);
    ImagesLayout layout;
    OP_CHECK_IF(ParseImagesLayout(nodeName.c_str(), imagesDesc, layout) != GRAPH_SUCCESS, , return GRAPH_FAILED);
    const int64_t realChannel = (layout.format == FORMAT_NC1HWC0_C04) ? kChannel1 : GetChannel(params);
    DataType srcDtype = DT_UINT8;
    const int64_t packedSize = GetSrcImageSizeDtype(params, layout, srcDtype);
    OP_CHECK_IF(WriteFeaturesDescFromImages(node, imagesDesc, packedSize) != GRAPH_SUCCESS, , return GRAPH_FAILED);
    imagesDesc.SetSize(packedSize);
    const int64_t srcH = PreferCfgSize(params.srcImageSizeH, layout.height);
    const int64_t srcW = PreferCfgSize(params.srcImageSizeW, layout.width);
    FillStaticPackedShape(layout, srcH, srcW, realChannel, imagesDesc);
    imagesDesc.SetDataType(srcDtype);
    OP_CHECK_IF(node.UpdateInputDesc(kImagesInputIndex, imagesDesc) != GRAPH_SUCCESS,
                OP_LOGE(nodeName.c_str(), "UpdateInputDesc images failed."), return GRAPH_FAILED);
    OP_CHECK_IF(SetHasInferedVerified(node) != GRAPH_SUCCESS, , return GRAPH_FAILED);
    std::vector<int64_t> aclInputDims = GetAclInputDims(params, layout.batch, srcH, srcW);
    if (aclInputDims.size() >= static_cast<size_t>(kNum4)) {
        AscendString dimsName(kAttrInputDims);
        OP_CHECK_IF(node.SetAttr(dimsName, aclInputDims) != GRAPH_SUCCESS,
                    OP_LOGE(nodeName.c_str(), "SetAttr input_dims failed."), return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

void FillDynamicShapeRange(int64_t srcImageSize, TensorDesc& imagesDesc)
{
    std::vector<std::pair<int64_t, int64_t>> imagesRange;
    (void)imagesDesc.GetShapeRange(imagesRange);
    if (imagesRange.size() != static_cast<size_t>(kNum4)) {
        return;
    }
    std::vector<std::pair<int64_t, int64_t>> newRange = {{kDynamicCompileBatch, kDynamicCompileBatch},
                                                         {kDynamicDimRangeMin, srcImageSize}};
    (void)imagesDesc.SetShapeRange(newRange);
}

graphStatus WriteBackDynamicCompile(GNode& node, const AippParamsCfg& params)
{
    const std::string nodeName = GetNodeName(node);
    OP_CHECK_IF(params.maxSrcImageSize <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName.c_str(), "max_src_image_size",
                                                      std::to_string(params.maxSrcImageSize).c_str(),
                                                      "max_src_image_size must be greater than 0"),
                return GRAPH_FAILED);
    TensorDesc imagesDesc;
    OP_CHECK_IF(node.GetInputDesc(kImagesInputIndex, imagesDesc) != GRAPH_SUCCESS,
                OP_LOGE(nodeName.c_str(), "GetInputDesc images failed."), return GRAPH_FAILED);
    OP_CHECK_IF(
        IsUnknownRankDims(imagesDesc.GetShape().GetDims()),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(nodeName.c_str(), "images", "[-2]", "unknown rank is not supported"),
        return GRAPH_FAILED);
    OP_CHECK_IF(WriteFeaturesDescFromImages(node, imagesDesc, imagesDesc.GetSize()) != GRAPH_SUCCESS, ,
                return GRAPH_FAILED);
    imagesDesc.SetSize(params.maxSrcImageSize);
    FillDynamicShapeRange(params.maxSrcImageSize, imagesDesc);
    SetShapeAndOrigin(imagesDesc, Shape({kDynamicCompileBatch, params.maxSrcImageSize}));
    imagesDesc.SetDataType(DT_UINT8);
    SetFormatAndOrigin(imagesDesc, FORMAT_NHWC);
    OP_CHECK_IF(node.UpdateInputDesc(kImagesInputIndex, imagesDesc) != GRAPH_SUCCESS,
                OP_LOGE(nodeName.c_str(), "UpdateInputDesc images failed."), return GRAPH_FAILED);
    return SetHasInferedVerified(node);
}

graphStatus ProcessAippNode(GNode& node)
{
    const std::string nodeName = GetNodeName(node);
    AscendString cfgAttrName(kAttrAippConfigPath);
    AscendString cfgPath;
    OP_CHECK_IF(node.GetAttr(cfgAttrName, cfgPath) != GRAPH_SUCCESS || cfgPath.GetString() == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(nodeName.c_str(), "aipp_config_path"), return GRAPH_FAILED);
    AippParamsCfg params;
    if (ParseAippCfg(nodeName.c_str(), cfgPath.GetString(), params) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    if (params.aippMode == kValueAippModeDynamic) {
        return WriteBackDynamicCompile(node, params);
    }
    return WriteBackStatic(node, params);
}

bool IsAippNode(const GNode& node)
{
    AscendString opType;
    if (node.GetType(opType) != GRAPH_SUCCESS || opType.GetString() == nullptr) {
        return false;
    }
    return AscendString(kOpTypeAipp) == opType;
}
} // namespace

namespace Ops {
ge::Status AippReshapeFusionPass::Run(ge::GraphPtr& graph, ge::CustomPassContext& passContext)
{
    OP_LOGD(kPassName, "Enter AippReshapeFusionPass");
    OP_CHECK_IF(graph == nullptr, OP_LOGE(kPassName, "%s is nullptr!", "graph"), return FAILED);

    bool changed = false;
    for (auto& node : graph->GetDirectNode()) {
        if (!IsAippNode(node)) {
            continue;
        }
        if (node.HasAttr(AscendString(kAttrHasInferedVerified))) {
            continue;
        }
        if (ProcessAippNode(node) != GRAPH_SUCCESS) {
            return FAILED;
        }
        changed = true;
    }

    if (!changed) {
        OP_LOGD(kPassName, "No Aipp images desc rewritten");
        return GRAPH_NOT_CHANGED;
    }
    OP_LOGD(kPassName, "Aipp images writeback success");
    return SUCCESS;
}
#if GE_COMPILER_VERSION_NUM >= 90200000U
REG_FUSION_PASS(AippReshapeFusionPass).Stage(CustomPassStage::kAfterInferShape);
#endif
} // namespace Ops
