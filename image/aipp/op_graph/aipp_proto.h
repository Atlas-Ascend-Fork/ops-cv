/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aipp_proto.h
 * \brief
 */

#ifndef OP_GRAPH_AIPP_PROTO_H
#define OP_GRAPH_AIPP_PROTO_H

#include "graph/operator_reg.h"

namespace ge {
/**
*@brief Performs AI pre-processing (AIPP) on images including color space conversion (CSC),
image normalization (by subtracting the mean value or multiplying a factor), image cropping
(by specifying the crop start and cropping the image to the size required by the neural network), and much more. \n

*@par Inputs:
*@li images: An NCHW or NHWC tensor of type uint8, specifying the input to the data layer.
*@li params: A optional Tensor. Dynamic AIPP configuration parameters of type uint8. \n

*@par Attributes:
*aipp_config_path: An optional string, specifying the path of the AIPP configuration file, default value is
*"./aipp.cfg". Note that the default value does not mean the configuration file can be absent: the file content
*and availability still comply with the existing AIPP constraints. \n

*@par Outputs:
*features: The AIPP-processed output tensor of type float16 or uint8.
*@par Third-party framework compatibility
* It is a custom operator. It has no corresponding operator in Caffe.
*@par Restrictions:
*Warning: This operator can be integrated only by configuring INSERT_OP_FILE of aclgrphBuildModel. Please do not use it
directly.
*/
REG_OP(Aipp)
    .INPUT(images, TensorType{DT_UINT8})
    .OPTIONAL_INPUT(params, TensorType{DT_UINT8})
    .OUTPUT(features, TensorType({DT_FLOAT16, DT_UINT8}))
    .ATTR(aipp_config_path, String, "./aipp.cfg")
    .OP_END_FACTORY_REG(Aipp)
} // namespace ge

#endif // OP_GRAPH_AIPP_PROTO_H
