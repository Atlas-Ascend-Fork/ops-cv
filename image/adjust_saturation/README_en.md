# AdjustSaturation

English | [简体中文](./README.md)

## Product Support

| Product | Supported |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function Description

- Operator function: This operator adjusts the saturation of an RGB image.

## Parameter Description

| Parameter name | Input/Output/Attr | Description | Data type | Data format |
|-----|----------|----------------------------------------------------------------------------------------------------------------------------------------------------|----------------|------|
| images | Input | Input Tensor. It has at least three dimensions. The size of the last dimension must be 3. The Tensor is interpreted as an image with three RGB channels. | FLOAT16, FLOAT | ND |
| scale | Input | A one-dimensional Tensor that contains one element. The only element indicates the saturation scaling factor. | FLOAT | ND |
| y | Output | Output Tensor. Its shape and data type are the same as those of images. | FLOAT16, FLOAT | ND |

## Constraints

- images has at least three dimensions, and the last dimension must be 3.
- scale must be a FLOAT Tensor that contains only one element.
- The data type and data size of y must be the same as those of images.

## Invocation Description
