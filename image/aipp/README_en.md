# Aipp

English | [简体中文](./README.md)

## Product Support

|Product | Supported |
|:-------------------------|:----------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Ascend 350 products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products </term> | × |
| <term>Atlas training products</term> | × |

## Function Description

- Operator function:

    AIPP (Artificial Intelligence Pre-Processing) performs AI pre-processing. This operator completes data pre-processing on the AI Core, including resizing images, converting color spaces (converting image formats), and subtracting the mean value or multiplying a coefficient (changing image pixels). The actual model inference starts after data pre-processing. **Both dynamic AIPP and static AIPP are currently supported.** For the complete description of AIPP functions, refer to the "Advanced Functions > Enabling AIPP" chapter in the [ATC Offline Model Compilation Tool](https://hiascend.com/document/redirect/CannCommunityAtc
      ) document.

- Computation process:
    1. The operator calculates the mapping between the input and output based on the passed-in cropping parameters, and crops the image.

    2. The operator converts the color space of the image based on the passed-in image format and conversion parameters. The conversion formulas are as follows:

        - YUV to RGB:

            ```text
            | R |   | matrix_r0c0 matrix_r0c1 matrix_r0c2 | | Y - input_bias_0 |
            | G | = | matrix_r1c0 matrix_r1c1 matrix_r1c2 | | U - input_bias_1 | >> 8
            | B |   | matrix_r2c0 matrix_r2c1 matrix_r2c2 | | V - input_bias_2 |
            ```

        - RGB to YUV:

            ```text
            | Y |   | matrix_r0c0 matrix_r0c1 matrix_r0c2 | | R |        | output_bias_0 |
            | U | = | matrix_r1c0 matrix_r1c1 matrix_r1c2 | | G | >> 8 + | output_bias_1 |
            | V |   | matrix_r2c0 matrix_r2c1 matrix_r2c2 | | B |        | output_bias_2 |
            ```

    3. The operator normalizes the data. The formula is as follows:

        ```text
        pixel_out_chx(i) = [pixel_in_chx(i) - mean_chn_i - min_chn_i] * var_reci_chn_i
        ```

## Parameter Description

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attr</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>images</td>
      <td>Input</td>
      <td>Raw image input.</td>
      <td>UINT8</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>params</td>
      <td>Input</td>
      <td>Optional parameter. Dynamic AIPP uses this parameter to receive preset parameters.</td>
      <td>UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>aipp_config_path</td>
      <td>Optional attr</td>
      <td>Path of the file that stores configuration parameters. The default value is "./aipp.cfg". For how to configure this file, refer to <a href='#constraints'>Constraints</a>.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>features</td>
      <td>Output</td>
      <td>Processed image. If the output color space type is a grayscale image, the supported number of output channels C is 1 or 3. If the output color space type is a non-grayscale image, the supported number of output channels C is 3.
      </td>
      <td>UINT8, FLOAT16</td>
      <td>NCHW, NHWC</td>
    </tr>
  </tbody></table>

## Constraints

- Description of the aipp_config_path configuration file: Both dynamic AIPP and static AIPP configurations are supported. Static AIPP supports image cropping parameter configuration (Crop), color space conversion parameter configuration (CSC), and output type conversion parameter configuration (DTC). Dynamic AIPP supports the input processing flag (related_input_rank) parameter and the maximum input image size (max_src_image_size) parameter.

- The following example shows the configuration of the aipp_config_path file. For details about the parameters, refer to the following table.

    ```Cpp
    aipp_op {
        aipp_mode : static
        src_image_size_w : 608
        src_image_size_h : 608
        crop : false
        input_format : YUV420SP_U8

        csc_switch : true
        matrix_r0c0 : 298
        matrix_r0c1 : 0
        matrix_r0c2 : 409
        matrix_r1c0 : 298
        matrix_r1c1 : -100
        matrix_r1c2 : -208
        matrix_r2c0 : 298
        matrix_r2c1 : 516
        matrix_r2c2 : 0
        input_bias_0 : 16
        input_bias_1 : 128
        input_bias_2 : 128

        padding : false
        left_padding_size : 0
        right_padding_size : 0
        top_padding_size : 0
        bottom_padding_size : 0
        padding_value : 0

        mean_chn_0: 0
        mean_chn_1: 0
        mean_chn_2: 0
        mean_chn_3: 0
        min_chn_0: 0.0
        min_chn_1: 0.0
        min_chn_2: 0.0
        min_chn_3: 0.0
        var_reci_chn_0: 1.0
        var_reci_chn_1: 1.0
        var_reci_chn_2: 1.0
        var_reci_chn_3: 1.0
    }

    aipp_op {
        aipp_mode: dynamic
        related_input_rank: 0
        max_src_image_size: 752640
    }
    ```

    <table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
    <col style="width: 150px">
    <col style="width: 120px">
    <col style="width: 150px">
    <col style="width: 350px">
    <col style="width: 120px">
    </colgroup>
    <thead>
        <tr >
        <th colspan="2"  style="text-align: center">Parameter name</th>
        <th>Input/Output/Attr</th>
        <th>Description</th>
        <th>Data type</th>
        </tr>
    </thead>
    <tbody>
        <tr>
        <td rowspan="1">Global settings</td>
        <td>aipp_mode</td>
        <td>Required attr</td>
        <td>AIPP operator running mode. dynamic indicates dynamic AIPP, and static indicates static AIPP.</td>
        <td>ENUM</td>
        </tr>
        <tr>
        <td rowspan="3">Static AIPP parameter settings</td>
        <td>input_format</td>
        <td>Required attr</td>
        <td>Input image type. This parameter is required only for static AIPP configuration and is not required for dynamic AIPP. Currently, only RGB888_U8, XRGB8888_U8, YUV420SP_U8, and YUV400_U8 are supported.</td>
        <td>ENUM</td>
        </tr>
        <tr>
        <td>src_image_size_h</td>
        <td>Optional attr</td>
        <td>Input image height. This parameter takes effect only for static AIPP configuration. Value range: [1,4096]. If the input image type is YUV420SP_U8, this value must be an even number.</td>
        <td>INT32</td>
        </tr>
        <tr>
        <td>src_image_size_w</td>
        <td>Optional attr</td>
        <td>Input image width. This parameter takes effect only for static AIPP configuration. Value range: [1,4096]. If the input image type is YUV420SP_U8, this value must be an even number.</td>
        <td>INT32</td>
        </tr>
        <tr>
        <td rowspan="2">Dynamic AIPP parameter settings</td>
        <td>related_input_rank</td>
        <td>Optional attr</td>
        <td>Identifies the input on which AIPP is performed. This parameter takes effect only for dynamic AIPP configuration. For example, 0 indicates that AIPP is performed on the first input. The value must be greater than or equal to 0.</td>
        <td>INT32</td>
        </tr>
        <tr>
        <td>max_src_image_size</td>
        <td>Required attr</td>
        <td>Maximum size of the input image. This parameter is required for dynamic AIPP configuration.</td>
        <td>INT32</td>
        </tr>
        <tr>
        <td rowspan="5">CROP parameter settings</td>
        <td>crop</td>
        <td>Optional attr</td>
        <td>Cropping switch. true indicates that cropping is supported, and false indicates that cropping is not supported.</td>
        <td>BOOL</td>
        </tr>
        <tr>
        <td>load_start_pos_h</td>
        <td>Optional attr</td>
        <td>Vertical coordinate of the start position for image cropping. Value range: [0,4095]. If the input image type is YUV420SP_U8, this value must be an even number.</td>
        <td>INT32</td>
        </tr>
        <tr>
        <td>load_start_pos_w</td>
        <td>Optional attr</td>
        <td>Horizontal coordinate of the start position for image cropping. Value range: [0,4095]. If the input image type is YUV420SP_U8, this value must be an even number.</td>
        <td>INT32</td>
        </tr>
        <tr>
        <td>crop_size_h</td>
        <td>Optional attr</td>
        <td>Image height after cropping. Value range: [1,4096].</td>
        <td>INT32</td>
        </tr>
        <tr>
        <td>crop_size_w</td>
        <td>Optional attr</td>
        <td>Image width after cropping. Value range: [1,4096].</td>
        <td>INT32</td>
        </tr>
        <tr>
        <td rowspan="6">CSC parameter settings</td>
        <td>csc_switch</td>
        <td>Optional attr</td>
        <td>Color space conversion switch. This parameter is configured for static AIPP. true indicates that color space conversion is enabled, and false indicates that it is disabled.</td>
        <td>BOOL</td>
        </tr>
        <tr>
        <td>rbuv_swap_switch</td>
        <td>Optional attr</td>
        <td>Switch for swapping the R channel with the B channel or the U channel with the V channel. true indicates that channel swapping is enabled, and false indicates that it is not enabled.</td>
        <td>BOOL</td>
        </tr>
        <tr>
        <td>ax_swap_switch</td>
        <td>Optional attr</td>
        <td>Switch for RGBA->ARGB swapping. true indicates that swapping is enabled, and false indicates that it is not enabled.</td>
        <td>BOOL</td>
        </tr>
        <tr>
        <td>
        matrix_r0c0 matrix_r0c1 matrix_r0c2
        matrix_r1c0 matrix_r1c1 matrix_r1c2
        matrix_r2c0 matrix_r2c1 matrix_r2c2
        </td>
        <td>Optional attr</td>
        <td>Elements of the 3*3 CSC matrix. Value range: [-32677,32676].</td>
        <td>INT16</td>
        </tr>
        <tr>
        <td>output_bias_0 output_bias_1 output_bias_2</td>
        <td>Optional attr</td>
        <td>Output offsets for RGB-to-YUV conversion. Value range: [0,255].</td>
        <td>UINT8</td>
        </tr>
        <tr>
        <td>input_bias_0 input_bias_1 input_bias_2</td>
        <td>Optional attr</td>
        <td>Input offsets for YUV-to-RGB conversion. Value range: [0,255].</td>
        <td>UINT8 </td>
        </tr>
        <tr>
        <td rowspan="3">DTC parameter settings</td>
        <td>mean_chn_0 mean_chn_1 mean_chn_2 mean_chn_3</td>
        <td>Optional attr</td>
        <td>Mean value of each channel. Value range: [0,255].</td>
        <td>UINT8</td>
        </tr>
        <tr>
        <td>min_chn_0 min_chn_1 min_chn_2 min_chn_3</td>
        <td>Optional attr</td>
        <td>Minimum value of each channel. Value range: [0,255].</td>
        <td>FLOAT16</td>
        </tr>
        <tr>
        <td>var_reci_chn_0 var_reci_chn_1 var_reci_chn_2 var_reci_chn_3</td>
        <td>Optional attr</td>
        <td>Reciprocal of the variance of each channel. Value range: [-65504,65504].</td>
        <td>FLOAT16</td>
        </tr>
        <tr>
        <td rowspan="3">padding parameter settings</td>
        <td>padding</td>
        <td>Optional attr</td>
        <td>Switch for enabling padding when AIPP processes an image. Value range: true/false.</td>
        <td>BOOL</td>
        </tr>
        <tr>
        <td>left_padding_size right_padding_size top_padding_size bottom_padding_size</td>
        <td>Optional attr</td>
        <td>Padding values of H and W. This parameter is configured for static AIPP. Value range: [0,32].</td>
        <td>INT32</td>
        </tr>
        <tr>
        <td>padding_value</td>
        <td>Optional attr</td>
        <td>Pixel value used for padding in the top, bottom, left, and right directions. This parameter is configured for static AIPP. Value range: [0,255]/[-65504,65504].</td>
        <td>UINT8/FLOAT16</td>
        </tr>
    </tbody></table>

## Invocation Description

| Invocation Method | Sample Code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_aipp](examples/arch35/test_geir_aipp.cpp) | Calls the Aipp operator by constructing a graph in the [operator IR](op_graph/aipp_proto.h) mode.|
