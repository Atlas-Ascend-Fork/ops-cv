/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "../../../op_host/arch35/aipp_tiling.h"
#include "../../../op_kernel/arch35/aipp_struct.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;
using namespace Aipp_Kernel;

// 设置 AIPP_DUMP_IO=1 环境变量可开启输入输出dump
static bool IsDumpEnabled()
{
    const char* env = getenv("AIPP_DUMP_IO");
    return env != nullptr && string(env) == "1";
}

// 设置 AIPP_DETERMINISTIC=1 环境变量可使用固定值初始化输入（便于复现问题）
// 默认使用随机数初始化，泛化验证
static bool IsDeterministicInit()
{
    const char* env = getenv("AIPP_DETERMINISTIC");
    return env != nullptr && string(env) == "1";
}

// kernel函数声明，参数顺序与aipp.cpp一致
extern "C" void Aipp(uint8_t* images, uint8_t* params, uint8_t* features, uint8_t* workspace, uint8_t* tiling);

class aipp_kernel_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "aipp_kernel_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "aipp_kernel_test TearDown" << endl; }
};

// 根据imageFormat计算每像素字节数
static float GetBytesPerPixel(uint8_t imageFormat)
{
    // 1=YUV420SP_U8, 2=RGB888_U8, 3=XRGB8888_U8, 4=YUV400_U8
    switch (imageFormat) {
        case 1:
            return 1.5f; // YUV420SP
        case 2:
            return 4.0f; // XRGB8888
        case 5:
            return 3.0f; // RGB888
        case 10:
            return 1.0f; // YUV400
        default:
            return 3.0f;
    }
}

// 根据outputFormat计算输出每像素每通道字节数
static size_t GetOutputBytesPerPixel(uint8_t outputFormat, uint32_t channelNum)
{
    // outputFormat: 1=NCHW(fp16), 2=NHWC(uint8)
    if (outputFormat == 1) {
        return channelNum * sizeof(uint16_t); // fp16
    }
    return channelNum * sizeof(uint8_t);
}

// ========== Golden 计算函数 ==========

// fp16 -> float 手动转换
static float Fp16ToFloat(uint16_t bits)
{
    uint32_t sign = (bits >> 15) & 0x1;
    uint32_t exponent = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x3FF;
    float val;
    if (exponent == 0 && mantissa == 0) {
        val = 0.0f;
    } else if (exponent == 0) {
        val = ldexpf(static_cast<float>(mantissa) / 1024.0f, -14);
    } else if (exponent == 31) {
        val = mantissa ? NAN : INFINITY;
    } else {
        val = ldexpf(1.0f + static_cast<float>(mantissa) / 1024.0f, static_cast<int>(exponent) - 15);
    }
    return sign ? -val : val;
}

// float -> fp16 手动转换 (round-to-nearest-even)
static uint16_t FloatToFp16(float val)
{
    if (val != val)
        return 0x7E00; // NaN
    if (val == 0.0f)
        return 0;
    if (val == INFINITY)
        return 0x7C00;
    if (val == -INFINITY)
        return 0xFC00;

    uint32_t sign = 0;
    if (val < 0) {
        sign = 1;
        val = -val;
    }

    int exp;
    float frac = frexpf(val, &exp); // frac in [0.5, 1), exp is unbiased+1
    // fp16: bias=15, exponent range [1,30], denorm when exp<=0
    int fp16Exp = exp - 1 + 15;             // unbiased exp = exp-1
    float mantissaF = (frac - 0.5f) * 2.0f; // mantissa in [0, 1)
    uint32_t mantissa = static_cast<uint32_t>(roundf(mantissaF * 1024.0f));

    if (mantissa >= 1024) {
        mantissa = 0;
        fp16Exp++;
    }
    if (fp16Exp <= 0) {
        // denorm
        mantissa = static_cast<uint32_t>(roundf(ldexpf(val, 14 - 0) * 1024.0f / 2.0f));
        if (mantissa >= 1024)
            mantissa = 1023;
        fp16Exp = 0;
    } else if (fp16Exp > 30) {
        return static_cast<uint16_t>(sign << 15 | 0x7C00); // inf
    }

    return static_cast<uint16_t>(sign << 15 | (fp16Exp << 10) | (mantissa & 0x3FF));
}

// CSC 矩阵计算 (与 kernel 的 ApplyCscMatrix 一致)
static void GoldenApplyCscMatrix(uint8_t& outR, uint8_t& outG, uint8_t& outB, uint8_t ch0, uint8_t ch1, uint8_t ch2,
                                 const CscParam& csc)
{
    int16_t t0 = static_cast<int16_t>(ch0) - csc.inBias0;
    int16_t t1 = static_cast<int16_t>(ch1) - csc.inBias1;
    int16_t t2 = static_cast<int16_t>(ch2) - csc.inBias2;

    int16_t r = static_cast<int16_t>(roundf(
        static_cast<float>(csc.cscMatrix00 * t0 * 2 + csc.cscMatrix01 * t1 * 2 + csc.cscMatrix02 * t2 * 2 + 1) / 512));
    int16_t g = static_cast<int16_t>(roundf(
        static_cast<float>(csc.cscMatrix10 * t0 * 2 + csc.cscMatrix11 * t1 * 2 + csc.cscMatrix12 * t2 * 2 + 1) / 512));
    int16_t b = static_cast<int16_t>(roundf(
        static_cast<float>(csc.cscMatrix20 * t0 * 2 + csc.cscMatrix21 * t1 * 2 + csc.cscMatrix22 * t2 * 2 + 1) / 512));

    outR = static_cast<uint8_t>(max(0, min(255, r + csc.outBias0)));
    outG = static_cast<uint8_t>(max(0, min(255, g + csc.outBias1)));
    outB = static_cast<uint8_t>(max(0, min(255, b + csc.outBias2)));
}

// DTC 计算 (与 kernel 的 DataConversion 一致, fp16 output)
// kernel: dst = (T(src - mean) - T(minVal)) * T(varReci), T=half
// 模拟 half 精度中间计算
static float GoldenDtc(uint8_t pixelVal, const DtcParam& dtc, int channelIndex)
{
    float mean = 0, minVal = 0, varReci = 1;
    if (channelIndex == 0) {
        mean = dtc.dtcPixelMeanChn0;
        minVal = dtc.dtcPixelMinChn0;
        varReci = dtc.dtcPixelVarReciChn0;
    } else if (channelIndex == 1) {
        mean = dtc.dtcPixelMeanChn1;
        minVal = dtc.dtcPixelMinChn1;
        varReci = dtc.dtcPixelVarReciChn1;
    } else if (channelIndex == 2) {
        mean = dtc.dtcPixelMeanChn2;
        minVal = dtc.dtcPixelMinChn2;
        varReci = dtc.dtcPixelVarReciChn2;
    } else {
        mean = dtc.dtcPixelMeanChn3;
        minVal = dtc.dtcPixelMinChn3;
        varReci = dtc.dtcPixelVarReciChn3;
    }

    // kernel: dst = (half(src - mean) - half(minVal)) * half(varReci)
    float afterMean = Fp16ToFloat(FloatToFp16(static_cast<float>(pixelVal) - mean));
    float afterMin = Fp16ToFloat(FloatToFp16(afterMean - minVal));
    float result = Fp16ToFloat(FloatToFp16(afterMin * varReci));
    return result;
}

// 生成 golden 数据，输出为 NCHW fp16 格式
// golden 大小: batchNum * outputChannelNum * outputSizeH * outputSizeW
// 计算输出索引（与kernel的RgbComputeDstIdx/GrayComputeDstIdx一致）
// NCHW: n*C*H*W + c*H*W + h*W + w, NHWC: n*H*W*C + h*W*C + w*C + c
static size_t ComputeOutputIdx(uint32_t n, uint32_t c, uint32_t h, uint32_t w, uint32_t H, uint32_t W,
                               uint32_t channelNum, uint8_t outputFormat)
{
    if (outputFormat == NCHW_FORMAT_INDEX) {
        return static_cast<size_t>(n) * channelNum * H * W + static_cast<size_t>(c) * H * W + h * W + w;
    } else {
        return static_cast<size_t>(n) * H * W * channelNum + static_cast<size_t>(h) * W * channelNum +
               static_cast<size_t>(w) * channelNum + c;
    }
}

// 生成 golden 数据，输出按outputFormat布局，值按输出数据类型存储
// isFp16Output: true=fp16输出, false=uint8输出
static vector<uint16_t> ComputeGolden(const uint8_t* images, const AippTilingData& tD, bool isFp16Output)
{
    uint32_t N = tD.batchNum;
    uint32_t H = tD.outputSizeH;
    uint32_t W = tD.outputSizeW;
    uint32_t C = tD.outputChannelNum;
    size_t totalPixels = static_cast<size_t>(N) * C * H * W;
    vector<uint16_t> golden(totalPixels, 0);

    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t h = 0; h < H; h++) {
            for (uint32_t w = 0; w < W; w++) {
                uint8_t cscR = 0, cscG = 0, cscB = 0;
                bool isPad = false;

                // Padding check
                if (tD.paddingParam.paddingSwitch != 0) {
                    isPad = (h < (uint32_t)tD.paddingParam.topPaddingSize) ||
                            (h >= (uint32_t)tD.paddingParam.topPaddingSize + tD.cropParam.cropSizeH) ||
                            (w < (uint32_t)tD.paddingParam.leftPaddingSize) ||
                            (w >= (uint32_t)tD.paddingParam.leftPaddingSize + tD.cropParam.cropSizeW);
                }

                if (isPad) {
                    float pv = tD.paddingParam.padValue;
                    uint16_t outVal = isFp16Output ?
                                          FloatToFp16(pv) :
                                          static_cast<uint16_t>(max(0, min(255, static_cast<int>(pv + 0.5f))));
                    for (uint32_t c = 0; c < C; c++) {
                        size_t idx = ComputeOutputIdx(n, c, h, w, H, W, C, tD.outputFormat);
                        golden[idx] = outVal;
                    }
                    continue;
                }

                // Compute crop coordinates
                uint32_t cropH = h - tD.paddingParam.topPaddingSize;
                uint32_t cropW = w - tD.paddingParam.leftPaddingSize;
                uint32_t srcH = tD.cropParam.cropStartPosH + cropH;
                uint32_t srcW = tD.cropParam.cropStartPosW + cropW;

                // Read input pixels based on imageFormat
                if (tD.imageFormat == IMAGE_FORMAT_MAP.at("RGB888_U8")) {
                    // RGB888: NHWC, 3 channels
                    size_t srcIdx = n * tD.inputSizeH * tD.inputSizeW * 3 + srcH * tD.inputSizeW * 3 + srcW * 3;
                    GoldenApplyCscMatrix(cscR, cscG, cscB, images[srcIdx], images[srcIdx + 1], images[srcIdx + 2],
                                         tD.cscParam);
                } else if (tD.imageFormat == IMAGE_FORMAT_MAP.at("XRGB8888_U8")) {
                    // XRGB8888: NHWC, 4 channels, srcChannelOffset determines start
                    size_t srcIdx = n * tD.inputSizeH * tD.inputSizeW * 4 + srcH * tD.inputSizeW * 4 + srcW * 4 +
                                    tD.srcChannelOffset;
                    GoldenApplyCscMatrix(cscR, cscG, cscB, images[srcIdx], images[srcIdx + 1], images[srcIdx + 2],
                                         tD.cscParam);
                } else if (tD.imageFormat == IMAGE_FORMAT_MAP.at("YUV420SP_U8")) {
                    // YUV420SP: Y plane then interleaved UV plane
                    size_t yPlaneSize = tD.inputSizeH * tD.inputSizeW;
                    size_t srcYIdx = n * yPlaneSize * 3 / 2 + srcH * tD.inputSizeW + srcW;
                    size_t srcUIdx = n * yPlaneSize * 3 / 2 + yPlaneSize + (srcH / 2) * tD.inputSizeW + (srcW & ~1u);
                    size_t srcVIdx = srcUIdx + 1;
                    GoldenApplyCscMatrix(cscR, cscG, cscB, images[srcYIdx], images[srcUIdx], images[srcVIdx],
                                         tD.cscParam);
                } else if (tD.imageFormat == IMAGE_FORMAT_MAP.at("YUV400_U8")) {
                    // YUV400: single Y channel, CSC with ch1=0, ch2=0
                    size_t srcIdx = n * tD.inputSizeH * tD.inputSizeW + srcH * tD.inputSizeW + srcW;
                    GoldenApplyCscMatrix(cscR, cscG, cscB, images[srcIdx], 0, 0, tD.cscParam);
                }

                // Gray detection: same logic as SetGrayFlag
                bool isGray = false;
                bool anyMatrix1NotZero = (tD.cscParam.cscMatrix10 != 0) || (tD.cscParam.cscMatrix11 != 0) ||
                                         (tD.cscParam.cscMatrix12 != 0);
                bool anyMatrix2NotZero = (tD.cscParam.cscMatrix20 != 0) || (tD.cscParam.cscMatrix21 != 0) ||
                                         (tD.cscParam.cscMatrix22 != 0);
                if (!anyMatrix1NotZero && !anyMatrix2NotZero) {
                    bool isRgbFmt = (tD.imageFormat == IMAGE_FORMAT_MAP.at("RGB888_U8") ||
                                     tD.imageFormat == IMAGE_FORMAT_MAP.at("XRGB8888_U8"));
                    bool isYuvFmt = (tD.imageFormat == IMAGE_FORMAT_MAP.at("YUV420SP_U8") ||
                                     tD.imageFormat == IMAGE_FORMAT_MAP.at("YUV400_U8"));
                    if (isRgbFmt && tD.cscParam.outBias0 == 0 && tD.cscParam.outBias1 == 0 &&
                        tD.cscParam.outBias2 == 0) {
                        isGray = true;
                    }
                    if (tD.imageFormat == IMAGE_FORMAT_MAP.at("YUV400_U8") &&
                        !static_cast<bool>(tD.cscParam.cscSwitch)) {
                        isGray = true;
                    }
                    if (tD.imageFormat == IMAGE_FORMAT_MAP.at("YUV420SP_U8") && tD.cscParam.cscMatrix01 == 0 &&
                        tD.cscParam.cscMatrix02 == 0 && tD.cscParam.inBias0 == 0 && tD.cscParam.inBias1 == 0 &&
                        tD.cscParam.inBias2 == 0) {
                        isGray = true;
                    }
                }

                // Apply DTC and write output
                // Gray: channel 1,2 are 0 (kernel writes DataConversion(0))
                // Non-gray: all 3 channels from CSC
                uint8_t chVals[3] = {cscR, isGray ? 0 : cscG, isGray ? 0 : cscB};
                for (uint32_t c = 0; c < C; c++) {
                    float dtcResult = GoldenDtc(chVals[c], tD.dtcParam, c);
                    uint16_t outVal = isFp16Output ?
                                          FloatToFp16(dtcResult) :
                                          static_cast<uint16_t>(max(0, min(255, static_cast<int>(roundf(dtcResult)))));
                    size_t idx = ComputeOutputIdx(n, c, h, w, H, W, C, tD.outputFormat);
                    golden[idx] = outVal;
                }
            }
        }
    }
    return golden;
}

// ========== End Golden ==========

// 检查输出buffer不全为0
static bool CheckOutputNonZero(uint8_t* data, size_t size)
{
    for (size_t i = 0; i < size; i++) {
        if (data[i] != 0) {
            return true;
        }
    }
    return false;
}

// 将输出写入文件，支持NCHW/NHWC布局和fp16/uint8数据类型
static void DumpOutputToFile(const std::string& testName, uint8_t* features, const AippTilingData& tD,
                             bool isFp16Output)
{
    std::string fileName = testName + "_output.txt";
    std::ofstream ofs(fileName);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << fileName << std::endl;
        return;
    }

    uint32_t C = tD.outputChannelNum;
    uint32_t H = tD.outputSizeH;
    uint32_t W = tD.outputSizeW;
    bool isNchw = (tD.outputFormat == NCHW_FORMAT_INDEX);

    ofs << "# AIPP kernel output: " << testName << std::endl;
    ofs << "# Format: " << (isNchw ? "NCHW" : "NHWC") << ", dtype=" << (isFp16Output ? "fp16" : "uint8")
        << ", N=" << tD.batchNum << " C=" << C << " H=" << H << " W=" << W << std::endl;

    for (uint32_t n = 0; n < tD.batchNum; n++) {
        for (uint32_t c = 0; c < C; c++) {
            ofs << "\n# N=" << n << " C=" << c << std::endl;
            for (uint32_t h = 0; h < H; h++) {
                for (uint32_t w = 0; w < W; w++) {
                    size_t idx = ComputeOutputIdx(n, c, h, w, H, W, C, tD.outputFormat);
                    if (isFp16Output) {
                        uint16_t* fp16Data = reinterpret_cast<uint16_t*>(features);
                        ofs << Fp16ToFloat(fp16Data[idx]);
                    } else {
                        ofs << (int)features[idx];
                    }
                    if (w < W - 1)
                        ofs << "\t";
                }
                ofs << "\n";
            }
        }
    }
    ofs.close();
    cout << "  Output dumped to: " << fileName << endl;
}

// 将输入image数据写入文件
static void DumpInputToFile(const std::string& testName, const uint8_t* images, const AippTilingData& tD)
{
    std::string fileName = testName + "_input.txt";
    std::ofstream ofs(fileName);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open input file: " << fileName << std::endl;
        return;
    }

    uint32_t N = tD.batchNum;
    uint32_t H = tD.inputSizeH;
    uint32_t W = tD.inputSizeW;
    float bpp = GetBytesPerPixel(tD.imageFormat);
    int channelsPerPixel = (tD.imageFormat == IMAGE_FORMAT_MAP.at("RGB888_U8"))   ? 3 :
                           (tD.imageFormat == IMAGE_FORMAT_MAP.at("XRGB8888_U8")) ? 4 :
                           (tD.imageFormat == IMAGE_FORMAT_MAP.at("YUV400_U8"))   ? 1 :
                                                                                    0;
    // YUV420SP special handling

    ofs << "# AIPP kernel input: " << testName << std::endl;
    ofs << "# imageFormat=" << (int)tD.imageFormat << " N=" << N << " H=" << H << " W=" << W << " bytesPerPixel=" << bpp
        << std::endl;

    for (uint32_t n = 0; n < N; n++) {
        ofs << "\n# Batch N=" << n << std::endl;
        if (tD.imageFormat == IMAGE_FORMAT_MAP.at("YUV420SP_U8")) {
            // Y plane
            ofs << "## Y plane (H=" << H << " W=" << W << ")" << std::endl;
            for (uint32_t h = 0; h < H; h++) {
                for (uint32_t w = 0; w < W; w++) {
                    size_t idx = n * H * W * 3 / 2 + h * W + w;
                    ofs << (int)images[idx];
                    if (w < W - 1)
                        ofs << "\t";
                }
                ofs << "\n";
            }
            // UV plane (interleaved, half resolution)
            ofs << "## UV plane (H=" << H / 2 << " W=" << W << ", interleaved U V)" << std::endl;
            size_t yPlaneSize = H * W;
            for (uint32_t h = 0; h < H / 2; h++) {
                for (uint32_t w = 0; w < W; w += 2) {
                    size_t uIdx = n * yPlaneSize * 3 / 2 + yPlaneSize + h * W + w;
                    ofs << "U=" << (int)images[uIdx] << ",V=" << (int)images[uIdx + 1];
                    if (w < W - 2)
                        ofs << "\t";
                }
                ofs << "\n";
            }
        } else {
            // RGB888 / XRGB8888 / YUV400: NHWC layout
            ofs << "## NHWC layout, channelsPerPixel=" << channelsPerPixel << std::endl;
            for (uint32_t h = 0; h < H; h++) {
                for (uint32_t w = 0; w < W; w++) {
                    size_t baseIdx = n * H * W * channelsPerPixel + h * W * channelsPerPixel + w * channelsPerPixel;
                    if (channelsPerPixel == 1) {
                        ofs << (int)images[baseIdx];
                    } else if (channelsPerPixel == 3) {
                        ofs << "R=" << (int)images[baseIdx] << ",G=" << (int)images[baseIdx + 1]
                            << ",B=" << (int)images[baseIdx + 2];
                    } else if (channelsPerPixel == 4) {
                        ofs << "X=" << (int)images[baseIdx] << ",R=" << (int)images[baseIdx + 1]
                            << ",G=" << (int)images[baseIdx + 2] << ",B=" << (int)images[baseIdx + 3];
                    }
                    if (w < W - 1)
                        ofs << "\t";
                }
                ofs << "\n";
            }
        }
    }
    ofs.close();
    cout << "  Input dumped to: " << fileName << endl;
}

// 通用执行函数：执行tiling + 分配内存 + 运行kernel + 校验输出
static void RunAippKernelTest(const std::string& testName, const gert::TilingContextPara& tilingContextPara)
{
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_EQ(tilingRet, true);

    // 从tiling data解析AippTilingData
    AippTilingData aippTiling;
    memcpy(&aippTiling, tilingInfo.tilingData.get(), sizeof(AippTilingData));

    // 根据tiling结果计算实际输入输出字节数
    size_t inputByteSize = static_cast<size_t>(aippTiling.batchNum * aippTiling.inputSizeH * aippTiling.inputSizeW *
                                               GetBytesPerPixel(aippTiling.imageFormat));
    // 输出buffer按tiling data中的真实输出通道数分配：
    // gray场景C=1时kernel每像素只写1个位置，C=3时写3个位置
    bool isFp16Output = (sizeof(DTYPE_FEATURES) == sizeof(uint16_t));
    size_t outputElementSize = isFp16Output ? sizeof(uint16_t) : sizeof(uint8_t);
    size_t outputByteSize = static_cast<size_t>(aippTiling.batchNum) * aippTiling.outputChannelNum *
                            aippTiling.outputSizeH * aippTiling.outputSizeW * outputElementSize;
    size_t paramsByteSize = 1;

    cout << "  tilingKey=" << tilingInfo.tilingKey << " imageFormat=" << (int)aippTiling.imageFormat
         << " outputFormat=" << (int)aippTiling.outputFormat << " batchNum=" << aippTiling.batchNum
         << " channelNum=" << aippTiling.channelNum << " outputChannelNum=" << aippTiling.outputChannelNum
         << " inputSizeW=" << aippTiling.inputSizeW << " inputSizeH=" << aippTiling.inputSizeH
         << " outputSizeW=" << aippTiling.outputSizeW << " outputSizeH=" << aippTiling.outputSizeH
         << " inputBytes=" << inputByteSize << " outputBytes=" << outputByteSize << " blockNum=" << tilingInfo.blockNum
         << endl;

    // 分配内存
    uint8_t* images = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* params = (uint8_t*)AscendC::GmAlloc(paramsByteSize);
    uint8_t* features = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    ASSERT_NE(images, nullptr);
    ASSERT_NE(features, nullptr);

    uint32_t numBlocks = tilingInfo.blockNum;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    // 初始化输入数据
    // 默认：随机数初始化，泛化验证
    // 设置 AIPP_DETERMINISTIC=1 可使用固定值初始化（便于复现问题）
    if (IsDeterministicInit()) {
        // RGB/XRGB: R=(h*W+w)%256, G=((h*W+w)*3+50)%256, B=((h*W+w)*7+100)%256
        // YUV420SP: Y=(h*W+w)%256, U/V交替用不同公式
        // YUV400: Y=(h*W+w)%256
        uint32_t H = aippTiling.inputSizeH;
        uint32_t W = aippTiling.inputSizeW;
        for (uint32_t n = 0; n < aippTiling.batchNum; n++) {
            if (aippTiling.imageFormat == IMAGE_FORMAT_MAP.at("RGB888_U8")) {
                for (uint32_t h = 0; h < H; h++) {
                    for (uint32_t w = 0; w < W; w++) {
                        size_t idx = n * H * W * 3 + h * W * 3 + w * 3;
                        images[idx] = static_cast<uint8_t>((h * W + w) % 256);
                        images[idx + 1] = static_cast<uint8_t>(((h * W + w) * 3 + 50) % 256);
                        images[idx + 2] = static_cast<uint8_t>(((h * W + w) * 7 + 100) % 256);
                    }
                }
            } else if (aippTiling.imageFormat == IMAGE_FORMAT_MAP.at("XRGB8888_U8")) {
                for (uint32_t h = 0; h < H; h++) {
                    for (uint32_t w = 0; w < W; w++) {
                        size_t idx = n * H * W * 4 + h * W * 4 + w * 4;
                        images[idx] = 0; // X channel, unused
                        images[idx + 1] = static_cast<uint8_t>((h * W + w) % 256);
                        images[idx + 2] = static_cast<uint8_t>(((h * W + w) * 3 + 50) % 256);
                        images[idx + 3] = static_cast<uint8_t>(((h * W + w) * 7 + 100) % 256);
                    }
                }
            } else if (aippTiling.imageFormat == IMAGE_FORMAT_MAP.at("YUV420SP_U8")) {
                // Y plane
                for (uint32_t h = 0; h < H; h++) {
                    for (uint32_t w = 0; w < W; w++) {
                        size_t idx = n * H * W * 3 / 2 + h * W + w;
                        images[idx] = static_cast<uint8_t>((h * W + w) % 256);
                    }
                }
                // UV plane (interleaved)
                size_t yPlaneSize = H * W;
                for (uint32_t h = 0; h < H / 2; h++) {
                    for (uint32_t w = 0; w < W; w += 2) {
                        size_t uIdx = n * yPlaneSize * 3 / 2 + yPlaneSize + h * W + w;
                        images[uIdx] = static_cast<uint8_t>(((h * W / 2 + w / 2) * 3 + 50) % 256);
                        images[uIdx + 1] = static_cast<uint8_t>(((h * W / 2 + w / 2) * 7 + 100) % 256);
                    }
                }
            } else if (aippTiling.imageFormat == IMAGE_FORMAT_MAP.at("YUV400_U8")) {
                for (uint32_t h = 0; h < H; h++) {
                    for (uint32_t w = 0; w < W; w++) {
                        size_t idx = n * H * W + h * W + w;
                        images[idx] = static_cast<uint8_t>((h * W + w) % 256);
                    }
                }
            }
        }
    } else {
        srand(static_cast<unsigned>(time(nullptr)));
        for (size_t i = 0; i < inputByteSize; i++) {
            images[i] = static_cast<uint8_t>(rand() % 256);
        }
    }
    memset(features, 0, outputByteSize);

    // 拷贝tiling数据 & 设置tiling key
    memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    // 打印输入数据（需设置 AIPP_DUMP_IO=1）
    if (IsDumpEnabled()) {
        DumpInputToFile(testName, images, aippTiling);
    }

    // 运行kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(Aipp, numBlocks, images, params, features, workspace, tiling);

    // 校验输出不全为0，说明kernel确实执行了
    EXPECT_TRUE(CheckOutputNonZero(features, outputByteSize))
        << "Output buffer is all zeros, kernel may not have executed correctly";

    // 计算 golden 并对比
    vector<uint16_t> golden = ComputeGolden(images, aippTiling, isFp16Output);
    uint32_t H = aippTiling.outputSizeH;
    uint32_t W = aippTiling.outputSizeW;
    size_t totalPixels = static_cast<size_t>(aippTiling.batchNum) * aippTiling.outputChannelNum * H * W;

    int mismatchCount = 0;
    int comparedCount = 0;
    float maxDiff = 0.0f;
    for (size_t i = 0; i < totalPixels; i++) {
        comparedCount++;
        float kernelVal, goldenVal;
        if (isFp16Output) {
            uint16_t* fp16Data = reinterpret_cast<uint16_t*>(features);
            kernelVal = Fp16ToFloat(fp16Data[i]);
            goldenVal = Fp16ToFloat(golden[i]);
        } else {
            kernelVal = static_cast<float>(features[i]);
            goldenVal = static_cast<float>(golden[i]);
        }
        float diff = fabsf(kernelVal - goldenVal);
        if (diff > maxDiff)
            maxDiff = diff;
        if (diff > 0.1f)
            mismatchCount++;
    }
    cout << "  [" << testName << "] Golden compare: compared=" << comparedCount << "/" << totalPixels
         << " mismatches=" << mismatchCount << " maxDiff=" << maxDiff << endl;
    EXPECT_EQ(mismatchCount, 0) << testName << ": " << mismatchCount << "/" << comparedCount
                                << " pixels differ from golden (maxDiff=" << maxDiff << ")";

    // 将输出以矩阵形式写入文件（需设置 AIPP_DUMP_IO=1）
    if (IsDumpEnabled()) {
        DumpOutputToFile(testName, features, aippTiling, isFp16Output);
    }

    // 释放内存
    AscendC::GmFree(images);
    AscendC::GmFree(params);
    AscendC::GmFree(features);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// 动态AIPP gray测试执行函数：params携带运行时CSC参数（灰度矩阵），golden按合并运行时参数后的tilingData计算
// batch参数全0（dtc varReci=fp16(1.0)），与kernel侧UpdateRealPara/resetRealPara/UpdateDynamicBatchPara语义一致
static void RunAippDynamicGrayKernelTest(const std::string& testName, const gert::TilingContextPara& tilingContextPara,
                                         const tagAippDynamicParaHeader& header)
{
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    ASSERT_EQ(tilingRet, true);

    AippTilingData aippTiling;
    memcpy(&aippTiling, tilingInfo.tilingData.get(), sizeof(AippTilingData));

    AippTilingData runtimeTiling = aippTiling;
    runtimeTiling.imageFormat = header.inputFormat;
    runtimeTiling.batchNum = static_cast<uint32_t>(header.batchNum);
    runtimeTiling.inputSizeW = static_cast<uint32_t>(header.srcImageSizeW);
    runtimeTiling.inputSizeH = static_cast<uint32_t>(header.srcImageSizeH);
    runtimeTiling.cscParam.cscSwitch = header.cscSwitch;
    runtimeTiling.cscParam.rbuvSwapSwitch = header.rbuvSwapSwitch;
    runtimeTiling.cscParam.axSwapSwitch = header.axSwapSwitch;
    runtimeTiling.cscParam.cscMatrix00 = header.cscMatrixR0C0;
    runtimeTiling.cscParam.cscMatrix01 = header.cscMatrixR0C1;
    runtimeTiling.cscParam.cscMatrix02 = header.cscMatrixR0C2;
    runtimeTiling.cscParam.cscMatrix10 = header.cscMatrixR1C0;
    runtimeTiling.cscParam.cscMatrix11 = header.cscMatrixR1C1;
    runtimeTiling.cscParam.cscMatrix12 = header.cscMatrixR1C2;
    runtimeTiling.cscParam.cscMatrix20 = header.cscMatrixR2C0;
    runtimeTiling.cscParam.cscMatrix21 = header.cscMatrixR2C1;
    runtimeTiling.cscParam.cscMatrix22 = header.cscMatrixR2C2;
    runtimeTiling.cscParam.inBias0 = 0;
    runtimeTiling.cscParam.inBias1 = 0;
    runtimeTiling.cscParam.inBias2 = 0;
    runtimeTiling.cscParam.outBias0 = static_cast<int16_t>(header.cscOutputBiasR0);
    runtimeTiling.cscParam.outBias1 = static_cast<int16_t>(header.cscOutputBiasR1);
    runtimeTiling.cscParam.outBias2 = static_cast<int16_t>(header.cscOutputBiasR2);
    runtimeTiling.cropParam = {};
    runtimeTiling.paddingParam = {};
    runtimeTiling.dtcParam = {};

    uint32_t batchNum = static_cast<uint32_t>(header.batchNum);
    size_t paramsByteSize = sizeof(tagAippDynamicParaHeader) + batchNum * sizeof(kAippDynamicBatchPara);
    size_t inputByteSize = static_cast<size_t>(runtimeTiling.batchNum * runtimeTiling.inputSizeH *
                                               runtimeTiling.inputSizeW * GetBytesPerPixel(runtimeTiling.imageFormat));
    bool isFp16Output = (sizeof(DTYPE_FEATURES) == sizeof(uint16_t));
    size_t outputElementSize = isFp16Output ? sizeof(uint16_t) : sizeof(uint8_t);
    size_t outputByteSize = static_cast<size_t>(runtimeTiling.batchNum) * runtimeTiling.outputChannelNum *
                            runtimeTiling.outputSizeH * runtimeTiling.outputSizeW * outputElementSize;

    cout << "  tilingKey=" << tilingInfo.tilingKey << " inputBytes=" << inputByteSize
         << " outputBytes=" << outputByteSize << " blockNum=" << tilingInfo.blockNum << endl;

    uint8_t* images = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* features = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    ASSERT_NE(images, nullptr);
    ASSERT_NE(features, nullptr);

    uint8_t* params = (uint8_t*)AscendC::GmAlloc(paramsByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    ASSERT_NE(params, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    memset(params, 0, paramsByteSize);
    tagAippDynamicParaHeader* headerBuf = reinterpret_cast<tagAippDynamicParaHeader*>(params);
    *headerBuf = header;
    kAippDynamicBatchPara* batchParas = reinterpret_cast<kAippDynamicBatchPara*>(params +
                                                                                 sizeof(tagAippDynamicParaHeader));
    for (uint32_t i = 0; i < batchNum; i++) {
        batchParas[i].dtcPixelVarReciChn0 = 0x3C00; // fp16(1.0)
        batchParas[i].dtcPixelVarReciChn1 = 0x3C00;
        batchParas[i].dtcPixelVarReciChn2 = 0x3C00;
        batchParas[i].dtcPixelVarReciChn3 = 0x3C00;
    }

    srand(static_cast<unsigned>(time(nullptr)));
    for (size_t i = 0; i < inputByteSize; i++) {
        images[i] = static_cast<uint8_t>(rand() % 256);
    }
    memset(features, 0, outputByteSize);

    memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    uint32_t numBlocks = tilingInfo.blockNum;
    ICPU_RUN_KF(Aipp, numBlocks, images, params, features, workspace, tiling);

    EXPECT_TRUE(CheckOutputNonZero(features, outputByteSize))
        << "Output buffer is all zeros, kernel may not have executed correctly";

    vector<uint16_t> golden = ComputeGolden(images, runtimeTiling, isFp16Output);
    uint32_t H = runtimeTiling.outputSizeH;
    uint32_t W = runtimeTiling.outputSizeW;
    size_t totalPixels = static_cast<size_t>(runtimeTiling.batchNum) * runtimeTiling.outputChannelNum * H * W;

    int mismatchCount = 0;
    float maxDiff = 0.0f;
    for (size_t i = 0; i < totalPixels; i++) {
        float kernelVal;
        float goldenVal;
        if (isFp16Output) {
            uint16_t* fp16Data = reinterpret_cast<uint16_t*>(features);
            kernelVal = Fp16ToFloat(fp16Data[i]);
            goldenVal = Fp16ToFloat(golden[i]);
        } else {
            kernelVal = static_cast<float>(features[i]);
            goldenVal = static_cast<float>(golden[i]);
        }
        float diff = fabsf(kernelVal - goldenVal);
        if (diff > maxDiff) {
            maxDiff = diff;
        }
        if (diff > 0.1f) {
            mismatchCount++;
        }
    }
    cout << "  [" << testName << "] Golden compare: mismatches=" << mismatchCount << "/" << totalPixels
         << " maxDiff=" << maxDiff << endl;
    EXPECT_EQ(mismatchCount, 0) << testName << ": " << mismatchCount << "/" << totalPixels
                                << " pixels differ from golden (maxDiff=" << maxDiff << ")";

    AscendC::GmFree(images);
    AscendC::GmFree(features);
    AscendC::GmFree(params);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// TilingKey=1: RGB pass-through (RGB888_U8 -> FP16 NCHW)
TEST_F(aipp_kernel_test, test_rgb_pass_through_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 3, 64, 64}, {1, 3, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr(
            "aipp_config_path",
            Ops::Cv::AnyValue::CreateFrom<string>(R"({"aipp_mode":"static","input_format":"RGB888_U8"})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_rgb_pass_through_fp16", tilingContextPara);
}

// TilingKey=2: YUV pass-through (YUV420SP_U8 -> FP16 NCHW)
TEST_F(aipp_kernel_test, test_yuv_pass_through_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 3, 64, 64}, {1, 3, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr(
            "aipp_config_path",
            Ops::Cv::AnyValue::CreateFrom<string>(R"({"aipp_mode":"static","input_format":"YUV420SP_U8"})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_yuv_pass_through_fp16", tilingContextPara);
}

// TilingKey=2 + odd crop: YUV420SP pass-through (YUV444 output) with 63x63 crop,
// regression for edge-unwritten bug in SimtComputeYuv420ToYuv444
TEST_F(aipp_kernel_test, test_yuv_pass_through_odd_crop_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 3, 63, 63}, {1, 3, 63, 63}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr(
            "aipp_config_path",
            Ops::Cv::AnyValue::CreateFrom<string>(R"({"aipp_mode":"static","input_format":"YUV420SP_U8","crop":true,)"
                                                  R"("crop_size_h":63,"crop_size_w":63})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_yuv_pass_through_odd_crop_fp16", tilingContextPara);
}

// TilingKey=3: RGB-to-YUV (RGB888_U8 with csc -> FP16 NCHW)
TEST_F(aipp_kernel_test, test_rgb_to_yuv_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 3, 64, 64}, {1, 3, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
                                         Ops::Cv::AnyValue::CreateFrom<string>(
                                             R"({"aipp_mode":"static","input_format":"RGB888_U8","csc_switch":true,)"
                                             R"("matrix_r0c0":256,"matrix_r0c1":101,"matrix_r0c2":-202,)"
                                             R"("matrix_r1c0":301,"matrix_r1c1":-256,"matrix_r1c2":402,)"
                                             R"("matrix_r2c0":503,"matrix_r2c1":601,"matrix_r2c2":256,)"
                                             R"("output_bias_0":110,"output_bias_1":120,"output_bias_2":83})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_rgb_to_yuv_fp16", tilingContextPara);
}

// TilingKey=4: RGB-to-Gray (XRGB8888_U8 with csc -> FP16 NCHW, 1 channel output)
TEST_F(aipp_kernel_test, test_rgb_to_gray_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 4}, {1, 64, 64, 4}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
                                         Ops::Cv::AnyValue::CreateFrom<string>(
                                             R"({"aipp_mode":"static","input_format":"XRGB8888_U8","csc_switch":true,)"
                                             R"("matrix_r0c0":76,"matrix_r0c1":150,"matrix_r0c2":30,)"
                                             R"("matrix_r1c0":0,"matrix_r1c1":0,"matrix_r1c2":0,)"
                                             R"("matrix_r2c0":0,"matrix_r2c1":0,"matrix_r2c2":0,)"
                                             R"("output_bias_0":0,"output_bias_1":0,"output_bias_2":0,)"
                                             R"("ax_swap_switch":true})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_rgb_to_gray_fp16", tilingContextPara);
}

// TilingKey=5: YUV-to-RGB (YUV420SP_U8 with csc -> FP16 NCHW)
TEST_F(aipp_kernel_test, test_yuv_to_rgb_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 3, 64, 64}, {1, 3, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
                                         Ops::Cv::AnyValue::CreateFrom<string>(
                                             R"({"aipp_mode":"static","input_format":"YUV420SP_U8","csc_switch":true,)"
                                             R"("matrix_r0c0":298,"matrix_r0c1":0,"matrix_r0c2":409,)"
                                             R"("matrix_r1c0":298,"matrix_r1c1":-100,"matrix_r1c2":-208,)"
                                             R"("matrix_r2c0":298,"matrix_r2c1":516,"matrix_r2c2":0,)"
                                             R"("input_bias_0":16,"input_bias_1":128,"input_bias_2":128,)"
                                             R"("rbuv_swap_switch":true})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_yuv_to_rgb_fp16", tilingContextPara);
}

// TilingKey=5 + odd crop: YUV-to-RGB with 63x63 crop (odd H and W), regression for edge-unwritten bug
TEST_F(aipp_kernel_test, test_yuv_to_rgb_odd_crop_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 3, 63, 63}, {1, 3, 63, 63}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr(
            "aipp_config_path",
            Ops::Cv::AnyValue::CreateFrom<string>(R"({"aipp_mode":"static","input_format":"YUV420SP_U8","crop":true,)"
                                                  R"("crop_size_h":63,"crop_size_w":63,"csc_switch":true,)"
                                                  R"("matrix_r0c0":298,"matrix_r0c1":0,"matrix_r0c2":409,)"
                                                  R"("matrix_r1c0":298,"matrix_r1c1":-100,"matrix_r1c2":-208,)"
                                                  R"("matrix_r2c0":298,"matrix_r2c1":516,"matrix_r2c2":0,)"
                                                  R"("input_bias_0":16,"input_bias_1":128,"input_bias_2":128})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_yuv_to_rgb_odd_crop_fp16", tilingContextPara);
}

// TilingKey=5 + odd crop (H only odd): 63x64 crop, last row unwritten before fix
TEST_F(aipp_kernel_test, test_yuv_to_rgb_odd_crop_h_only_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 3, 63, 64}, {1, 3, 63, 64}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr(
            "aipp_config_path",
            Ops::Cv::AnyValue::CreateFrom<string>(R"({"aipp_mode":"static","input_format":"YUV420SP_U8","crop":true,)"
                                                  R"("crop_size_h":63,"crop_size_w":64,"csc_switch":true,)"
                                                  R"("matrix_r0c0":298,"matrix_r0c1":0,"matrix_r0c2":409,)"
                                                  R"("matrix_r1c0":298,"matrix_r1c1":-100,"matrix_r1c2":-208,)"
                                                  R"("matrix_r2c0":298,"matrix_r2c1":516,"matrix_r2c2":0,)"
                                                  R"("input_bias_0":16,"input_bias_1":128,"input_bias_2":128})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_yuv_to_rgb_odd_crop_h_only_fp16", tilingContextPara);
}

// TilingKey=5 + odd crop (W only odd): 64x63 crop, last column unwritten before fix
TEST_F(aipp_kernel_test, test_yuv_to_rgb_odd_crop_w_only_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 3, 64, 63}, {1, 3, 64, 63}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr(
            "aipp_config_path",
            Ops::Cv::AnyValue::CreateFrom<string>(R"({"aipp_mode":"static","input_format":"YUV420SP_U8","crop":true,)"
                                                  R"("crop_size_h":64,"crop_size_w":63,"csc_switch":true,)"
                                                  R"("matrix_r0c0":298,"matrix_r0c1":0,"matrix_r0c2":409,)"
                                                  R"("matrix_r1c0":298,"matrix_r1c1":-100,"matrix_r1c2":-208,)"
                                                  R"("matrix_r2c0":298,"matrix_r2c1":516,"matrix_r2c2":0,)"
                                                  R"("input_bias_0":16,"input_bias_1":128,"input_bias_2":128})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_yuv_to_rgb_odd_crop_w_only_fp16", tilingContextPara);
}

// TilingKey=5 + odd crop with even start offset: load_start_pos (4,6), crop 59x61
TEST_F(aipp_kernel_test, test_yuv_to_rgb_odd_crop_offset_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 3, 59, 57}, {1, 3, 59, 57}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr(
            "aipp_config_path",
            Ops::Cv::AnyValue::CreateFrom<string>(R"({"aipp_mode":"static","input_format":"YUV420SP_U8","crop":true,)"
                                                  R"("load_start_pos_h":4,"load_start_pos_w":6,)"
                                                  R"("crop_size_h":59,"crop_size_w":57,"csc_switch":true,)"
                                                  R"("matrix_r0c0":298,"matrix_r0c1":0,"matrix_r0c2":409,)"
                                                  R"("matrix_r1c0":298,"matrix_r1c1":-100,"matrix_r1c2":-208,)"
                                                  R"("matrix_r2c0":298,"matrix_r2c1":516,"matrix_r2c2":0,)"
                                                  R"("input_bias_0":16,"input_bias_1":128,"input_bias_2":128})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_yuv_to_rgb_odd_crop_offset_fp16", tilingContextPara);
}

// TilingKey=5 + odd crop + batch 2: verifies nIdx decomposition with ceil block grid
TEST_F(aipp_kernel_test, test_yuv_to_rgb_odd_crop_batch2_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{2, 64, 64, 3}, {2, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{2, 3, 63, 63}, {2, 3, 63, 63}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr(
            "aipp_config_path",
            Ops::Cv::AnyValue::CreateFrom<string>(R"({"aipp_mode":"static","input_format":"YUV420SP_U8","crop":true,)"
                                                  R"("crop_size_h":63,"crop_size_w":63,"csc_switch":true,)"
                                                  R"("matrix_r0c0":298,"matrix_r0c1":0,"matrix_r0c2":409,)"
                                                  R"("matrix_r1c0":298,"matrix_r1c1":-100,"matrix_r1c2":-208,)"
                                                  R"("matrix_r2c0":298,"matrix_r2c1":516,"matrix_r2c2":0,)"
                                                  R"("input_bias_0":16,"input_bias_1":128,"input_bias_2":128})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_yuv_to_rgb_odd_crop_batch2_fp16", tilingContextPara);
}

// TilingKey=5 + odd crop + even padding: output 67x71, skipped lanes interplay with padding
TEST_F(aipp_kernel_test, test_yuv_to_rgb_odd_crop_padding_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    // input: 1x64x64x3(YUV420SP), crop 63x63, padding l/r=4 t/b=2 -> output: 1x3x67x71
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 3, 67, 71}, {1, 3, 67, 71}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr(
            "aipp_config_path",
            Ops::Cv::AnyValue::CreateFrom<string>(R"({"aipp_mode":"static","input_format":"YUV420SP_U8","crop":true,)"
                                                  R"("crop_size_h":63,"crop_size_w":63,"csc_switch":true,)"
                                                  R"("matrix_r0c0":298,"matrix_r0c1":0,"matrix_r0c2":409,)"
                                                  R"("matrix_r1c0":298,"matrix_r1c1":-100,"matrix_r1c2":-208,)"
                                                  R"("matrix_r2c0":298,"matrix_r2c1":516,"matrix_r2c2":0,)"
                                                  R"("input_bias_0":16,"input_bias_1":128,"input_bias_2":128,)"
                                                  R"("padding":true,"left_padding_size":4,"right_padding_size":4,)"
                                                  R"("top_padding_size":2,"bottom_padding_size":2,)"
                                                  R"("padding_value":0.5})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_yuv_to_rgb_odd_crop_padding_fp16", tilingContextPara);
}

// TilingKey=6: YUV-to-Gray (YUV400_U8 -> FP16 NCHW, 1 channel output)
TEST_F(aipp_kernel_test, test_yuv_to_gray_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 1}, {1, 64, 64, 1}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr(
            "aipp_config_path",
            Ops::Cv::AnyValue::CreateFrom<string>(R"({"aipp_mode":"static","input_format":"YUV400_U8"})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_yuv_to_gray_fp16", tilingContextPara);
}

// ========== Padding 测试用例 ==========
// input 64x64, padding: left=4, top=2, right=4, bottom=2
// output: H=64+2+2=68, W=64+4+4=72

// TilingKey=1 + padding: RGB pass-through with padding
TEST_F(aipp_kernel_test, test_rgb_pass_through_padding_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    // input: 1x64x64x3, output: 1x3x68x72 (64+2+2=68, 64+4+4=72)
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 3, 68, 72}, {1, 3, 68, 72}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr(
            "aipp_config_path",
            Ops::Cv::AnyValue::CreateFrom<string>(R"({"aipp_mode":"static","input_format":"RGB888_U8",)"
                                                  R"("padding":true,"left_padding_size":4,"right_padding_size":4,)"
                                                  R"("top_padding_size":2,"bottom_padding_size":2,)"
                                                  R"("padding_value":128.0})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_rgb_pass_through_padding_fp16", tilingContextPara);
}

// TilingKey=5 + padding: YUV-to-RGB CSC with padding
TEST_F(aipp_kernel_test, test_yuv_to_rgb_padding_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    // input: 1x64x64x3(YUV420SP), output: 1x3x68x72
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 3, 68, 72}, {1, 3, 68, 72}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
                                         Ops::Cv::AnyValue::CreateFrom<string>(
                                             R"({"aipp_mode":"static","input_format":"YUV420SP_U8","csc_switch":true,)"
                                             R"("matrix_r0c0":298,"matrix_r0c1":0,"matrix_r0c2":409,)"
                                             R"("matrix_r1c0":298,"matrix_r1c1":-100,"matrix_r1c2":-208,)"
                                             R"("matrix_r2c0":298,"matrix_r2c1":516,"matrix_r2c2":0,)"
                                             R"("input_bias_0":16,"input_bias_1":128,"input_bias_2":128,)"
                                             R"("rbuv_swap_switch":true,)"
                                             R"("padding":true,"left_padding_size":4,"right_padding_size":4,)"
                                             R"("top_padding_size":2,"bottom_padding_size":2,)"
                                             R"("padding_value":0.5})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_yuv_to_rgb_padding_fp16", tilingContextPara);
}

// TilingKey=4 + padding: RGB-to-Gray with padding
TEST_F(aipp_kernel_test, test_rgb_to_gray_padding_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    // input: 1x64x64x4(XRGB8888), output: 1x1x68x72
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 4}, {1, 64, 64, 4}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 1, 68, 72}, {1, 1, 68, 72}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
                                         Ops::Cv::AnyValue::CreateFrom<string>(
                                             R"({"aipp_mode":"static","input_format":"XRGB8888_U8","csc_switch":true,)"
                                             R"("matrix_r0c0":76,"matrix_r0c1":150,"matrix_r0c2":30,)"
                                             R"("matrix_r1c0":0,"matrix_r1c1":0,"matrix_r1c2":0,)"
                                             R"("matrix_r2c0":0,"matrix_r2c1":0,"matrix_r2c2":0,)"
                                             R"("output_bias_0":0,"output_bias_1":0,"output_bias_2":0,)"
                                             R"("ax_swap_switch":true,)"
                                             R"("padding":true,"left_padding_size":4,"right_padding_size":4,)"
                                             R"("top_padding_size":2,"bottom_padding_size":2,)"
                                             R"("padding_value":1.0})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_rgb_to_gray_padding_fp16", tilingContextPara);
}

// ========== NHWC 输出测试用例 ==========

// RGB pass-through, NHWC fp16输出
TEST_F(aipp_kernel_test, test_rgb_pass_through_nhwc_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_FLOAT16, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr(
            "aipp_config_path",
            Ops::Cv::AnyValue::CreateFrom<string>(R"({"aipp_mode":"static","input_format":"RGB888_U8"})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_rgb_pass_through_nhwc_fp16", tilingContextPara);
}

// YUV-to-RGB CSC, NHWC fp16输出
TEST_F(aipp_kernel_test, test_yuv_to_rgb_nhwc_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_FLOAT16, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
                                         Ops::Cv::AnyValue::CreateFrom<string>(
                                             R"({"aipp_mode":"static","input_format":"YUV420SP_U8","csc_switch":true,)"
                                             R"("matrix_r0c0":298,"matrix_r0c1":0,"matrix_r0c2":409,)"
                                             R"("matrix_r1c0":298,"matrix_r1c1":-100,"matrix_r1c2":-208,)"
                                             R"("matrix_r2c0":298,"matrix_r2c1":516,"matrix_r2c2":0,)"
                                             R"("input_bias_0":16,"input_bias_1":128,"input_bias_2":128,)"
                                             R"("rbuv_swap_switch":true})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_yuv_to_rgb_nhwc_fp16", tilingContextPara);
}

// RGB-to-Gray, NHWC fp16输出
TEST_F(aipp_kernel_test, test_rgb_to_gray_nhwc_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 4}, {1, 64, 64, 4}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 64, 64, 1}, {1, 64, 64, 1}}, ge::DT_FLOAT16, ge::FORMAT_NHWC}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
                                         Ops::Cv::AnyValue::CreateFrom<string>(
                                             R"({"aipp_mode":"static","input_format":"XRGB8888_U8","csc_switch":true,)"
                                             R"("matrix_r0c0":76,"matrix_r0c1":150,"matrix_r0c2":30,)"
                                             R"("matrix_r1c0":0,"matrix_r1c1":0,"matrix_r1c2":0,)"
                                             R"("matrix_r2c0":0,"matrix_r2c1":0,"matrix_r2c2":0,)"
                                             R"("output_bias_0":0,"output_bias_1":0,"output_bias_2":0,)"
                                             R"("ax_swap_switch":true})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_rgb_to_gray_nhwc_fp16", tilingContextPara);
}

// RGB-to-Gray, 3通道NCHW输出（兼容C=3的存量用法，kernel保留三通道写入）
TEST_F(aipp_kernel_test, test_rgb_to_gray_3channel_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 4}, {1, 64, 64, 4}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 3, 64, 64}, {1, 3, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
                                         Ops::Cv::AnyValue::CreateFrom<string>(
                                             R"({"aipp_mode":"static","input_format":"XRGB8888_U8","csc_switch":true,)"
                                             R"("matrix_r0c0":76,"matrix_r0c1":150,"matrix_r0c2":30,)"
                                             R"("matrix_r1c0":0,"matrix_r1c1":0,"matrix_r1c2":0,)"
                                             R"("matrix_r2c0":0,"matrix_r2c1":0,"matrix_r2c2":0,)"
                                             R"("output_bias_0":0,"output_bias_1":0,"output_bias_2":0,)"
                                             R"("ax_swap_switch":true})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_rgb_to_gray_3channel_fp16", tilingContextPara);
}

// TilingKey=6: YUV420SP-to-Gray with CSC (YUV420SP_U8 -> FP16 NCHW, 1 channel output)
// 覆盖C=1分支的YUV420SP源索引（nIdx*H*W*3/2跨batch偏移）
TEST_F(aipp_kernel_test, test_yuv420sp_to_gray_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path",
                                         Ops::Cv::AnyValue::CreateFrom<string>(
                                             R"({"aipp_mode":"static","input_format":"YUV420SP_U8","csc_switch":true,)"
                                             R"("matrix_r0c0":256,"matrix_r0c1":0,"matrix_r0c2":0,)"
                                             R"("matrix_r1c0":0,"matrix_r1c1":0,"matrix_r1c2":0,)"
                                             R"("matrix_r2c0":0,"matrix_r2c1":0,"matrix_r2c2":0,)"
                                             R"("input_bias_0":0,"input_bias_1":0,"input_bias_2":0})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_yuv420sp_to_gray_fp16", tilingContextPara);
}

// TilingKey=6: YUV400-to-Gray 3通道输出（存量兼容路径，kernel按三通道写入，通道1/2写0）
TEST_F(aipp_kernel_test, test_yuv400_to_gray_3channel_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 1}, {1, 64, 64, 1}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 3, 64, 64}, {1, 3, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr(
            "aipp_config_path",
            Ops::Cv::AnyValue::CreateFrom<string>(R"({"aipp_mode":"static","input_format":"YUV400_U8"})"))},
        &compileInfo, socVersion);

    RunAippKernelTest("test_yuv400_to_gray_3channel_fp16", tilingContextPara);
}

// 动态AIPP gray端到端：tiling key=100，运行时下发RGB灰度CSC矩阵，kernel重派发到RGB-to-Gray（C=1单通道写入）
TEST_F(aipp_kernel_test, test_dynamic_rgb_to_gray_fp16)
{
    AippCompileInfo compileInfo = {56, 253952};
    string socVersion = "Ascend950";
    gert::TilingContextPara tilingContextPara(
        "Aipp",
        {{{{1, 64, 64, 3}, {1, 64, 64, 3}}, ge::DT_UINT8, ge::FORMAT_NHWC}, {{{1}, {1}}, ge::DT_UINT8, ge::FORMAT_ND}},
        {{{{1, 1, 64, 64}, {1, 1, 64, 64}}, ge::DT_FLOAT16, ge::FORMAT_NCHW}},
        {gert::TilingContextPara::OpAttr("aipp_config_path", Ops::Cv::AnyValue::CreateFrom<string>(
                                                                 R"({"aipp_mode":"dynamic","related_input_rank":0,)"
                                                                 R"("max_src_image_size":12288})"))},
        &compileInfo, socVersion);

    tagAippDynamicParaHeader header = {};
    header.inputFormat = IMAGE_FORMAT_MAP.at("RGB888_U8");
    header.cscSwitch = 1;
    header.rbuvSwapSwitch = 0;
    header.axSwapSwitch = 0;
    header.batchNum = 1;
    header.srcImageSizeW = 64;
    header.srcImageSizeH = 64;
    header.cscMatrixR0C0 = 76;
    header.cscMatrixR0C1 = 150;
    header.cscMatrixR0C2 = 30;
    header.cscMatrixR1C0 = 0;
    header.cscMatrixR1C1 = 0;
    header.cscMatrixR1C2 = 0;
    header.cscMatrixR2C0 = 0;
    header.cscMatrixR2C1 = 0;
    header.cscMatrixR2C2 = 0;
    header.cscOutputBiasR0 = 0;
    header.cscOutputBiasR1 = 0;
    header.cscOutputBiasR2 = 0;

    RunAippDynamicGrayKernelTest("test_dynamic_rgb_to_gray_fp16", tilingContextPara, header);
}
