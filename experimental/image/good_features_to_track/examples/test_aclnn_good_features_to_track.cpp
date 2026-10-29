/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <unistd.h>
#include <sys/wait.h>
#include "acl/acl.h"
#include "aclnn_good_features_to_track.h"

#define CHECK_RET(cond, ret)                                                                                    \
    do {                                                                                                        \
        if (!(cond)) {                                                                                          \
            std::cerr << "CHECK_RET failed: " << #cond << ", ret=" << (ret) << ", err=" << aclGetRecentErrMsg() \
                      << std::endl;                                                                             \
            return (ret);                                                                                       \
        }                                                                                                       \
    } while (0)

// ---------- 数据类型 / 图像生成 ----------

// fp32 -> fp16 (IEEE 754, round-to-nearest-even 的简化实现)
static uint16_t FloatToHalf(float f)
{
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000U;
    const int32_t exp = static_cast<int32_t>((x >> 23) & 0xffU) - 127 + 15;
    const uint32_t mant = x & 0x7fffffU;
    if (((x >> 23) & 0xffU) == 0xffU) { // inf / nan
        return static_cast<uint16_t>(sign | 0x7c00U | (mant ? 0x0200U : 0U));
    }
    if (exp >= 31) { // 溢出 -> inf
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
    if (exp <= 0) { // 次正规/下溢 -> 0
        return static_cast<uint16_t>(sign);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

// 合成图像 1：多个独立亮色方块 -> 若干清晰角点
static void GenSyntheticImage(std::vector<float>& img, int64_t h, int64_t w,
                              const std::vector<std::pair<int64_t, int64_t>>& squares, int64_t sq)
{
    img.assign(static_cast<size_t>(h) * static_cast<size_t>(w), 0.0f);
    for (const auto& sq0 : squares) {
        const int64_t y0 = sq0.first;
        const int64_t x0 = sq0.second;
        for (int64_t y = y0; y < std::min(y0 + sq, h); ++y) {
            for (int64_t x = x0; x < std::min(x0 + sq, w); ++x) {
                img[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = 255.0f;
            }
        }
    }
}

// 合成图像 2：棋盘格 -> 大量角点
static void GenCheckerboard(std::vector<float>& img, int64_t h, int64_t w, int64_t cell)
{
    img.assign(static_cast<size_t>(h) * static_cast<size_t>(w), 0.0f);
    for (int64_t y = 0; y < h; ++y) {
        for (int64_t x = 0; x < w; ++x) {
            const int64_t cy = y / cell;
            const int64_t cx = x / cell;
            img[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = ((cy + cx) & 1) ? 255.0f :
                                                                                                              0.0f;
        }
    }
}

// 合成图像 3：中心单点 -> 恰好 1 个角点
static void GenSinglePoint(std::vector<float>& img, int64_t h, int64_t w, int64_t py, int64_t px)
{
    img.assign(static_cast<size_t>(h) * static_cast<size_t>(w), 0.0f);
    img[static_cast<size_t>(py) * static_cast<size_t>(w) + static_cast<size_t>(px)] = 255.0f;
}

// 合成图像 4：纯背景（无角点）
static void GenEmptyImage(std::vector<float>& img, int64_t h, int64_t w)
{
    img.assign(static_cast<size_t>(h) * static_cast<size_t>(w), 0.0f);
}

// 掩码：除 [y0,y1)x[x0,x1) 外全 0，用于验证 mask 抑制
static void GenMask(std::vector<uint8_t>& mask, int64_t h, int64_t w, int64_t y0, int64_t y1, int64_t x0, int64_t x1)
{
    mask.assign(static_cast<size_t>(h) * static_cast<size_t>(w), 0);
    for (int64_t y = y0; y < std::min(y1, h); ++y) {
        for (int64_t x = x0; x < std::min(x1, w); ++x) {
            mask[static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)] = 255;
        }
    }
}

// ---------- Tensor 创建 ----------

// 通用输入 tensor 创建（支持 fp16/fp32/u8/i32）
static int CreateTensor(const void* data, int64_t numel, aclDataType dtype, const std::vector<int64_t>& shape,
                        aclTensor** tensor, void** addr)
{
    size_t elemSize = 4;
    if (dtype == aclDataType::ACL_FLOAT16) {
        elemSize = 2;
    } else if (dtype == aclDataType::ACL_UINT8) {
        elemSize = 1;
    } else if (dtype == aclDataType::ACL_INT32) {
        elemSize = 4;
    }
    auto ret = aclrtMalloc(addr, static_cast<size_t>(numel) * elemSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, ret);
    ret = aclrtMemcpy(*addr, static_cast<size_t>(numel) * elemSize, data, static_cast<size_t>(numel) * elemSize,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, ret);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, aclFormat::ACL_FORMAT_ND, shape.data(),
                              shape.size(), *addr);
    CHECK_RET(*tensor != nullptr, ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

// corners 输出 tensor（恒 FLOAT，shape=[N, capacity, 2]）
static int CreateOutputTensor(const std::vector<int64_t>& shape, aclTensor** tensor, void** addr)
{
    int64_t numel = 1;
    for (auto d : shape) {
        numel *= d;
    }
    auto ret = aclrtMalloc(addr, static_cast<size_t>(numel) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, ret);
    *tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_FLOAT, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *addr);
    CHECK_RET(*tensor != nullptr, ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

// num_corners 输出 tensor（恒 INT32，shape=[N]）
static int CreateCountTensor(const std::vector<int64_t>& shape, aclTensor** tensor, void** addr)
{
    int64_t numel = 1;
    for (auto d : shape) {
        numel *= d;
    }
    auto ret = aclrtMalloc(addr, static_cast<size_t>(numel) * sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, ret);
    *tensor = aclCreateTensor(shape.data(), shape.size(), aclDataType::ACL_INT32, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *addr);
    CHECK_RET(*tensor != nullptr, ACL_ERROR_FAILURE);
    return ACL_SUCCESS;
}

// ---------- 测试参数 ----------

struct Params {
    int64_t maxCorners = 1000;
    double qualityLevel = 0.01;
    double minDistance = 1.0;
    int64_t blockSize = 3;
    bool useHarris = false;
    double k = 0.04;
};

// 单个用例描述（注册式，便于后续增删）
struct TestCase {
    std::string name;
    std::vector<float> image; // fp32 数据（fp16 时内部转换）
    aclDataType dtype = aclDataType::ACL_FLOAT;
    std::vector<int64_t> shape;
    std::vector<uint8_t> mask; // hasMask=false 时不使用
    bool hasMask = false;
    Params params;
    int64_t expectMin = 0;
    int64_t expectMax = 1000;
};

struct TestResult {
    std::string name;
    std::string dtype;
    bool passed = false;
    int32_t numCorners = -1;
    std::vector<int32_t> counts; // 逐图角点数（batch>1 时用于逐图断言）
    int64_t expectedMin = 0;
    int64_t expectedMax = 0;
};

static std::vector<TestResult> g_results;

// 执行单次 GoodFeaturesToTrack，返回逐图角点数与整块 corners
static int RunGoodFeatures(aclrtStream stream, const void* imageData, aclDataType dtype,
                           const std::vector<int64_t>& imageShape, const void* maskData, const Params& params,
                           std::vector<float>& corners, std::vector<int32_t>& counts, int32_t& numCorners)
{
    const int64_t n = imageShape[0];
    const int64_t h = imageShape[1];
    const int64_t w = imageShape[2];

    aclTensor *imageTensor = nullptr, *maskTensor = nullptr, *cornersTensor = nullptr, *countTensor = nullptr;
    void *imageAddr = nullptr, *maskAddr = nullptr, *cornersAddr = nullptr, *countAddr = nullptr;

    const int64_t numel = n * h * w;
    auto ret = CreateTensor(imageData, numel, dtype, imageShape, &imageTensor, &imageAddr);
    CHECK_RET(ret == ACL_SUCCESS, ret);

    const std::vector<int64_t> cornersShape = {n, params.maxCorners, 2};
    ret = CreateOutputTensor(cornersShape, &cornersTensor, &cornersAddr);
    CHECK_RET(ret == ACL_SUCCESS, ret);

    const std::vector<int64_t> countShape = {n};
    ret = CreateCountTensor(countShape, &countTensor, &countAddr);
    CHECK_RET(ret == ACL_SUCCESS, ret);

    if (maskData != nullptr) {
        ret = CreateTensor(maskData, numel, aclDataType::ACL_UINT8, imageShape, &maskTensor, &maskAddr);
        CHECK_RET(ret == ACL_SUCCESS, ret);
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnGoodFeaturesToTrackGetWorkspaceSize(imageTensor, maskTensor, params.maxCorners, params.qualityLevel,
                                                   params.minDistance, params.blockSize, params.useHarris, params.k,
                                                   cornersTensor, countTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, ret);

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, ret);
    }
    ret = aclnnGoodFeaturesToTrack(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, ret);

    counts.assign(static_cast<size_t>(n), 0);
    ret = aclrtMemcpy(counts.data(), static_cast<size_t>(n) * sizeof(int32_t), countAddr,
                      static_cast<size_t>(n) * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, ret);
    numCorners = counts[0];

    // 逐图校验需要每张图自己的角点：整块搬回 [n, maxCorners, 2]（测试规模下很小）
    const size_t cornersElems = static_cast<size_t>(n) * static_cast<size_t>(params.maxCorners) * 2U;
    corners.assign(cornersElems, 0.0f);
    if (cornersElems > 0U) {
        ret = aclrtMemcpy(corners.data(), cornersElems * sizeof(float), cornersAddr, cornersElems * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS, ret);
    }

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyTensor(imageTensor);
    aclDestroyTensor(cornersTensor);
    aclDestroyTensor(countTensor);
    if (maskTensor != nullptr) {
        aclDestroyTensor(maskTensor);
    }
    aclrtFree(imageAddr);
    aclrtFree(cornersAddr);
    aclrtFree(countAddr);
    if (maskAddr != nullptr) {
        aclrtFree(maskAddr);
    }
    return ACL_SUCCESS;
}

// 在子进程内运行单个用例（完整 ACL 环境：context + stream，运行后销毁）
// 返回后 res 填充结果
static void RunOneCase(const TestCase& tc, TestResult& res)
{
    res.name = tc.name;
    res.dtype = (tc.dtype == aclDataType::ACL_FLOAT16) ? "fp16" : "fp32";
    res.expectedMin = tc.expectMin;
    res.expectedMax = tc.expectMax;

    if (aclrtSetDevice(0) != ACL_SUCCESS) {
        res.passed = false;
        return;
    }
    aclrtContext context = nullptr;
    if (aclrtCreateContext(&context, 0) != ACL_SUCCESS) {
        res.passed = false;
        return;
    }
    aclrtSetCurrentContext(context);
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        aclrtDestroyContext(context);
        res.passed = false;
        return;
    }

    std::vector<float> corners;
    std::vector<int32_t> counts;
    int32_t numCorners = 0;
    std::vector<uint16_t> imageFp16;
    const void* imageData = nullptr;
    if (tc.dtype == aclDataType::ACL_FLOAT16) {
        imageFp16.resize(tc.image.size());
        for (size_t i = 0; i < tc.image.size(); ++i) {
            imageFp16[i] = FloatToHalf(tc.image[i]);
        }
        imageData = imageFp16.data();
    } else {
        imageData = tc.image.data();
    }
    const void* maskData = tc.hasMask ? tc.mask.data() : nullptr;

    const int ret = RunGoodFeatures(stream, imageData, tc.dtype, tc.shape, maskData, tc.params, corners, counts,
                                    numCorners);
    aclrtDestroyStream(stream);
    aclrtDestroyContext(context);
    if (ret != ACL_SUCCESS) {
        res.numCorners = -1;
        res.passed = false;
        return;
    }

    res.numCorners = numCorners;
    res.counts = counts;
    const int64_t h = tc.shape[1];
    const int64_t w = tc.shape[2];
    const int64_t n = tc.shape[0];
    const int64_t cap = tc.params.maxCorners;
    // 逐图断言：每张图的角点数都要在期望区间内，且各自的坐标都在界内
    // （batch>1 时若阈值错用全局 R_max，弱纹理图会被漏检，这里能直接暴露）
    bool ok = true;
    for (int64_t img = 0; img < n; ++img) {
        const int32_t cnt = (img < static_cast<int64_t>(counts.size())) ? counts[img] : 0;
        if (cnt < 0 || cnt > cap || cnt < tc.expectMin || cnt > tc.expectMax) {
            ok = false;
        }
        const size_t base = static_cast<size_t>(img) * static_cast<size_t>(cap) * 2U;
        for (int32_t i = 0; i < cnt; ++i) {
            if (base + static_cast<size_t>(i) * 2U + 1U >= corners.size()) {
                ok = false;
                break;
            }
            // 输出按 (x, y) 排列（对齐 OpenCV cv::goodFeaturesToTrack）
            const float x = corners[base + static_cast<size_t>(i) * 2U];
            const float y = corners[base + static_cast<size_t>(i) * 2U + 1U];
            if (y < 0 || y >= h || x < 0 || x >= w || !std::isfinite(y) || !std::isfinite(x)) {
                ok = false;
            }
        }
    }
    if (n > 1) {
        std::cout << "  [per-image] " << tc.name << ":";
        for (int64_t img = 0; img < n; ++img) {
            std::cout << " img" << img << "=" << (img < static_cast<int64_t>(counts.size()) ? counts[img] : -1);
        }
        std::cout << std::endl;
    }
    // 对照转储：打印第 0 张图的角点坐标（GFT_DUMP_CORNERS=1），用于新旧实现逐位比对
    if (std::getenv("GFT_DUMP_CORNERS") != nullptr) {
        std::cout << "CORNERS|" << tc.name << "|n=" << numCorners << "|";
        for (int32_t i = 0; i < numCorners; ++i) {
            std::cout << corners[static_cast<size_t>(i) * 2U] << "," << corners[static_cast<size_t>(i) * 2U + 1U]
                      << ";";
        }
        std::cout << std::endl;
    }
    res.passed = ok;
}

int main()
{
    // 父进程只负责构建用例列表并 fork；ACL 初始化在各自子进程内完成（进程级隔离）

    // ---- 基础测试图（一次生成，供各用例复用）----
    constexpr int64_t kH = 64;
    constexpr int64_t kW = 64;
    constexpr int64_t kSq = 8;
    std::vector<float> squaresImg; // 4 个方块
    GenSyntheticImage(squaresImg, kH, kW, {{4, 4}, {4, 40}, {40, 4}, {40, 40}}, kSq);
    std::vector<float> checkerImg; // 棋盘格
    GenCheckerboard(checkerImg, kH, kW, 8);
    // 256x256 棋盘格：更大画布下的密角点；同时作为 tests/opencv_crosscheck 的对拍输入
    // （numpy 可逐位重建，见该目录 README）
    constexpr int64_t kBigH = 256;
    std::vector<float> checkerBigImg;
    GenCheckerboard(checkerBigImg, kBigH, kBigH, 8);
    std::vector<float> singleImg; // 中心单点
    GenSinglePoint(singleImg, kH, kW, 32, 32);
    std::vector<float> emptyImg; // 空图
    GenEmptyImage(emptyImg, kH, kW);
    std::vector<float> rectImg; // 非方形 128x64
    GenSyntheticImage(rectImg, 128, 64, {{8, 8}, {8, 40}, {72, 8}, {72, 40}}, 8);
    std::vector<float> batch2; // batch=2（方块 + 单点）
    batch2.insert(batch2.end(), squaresImg.begin(), squaresImg.end());
    batch2.insert(batch2.end(), singleImg.begin(), singleImg.end());
    std::vector<uint8_t> maskBR; // 仅右下区域有效
    GenMask(maskBR, kH, kW, 32, 64, 32, 64);

    const std::vector<int64_t> sqShape = {1, kH, kW};
    const std::vector<int64_t> rectShape = {1, 128, 64};
    const std::vector<int64_t> b2Shape = {2, kH, kW};

    // ---- 用例注册表（每行一个用例，便于增删）----
    std::vector<TestCase> cases;
    auto addCase = [&](const std::string& name, const std::vector<float>& img, aclDataType dt,
                       const std::vector<int64_t>& shape, const std::vector<uint8_t>* mask, const Params& p, int64_t lo,
                       int64_t hi) {
        TestCase tc;
        tc.name = name;
        tc.image = img;
        tc.dtype = dt;
        tc.shape = shape;
        if (mask != nullptr) {
            tc.mask = *mask;
            tc.hasMask = true;
        }
        tc.params = p;
        tc.expectMin = lo;
        tc.expectMax = hi;
        cases.push_back(std::move(tc));
    };

    Params p5;
    p5.blockSize = 5;
    Params p7;
    p7.blockSize = 7;
    Params pH;
    pH.useHarris = true;
    Params pMax;
    pMax.maxCorners = 10;
    Params pDist;
    pDist.minDistance = 10.0;
    Params pBig; // 256x256 棋盘格对拍用：不限制 maxCorners，避免被输出容量截断
    pBig.maxCorners = 20000;

    // dtype / 形状
    // 4 个独立 8x8 方块 -> 每块 4 角，共 16 角点（numpy Shi-Tomasi 参考验证）
    addCase("fp32 方块 blockSize=3", squaresImg, aclDataType::ACL_FLOAT, sqShape, nullptr, Params{}, 10, 20);
    addCase("fp16 方块 blockSize=3", squaresImg, aclDataType::ACL_FLOAT16, sqShape, nullptr, Params{}, 10, 20);
    addCase("fp32 128x64 非方形", rectImg, aclDataType::ACL_FLOAT, rectShape, nullptr, Params{}, 1, 20);
    addCase("fp32 棋盘格", checkerImg, aclDataType::ACL_FLOAT, sqShape, nullptr, Params{}, 20, 80);
    addCase("fp32 棋盘格 256x256", checkerBigImg, aclDataType::ACL_FLOAT, {1, kBigH, kBigH}, nullptr, pBig, 950, 1200);
    addCase("fp32 单点", singleImg, aclDataType::ACL_FLOAT, sqShape, nullptr, Params{}, 1, 1);
    addCase("fp32 空图", emptyImg, aclDataType::ACL_FLOAT, sqShape, nullptr, Params{}, 0, 0);
    // 参数
    addCase("fp32 方块 blockSize=5", squaresImg, aclDataType::ACL_FLOAT, sqShape, nullptr, p5, 10, 20);
    addCase("fp32 方块 blockSize=7", squaresImg, aclDataType::ACL_FLOAT, sqShape, nullptr, p7, 10, 40);
    addCase("fp32 方块 Harris", squaresImg, aclDataType::ACL_FLOAT, sqShape, nullptr, pH, 10, 20);
    addCase("fp32 方块 maxCorners=10", squaresImg, aclDataType::ACL_FLOAT, sqShape, nullptr, pMax, 1, 10);
    addCase("fp32 方块 minDistance=10", squaresImg, aclDataType::ACL_FLOAT, sqShape, nullptr, pDist, 1, 5);
    // 批量 / 掩码
    addCase("fp32 batch=2", batch2, aclDataType::ACL_FLOAT, b2Shape, nullptr, Params{}, 1, 20);
    addCase("fp32 方块+mask 右下区", squaresImg, aclDataType::ACL_FLOAT, sqShape, &maskBR, Params{}, 1, 5);
    // ---- 宽画布用例：覆盖 W 方向多分块（UB 快路径在宽图上才启用）----
    // 图案 = 亮方块 on 黑底：角点响应孤立、对浮点累加顺序鲁棒，可做紧断言。
    // 旧实现在 w>1456(fp32) 时整体掉出 UB 快路径（6.7~9.7× 惩罚），这组用例专门回归该路径，
    // 并覆盖「同一核内多个行块」与「多核行带切分」两种分块结构。
    constexpr int64_t kWidthH = 64;
    constexpr int64_t kWidthSq = 8;
    {
        // w=1408：32B 对齐且单行 5632B < 8160B → 必走 UB 快路径
        std::vector<float> img;
        GenSyntheticImage(img, kWidthH, 1408,
                          {{4, 4}, {4, 480}, {4, 960}, {4, 1392}, {40, 4}, {40, 480}, {40, 960}, {40, 1392}}, kWidthSq);
        addCase("fp32 宽画布 W=1408", img, aclDataType::ACL_FLOAT, {1, kWidthH, 1408}, nullptr, Params{}, 32, 32);
    }
    {
        // w=2040：单行 8160B（旧实现的块拷贝上限，现由上界证伪；见 W=2048 用例）
        std::vector<float> img;
        GenSyntheticImage(img, kWidthH, 2040,
                          {{4, 4},
                           {4, 300},
                           {4, 600},
                           {4, 900},
                           {4, 1200},
                           {4, 1500},
                           {4, 1800},
                           {4, 2024},
                           {40, 4},
                           {40, 300},
                           {40, 600},
                           {40, 900},
                           {40, 1200},
                           {40, 1500},
                           {40, 1800},
                           {40, 2024}},
                          kWidthSq);
        addCase("fp32 宽画布 W=2040", img, aclDataType::ACL_FLOAT, {1, kWidthH, 2040}, nullptr, Params{}, 64, 64);
    }
    {
        // w=2048（2K DCI）：单行 8192B，超过旧的 kMaxBlockBytes=255*32=8160 ⇒ 旧实现整体掉到
        // GM 回退路径（实测 2040→2.19 ns/px、2041→46.6 ns/px，21.7×）。`DataCopyPad` 的
        // DataCopyParams.blockLen 实为字节单位、上限 65535，修正后该宽度由 UB 预算约束
        // （r=1 时 fp32 上限 w≈2275），仍走 UB 快路径。本用例把这条刚被启用的路径钉住。
        std::vector<float> img;
        GenSyntheticImage(img, kWidthH, 2048,
                          {{4, 4},
                           {4, 300},
                           {4, 600},
                           {4, 900},
                           {4, 1200},
                           {4, 1500},
                           {4, 1800},
                           {4, 2024},
                           {40, 4},
                           {40, 300},
                           {40, 600},
                           {40, 900},
                           {40, 1200},
                           {40, 1500},
                           {40, 1800},
                           {40, 2024}},
                          kWidthSq);
        addCase("fp32 宽画布 W=2048(2K)", img, aclDataType::ACL_FLOAT, {1, kWidthH, 2048}, nullptr, Params{}, 64, 64);
    }
    // ---- 宽度等价性用例：图案水平重复 N 次 ⇒ 角点数必须是 N 倍 ----
    // 这是「宽度无关性」的强断言：同一个孤立方块图案，无论图像多宽，其角点响应必须不变。
    // 覆盖 fp32 >2275（超出现 UB 预算上沿）与 fp16 >2642 的宽度——这些宽度当前走 GM 回退路径、
    // 且没有任何其它测试覆盖。若日后实现「阶段 1 列分块」把这些宽度切到 UB 快路径，
    // 本组用例必须继续通过（即分块不得改变结果）。
    {
        // W=1024 基准：x∈{4,500} × y∈{4,40} ⇒ 4 个方块 ⇒ 16 个角点
        std::vector<float> img;
        GenSyntheticImage(img, kWidthH, 1024, {{4, 4}, {4, 500}, {40, 4}, {40, 500}}, kWidthSq);
        addCase("fp32 宽画布 W=1024(基准)", img, aclDataType::ACL_FLOAT, {1, kWidthH, 1024}, nullptr, Params{}, 16, 16);
    }
    {
        // W=2048 = 基准图案 ×2 ⇒ 必须恰好 32 个角点
        std::vector<float> img;
        GenSyntheticImage(img, kWidthH, 2048,
                          {{4, 4}, {4, 500}, {4, 1028}, {4, 1524}, {40, 4}, {40, 500}, {40, 1028}, {40, 1524}},
                          kWidthSq);
        addCase("fp32 宽画布 W=2048(图案×2)", img, aclDataType::ACL_FLOAT, {1, kWidthH, 2048}, nullptr, Params{}, 32,
                32);
    }
    {
        // W=4096 = 基准图案 ×4 ⇒ 恰好 64 个角点。
        // w=4096 远超声预算上沿（fp32 2275）⇒ 当前走 GM 回退路径；列分块实施后应切到 UB 且结果不变。
        std::vector<float> img;
        GenSyntheticImage(img, kWidthH, 4096,
                          {{4, 4},
                           {4, 500},
                           {4, 1028},
                           {4, 1524},
                           {4, 2052},
                           {4, 2548},
                           {4, 3076},
                           {4, 3572},
                           {40, 4},
                           {40, 500},
                           {40, 1028},
                           {40, 1524},
                           {40, 2052},
                           {40, 2548},
                           {40, 3076},
                           {40, 3572}},
                          kWidthSq);
        addCase("fp32 宽画布 W=4096(图案×4)", img, aclDataType::ACL_FLOAT, {1, kWidthH, 4096}, nullptr, Params{}, 64,
                64);
    }
    {
        // fp16 W=3072 = 基准图案 ×3 ⇒ 恰好 48 个角点（3072 > fp16 预算上沿 2642 ⇒ GM 回退路径）
        std::vector<float> img;
        GenSyntheticImage(img, kWidthH, 3072,
                          {{4, 4},
                           {4, 500},
                           {4, 1028},
                           {4, 1524},
                           {4, 2052},
                           {4, 2548},
                           {40, 4},
                           {40, 500},
                           {40, 1028},
                           {40, 1524},
                           {40, 2052},
                           {40, 2548}},
                          kWidthSq);
        addCase("fp16 宽画布 W=3072(图案×3)", img, aclDataType::ACL_FLOAT16, {1, kWidthH, 3072}, nullptr, Params{}, 48,
                48);
    }
    {
        // w=1400：宽且 32B 对齐，覆盖「单核内多个行块 + 跨核行带」组合
        // 注：w % 8 != 0（fp32）的宽度走「UB 行尾补齐」路径（ubStride_ = align32(行字节)/sizeof(T)），
        //     见下方 W=1366 / W=1500 用例。
        std::vector<float> img;
        GenSyntheticImage(img, kWidthH, 1400,
                          {{4, 4},
                           {4, 300},
                           {4, 600},
                           {4, 900},
                           {4, 1100},
                           {4, 1384},
                           {40, 4},
                           {40, 300},
                           {40, 600},
                           {40, 900},
                           {40, 1100},
                           {40, 1384}},
                          kWidthSq);
        addCase("fp32 宽画布 W=1400", img, aclDataType::ACL_FLOAT, {1, kWidthH, 1400}, nullptr, Params{}, 48, 48);
    }
    // ---- 非 8 对齐宽度用例：行字节非 32B 对齐，走「UB 行尾补齐」路径 ----
    // 覆盖实际常见分辨率（1366/1500 等），验证 ubStride_ = align32(行字节) 的索引与逐行拷贝一致。
    {
        // w=1366（1366 % 8 == 6）：单行 5464B → UB 行距补齐到 5472B（ubStride=1368）
        std::vector<float> img;
        GenSyntheticImage(
            img, kWidthH, 1366,
            {{4, 4}, {4, 300}, {4, 600}, {4, 900}, {4, 1200}, {40, 4}, {40, 300}, {40, 600}, {40, 900}, {40, 1200}},
            kWidthSq);
        addCase("fp32 宽画布 W=1366(非对齐)", img, aclDataType::ACL_FLOAT, {1, kWidthH, 1366}, nullptr, Params{}, 40,
                40);
    }
    {
        // w=1500（1500 % 8 == 4）：单行 6000B → 补齐到 6016B
        std::vector<float> img;
        GenSyntheticImage(img, kWidthH, 1500,
                          {{4, 4}, {4, 400}, {4, 800}, {4, 1200}, {40, 4}, {40, 400}, {40, 800}, {40, 1200}}, kWidthSq);
        addCase("fp32 宽画布 W=1500(非对齐)", img, aclDataType::ACL_FLOAT, {1, kWidthH, 1500}, nullptr, Params{}, 32,
                32);
    }
    {
        // fp16 非对齐：1500 % 16 == 12，行字节 3000B → 补齐到 3008B
        std::vector<float> img;
        GenSyntheticImage(img, kWidthH, 1500,
                          {{4, 4}, {4, 400}, {4, 800}, {4, 1200}, {40, 4}, {40, 400}, {40, 800}, {40, 1200}}, kWidthSq);
        addCase("fp16 宽画布 W=1500(非对齐)", img, aclDataType::ACL_FLOAT16, {1, kWidthH, 1500}, nullptr, Params{}, 32,
                32);
    }
    // ---- 执行：每用例 fork 子进程，进程级隔离（避免同进程内多次 aclnn 调用的状态污染）----
    int passed = 0;
    std::vector<int> failedIdx;
    for (size_t i = 0; i < cases.size(); ++i) {
        const pid_t pid = fork();
        if (pid < 0) {
            std::cout << "✗ fork 失败 @case " << i << "\n";
            failedIdx.push_back(static_cast<int>(i));
            continue;
        }
        if (pid == 0) {
            // 子进程：独立 ACL 环境，运行单个用例
            aclInit(nullptr);
            TestResult res;
            RunOneCase(cases[i], res);
            aclFinalize();
            std::cout << (res.passed ? "✓ " : "✗ ") << res.name << ": 角点数=" << res.numCorners << " (期望 ["
                      << res.expectedMin << "," << res.expectedMax << "])" << std::endl;
            _exit(res.passed ? 0 : 1);
        }
        int status = 0;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            ++passed;
        } else {
            failedIdx.push_back(static_cast<int>(i));
        }
    }

    // ---- 汇总 ----
    std::cout << "\n=== 汇总: " << passed << "/" << static_cast<int>(cases.size()) << " 通过 ===\n";
    if (!failedIdx.empty()) {
        std::cout << "失败用例索引: ";
        for (int idx : failedIdx) {
            std::cout << idx << "(" << cases[static_cast<size_t>(idx)].name << ") ";
        }
        std::cout << "\n";
    }

    return (passed == static_cast<int>(cases.size())) ? 0 : 1;
}
