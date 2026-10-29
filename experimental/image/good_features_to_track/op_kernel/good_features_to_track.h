/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __GOOD_FEATURES_TO_TRACK_H__
#define __GOOD_FEATURES_TO_TRACK_H__

#include "kernel_operator.h"
#include "good_features_to_track_tiling_data.h"

namespace NsGoodFeaturesToTrack {
using namespace AscendC;

// CPU 模拟(host)编译时，cpudebug 的 stub_fun.h 会在全局名字空间引入 min/max 的整型重载，
// 与 AscendC::min/max 一起解析 int32_t 实参时转换级相同而歧义。用精确签名的本地助手绕开，
// host 与 device 两侧语义完全一致（标量取小/取大），真机编译行为不变。
__aicore__ inline int32_t GftMin(int32_t a, int32_t b) { return a < b ? a : b; }
__aicore__ inline int32_t GftMax(int32_t a, int32_t b) { return a > b ? a : b; }

template <typename T>
class GoodFeaturesToTrack {
public:
    __aicore__ inline void Init(GM_ADDR image, GM_ADDR mask, GM_ADDR corners, GM_ADDR numCorners, GM_ADDR workspace,
                                const GoodFeaturesToTrackTilingData* tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline float Pixel(uint32_t image, int32_t y, int32_t x) const;
    __aicore__ inline float SobelGx(uint32_t image, int32_t y, int32_t x) const;
    __aicore__ inline float SobelGy(uint32_t image, int32_t y, int32_t x) const;
    __aicore__ inline float Response(uint32_t image, int32_t y, int32_t x) const;
    // 由窗口累加和（sumXX/sumYY/sumXY）计算最终响应（Shi-Tomasi），不含窗口读取
    __aicore__ inline float ResponseFromSum(float sumXX, float sumYY, float sumXY) const;
    __aicore__ inline bool IsLocalMaximum(uint32_t image, int32_t y, int32_t x, float value) const;
    __aicore__ inline bool FarEnough(uint32_t image, int32_t y, int32_t x, uint32_t selected) const;
    // 阶段 2a：全核并行收集候选（各自行带 → 各自分区）
    __aicore__ inline void CollectCandidates(uint32_t image, float maxResponse, LocalTensor<float> scanStage,
                                             int32_t scanStageRows, uint32_t coreIdx, uint32_t coreNum, float coreMax);
    // 阶段 2a 辅助：记录一个候选（UB 暂存 + 桶直方图 + 满块 DMA 写出）
    __aicore__ inline void PushCandidate(int32_t gy, int32_t x, float value, uint64_t bandCandBase, uint32_t& staged,
                                         uint32_t& localCount, LocalTensor<int32_t> stageIdx,
                                         LocalTensor<float> stageVal, LocalTensor<float> keyHist);
    // 阶段 2a 辅助：标量 3×3 局部极大判定（8 邻居全走 staging UB；图像外邻居跳过）
    __aicore__ inline bool IsLocalMaxStaged(LocalTensor<float> rLocal, int32_t gy, int32_t x, int32_t w, int32_t h,
                                            int32_t srcY0, float value) const;
    // 阶段 2b-A（仅 0 号核）：预填 -1、向量归约 H1(键 bit24-31) 定位 b*、算各核「小于组」偏移
    __aicore__ inline void TopKBuildMeta(uint32_t image, uint32_t coreNum);
    // 阶段 2b-B/D（全核并行）：mode=0 写出 bucket<b* 并统计 H2(bit16-23)；
    //                          mode=1 写出 bucket==b* 且 sub<=b**（二级细化，把保留集压回 ≈K）
    __aicore__ inline void TopKFilterOwn(uint32_t image, uint32_t coreIdx, uint32_t coreNum, uint32_t mode);
    // 阶段 2b-C（仅 0 号核）：向量归约 H2 定位 b**、算各核「等于组」偏移与保留总数
    __aicore__ inline void TopKRefineMeta(uint32_t image, uint32_t coreNum);
    // 阶段 2b-E（仅 0 号核）：对保留集（≈K）做基数排序 + 贪心选取
    __aicore__ inline void SelectFinal(uint32_t image, uint32_t coreNum);
    // 2b-A/2b-C 共用：按 mask 统计「每核计数」并写各核前缀偏移；maxPerCore >= 0 时给每核保留数加
    // 上限（前缀仍稠密），-1 表示不限制
    __aicore__ inline uint32_t PerCoreOffsets(GlobalTensor<float> histGm, uint32_t coreNum, LocalTensor<int32_t> meta,
                                              uint32_t metaBase, uint32_t offStart, int32_t maskHi,
                                              int32_t maxPerCore = -1);
    // 每核统计量槽：[(image * coreNum + core) * 8]，[0]=该图的局部 max 响应、[1]=该核在该图的候选数
    // （逐图阈值：每张图用自己的 R_max，与 cv::goodFeaturesToTrack 语义一致）
    __aicore__ inline uint64_t StatSlot(uint32_t image, uint32_t core) const;
    // 把某核在某张图上的局部 max 用 MTE3 写回统计槽（32B 单块）
    __aicore__ inline void WriteCoreMax(uint32_t image, uint32_t core, float value);
    // 求与该图行区间相交的核区间 [c0, c1]（按全局行等分公式推导，空行带的核不参与）
    __aicore__ inline void CoreRangeForImage(uint32_t image, uint32_t& c0, uint32_t& c1) const;
    // 用 MTE2 把某张图相关核的统计量搬到 UB；跨核读只能走 DMA
    __aicore__ inline LocalTensor<float> LoadCoreStats(uint32_t image);
    // 把 UB 暂存的 count 个候选用 MTE3 写到 dstBase（候选数组的扁平下标）
    __aicore__ inline void FlushCandidates(uint64_t dstBase, uint32_t count);
    // 候选排序键：float 位模式 → 单调 uint32（取反后按升序键即得响应降序）
    __aicore__ inline static uint32_t SortKeyOf(float value);
    // 基数排序单趟（8bit 稳定散列），src/dst 乒乓
    __aicore__ inline void RadixPass(GlobalTensor<float> srcVal, GlobalTensor<uint32_t> srcIdx,
                                     GlobalTensor<float> dstVal, GlobalTensor<uint32_t> dstIdx, uint64_t base,
                                     uint32_t count, int32_t shift, LocalTensor<int32_t> hist);

    GlobalTensor<T> imageGm_;
    GlobalTensor<uint8_t> maskGm_;
    GlobalTensor<float> cornersGm_;
    GlobalTensor<int32_t> numCornersGm_;
    GlobalTensor<float> responseGm_;
    const GoodFeaturesToTrackTilingData* tiling_;

    // 响应计算性能优化：图像行带 UB 缓存（LocalTensor 读取代标量 GM 读）
    TPipe pipe;
    TBuf<QuePosition::VECCALC> imageBuf_;
    LocalTensor<T> imgLocal_;
    bool ubCacheOn_ = false;
    int32_t ubImage_ = -1;   // 当前缓存的 image
    int32_t ubRowStart_ = 0; // UB 第 0 行对应的 image 内行号
    int32_t ubRows_ = 0;     // 缓存总行数（含 halo）
    // UB 中每行的元素跨度 = align32(行字节) / sizeof(T)（>= w）。
    // DataCopyPad 搬运时本就把不足 32B 的尾部补齐到 32B，因此 UB 行距天然是 align32(行字节)；
    // 所有 UB 内的行/列索引都必须用本值（而非 w），这样 w 非 8 对齐时也能走 UB 快路径。
    int32_t ubStride_ = 0;
    // UB 行距是否因 32B 补齐而大于原始行字节。为真时必须逐行 DMA 拷贝（把每行 UB 基址显式写成
    // r*ubStride_）；为假时可走整块多行拷贝（少一层拆分开销）。
    // 图像行与响应行要分开判断：fp16 下 w*2 可能非 32B 对齐，而同一列的 w*4 却对齐。
    bool imgRowsPadded_ = false;
    bool respRowsPadded_ = false;

    // 响应行带暂存缓冲：标量算完后用 DataCopyPad(DMA) 批量写回 GM，
    // DMA 走 MTE 管道保证跨核可见（标量 SetValue 写 GM 无跨核 cache 一致性保证）
    TBuf<QuePosition::VECCALC> respBuf_;

    // 梯度暂存缓冲：存每像素 gx²/gy²/gx*gy 三通道（每行 3*w 个 float，行数 = cacheRows）
    // 响应垂直窗口(vsum)只读行 ly∈[1, cacheRows-2]，首/尾行仅贴图像边界时需补算（见 Process）
    TBuf<QuePosition::VECCALC> gradBuf_;
    LocalTensor<float> gradLocal_; // gradBuf_ 的 float 视图
    int32_t gradRingRows_ = 1;     // 梯度环深度（= gradBuf_ 行数）；grad 行 py 存于 py % gradRingRows_
    int32_t gradValidHi_ = -1;     // 梯度环中已计算的最大 image 行（-1 = 尚未计算）
    bool useGradCache_ = false;    // 本次块是否使用梯度预计算（useUb=true 时）

    // 垂直窗口和缓冲（响应滑动窗口优化）：存 vsum = grad[y-r]+...+grad[y+r]（3 通道，单行）
    // 水平方向用滑动窗口 O(1) 递推，避免每像素 blockSize² 次窗口重读
    TBuf<QuePosition::VECCALC> vsumBuf_;

    // 每核局部 max 响应（DMA 写回，供 0 号核归约全局 max）
    GlobalTensor<float> maxPerCoreGm_;
    // 并行 TopK：每核候选键高 8 位直方图（256 个 fp32/核，阶段 2a 由各核自行统计并写出）。
    // 用 fp32 而非 int32：int32 的 ReduceSum 会退化到 fp16 的 vcadd（编译失败），
    // 而 Cast(int32→fp32) 路径实测会算错计数；计数 <= H*W <= 2^24，fp32 可精确表示。
    // （tiling 已强制 H*W <= 2^24，超出直接返回失败 ⇒ 该精确性前提成立。）
    GlobalTensor<float> keyHistGm_;
    // 并行 TopK 元数据：[0]=b*（第 K 小键所在桶），[1+c]=核 c 在保留集里的写出偏移（0 号核填）
    GlobalTensor<int32_t> topkMetaGm_;
    // 二级直方图（键 bit16-23，只统计 bucket==b* 的候选；每核 256 个 fp32）
    GlobalTensor<float> subHistGm_;
    // 候选列表：各核一次扫描收集自己行带的局部极大候选（索引 + 响应值），写入自己的分区
    GlobalTensor<uint32_t> candIdxGm_;
    GlobalTensor<int32_t> candIdxI32Gm_; // candIdxGm_ 的 int32 视图（仅供 DMA 使用）
    GlobalTensor<float> candValGm_;
    // 保留集缓冲：并行过滤的输出（各核写自己的偏移区间），随后作基数排序的乒乓
    GlobalTensor<uint32_t> candIdx2Gm_;
    GlobalTensor<int32_t> candIdx2I32Gm_;
    GlobalTensor<float> candVal2Gm_;
    // 基数排序直方图（256 桶）
    TBuf<QuePosition::VECCALC> histBuf_;
    // 本核候选键高 8 位直方图（256 桶，阶段 2a 累加，结束时 DMA 写出）
    TBuf<QuePosition::VECCALC> keyHistBuf_;
    // 二级直方图累加（256 桶，fp32）
    TBuf<QuePosition::VECCALC> subHistBuf_;
    // ReduceSum 的 fp32 结果（32 元素足够）与临时工作区
    TBuf<QuePosition::VECCALC> histRedBuf_;
    TBuf<QuePosition::VECCALC> histWorkBuf_;
    // 跨核元数据中转（布局见 TopKBuildMeta 注释，MTE2 读写）
    TBuf<QuePosition::VECCALC> metaBuf_;
    // 跨核统计量（每核 8 个 float：局部 max + 候选数）的 MTE2 中转缓冲。
    // 跨核生产者-消费者必须用 DMA 读写，标量访问 GM 可能命中本核陈旧缓存行 / 写入不落地。
    TBuf<QuePosition::VECCALC> statBuf_;
    // 候选 DMA 暂存（UB）：前半 kCandStage 个 uint32 存 idx，后半 kCandStage 个 float 存 val。
    // 候选是跨核数据，必须攒够一块用 MTE3 写出（标量 SetValue 写 GM 对其它核不可见）。
    TBuf<QuePosition::VECCALC> candStageBuf_;
    static constexpr uint32_t kCandStage = 1024U;
    // 跨核直方图归约（2b-A 键高 8 位 / 2b-C 二级）向量化：每核一行 256 个 fp32（= 1 KB，
    // 天然 32B 对齐，行内所有偏移都是 32B 倍数 ⇒ 可直接做向量 Add 的源偏移）。
    // 不要改成「8 核一组一次搬 8 KB」：一次搬多行会显著变慢。
    static constexpr uint32_t kHistRowElems = 256U;
    // 响应向量化：分块暂存 3 个窗口和（XX/YY/XY）+ 向量暂存（delta / trace / maxVec）
    TBuf<QuePosition::VECCALC> winBuf_;
    TBuf<QuePosition::VECCALC> winScrBuf_;
    static constexpr uint32_t kWinChunk = 256U;
    // Sobel 向量化：9 个向量槽（pA/pB/pC/cA/cC/nA/nC + gx/gy，pB 兼作临时）+ Gather 字节偏移表
    TBuf<QuePosition::VECCALC> sobVecBuf_;
    TBuf<QuePosition::VECCALC> sobOffBuf_;
};

template <typename T>
__aicore__ inline void GoodFeaturesToTrack<T>::Init(GM_ADDR image, GM_ADDR mask, GM_ADDR corners, GM_ADDR numCorners,
                                                    GM_ADDR workspace, const GoodFeaturesToTrackTilingData* tiling)
{
    tiling_ = tiling;
    imageGm_.SetGlobalBuffer((__gm__ T*)image, static_cast<uint64_t>(tiling->batch) * tiling->imageElements);
    if (tiling->hasMask != 0U) {
        maskGm_.SetGlobalBuffer((__gm__ uint8_t*)mask, static_cast<uint64_t>(tiling->batch) * tiling->imageElements);
    }
    cornersGm_.SetGlobalBuffer((__gm__ float*)corners,
                               static_cast<uint64_t>(tiling->batch) * tiling->outputCapacity * 2U);
    numCornersGm_.SetGlobalBuffer((__gm__ int32_t*)numCorners, tiling->batch);
    GM_ADDR userWs = GetUserWorkspace(workspace);
    const uint64_t elements = static_cast<uint64_t>(tiling->batch) * tiling->imageElements;
    const uint64_t respBytes = elements * sizeof(float);
    const uint64_t candBytes = elements * sizeof(uint32_t);
    const uint64_t maxBytes = static_cast<uint64_t>(tiling->batch) * tiling->coreNum * 8U * sizeof(float);
    responseGm_.SetGlobalBuffer((__gm__ float*)userWs, elements);
    maxPerCoreGm_.SetGlobalBuffer((__gm__ float*)(userWs + respBytes),
                                  static_cast<uint64_t>(tiling->batch) * tiling->coreNum * 8U);
    // 并行 TopK：每核候选键高 8 位直方图（256 int32/核）+ 元数据（[0]=b*，[1+c]=核 c 写出偏移）
    uint64_t off = respBytes + maxBytes;
    const uint64_t keyHistElems = static_cast<uint64_t>(tiling->coreNum) * 256U;
    keyHistGm_.SetGlobalBuffer((__gm__ float*)(userWs + off), keyHistElems);
    off += keyHistElems * sizeof(int32_t);
    // 元数据 2*coreNum+8 个 int32（= 8*coreNum+32 字节，天然 32B 对齐）。
    // 必须与 tiling 完全一致：少算一个元素就会让后续候选缓冲起点偏离 32B 对齐
    // （DataCopyPad 要求），表现为结果全 0 / 内核异常。
    const uint64_t metaElems = 2U * static_cast<uint64_t>(tiling->coreNum) + 8U;
    topkMetaGm_.SetGlobalBuffer((__gm__ int32_t*)(userWs + off), metaElems);
    off += metaElems * sizeof(int32_t);
    subHistGm_.SetGlobalBuffer((__gm__ float*)(userWs + off), static_cast<uint64_t>(tiling->coreNum) * 256U);
    off += static_cast<uint64_t>(tiling->coreNum) * 256U * sizeof(int32_t);
    candIdxGm_.SetGlobalBuffer((__gm__ uint32_t*)(userWs + off), elements);
    // 同一块内存的 int32 视图：AscendC 的 DataCopyPad 不支持 uint32_t，DMA 走这个视图
    candIdxI32Gm_.SetGlobalBuffer((__gm__ int32_t*)(userWs + off), elements);
    off += candBytes;
    candValGm_.SetGlobalBuffer((__gm__ float*)(userWs + off), elements);
    off += candBytes;
    // 保留集缓冲（并行过滤的输出；随后作为基数排序的乒乓）
    candIdx2Gm_.SetGlobalBuffer((__gm__ uint32_t*)(userWs + off), elements);
    candIdx2I32Gm_.SetGlobalBuffer((__gm__ int32_t*)(userWs + off), elements);
    off += candBytes;
    candVal2Gm_.SetGlobalBuffer((__gm__ float*)(userWs + off), elements);
}

template <typename T>
__aicore__ inline float GoodFeaturesToTrack<T>::Pixel(uint32_t image, int32_t y, int32_t x) const
{
    if (ubCacheOn_ && image == static_cast<uint32_t>(ubImage_)) {
        const int32_t ly = y - ubRowStart_;
        if (ly < 0 || ly >= ubRows_ || x < 0 || x >= static_cast<int32_t>(tiling_->width)) {
            return 0.0F;
        }
        return static_cast<float>(imgLocal_.GetValue(static_cast<uint32_t>(ly) * static_cast<uint32_t>(ubStride_) +
                                                     static_cast<uint32_t>(x)));
    }
    if (y < 0 || x < 0 || y >= static_cast<int32_t>(tiling_->height) || x >= static_cast<int32_t>(tiling_->width)) {
        return 0.0F;
    }
    return static_cast<float>(imageGm_.GetValue(static_cast<uint64_t>(image) * tiling_->imageElements +
                                                static_cast<uint32_t>(y) * tiling_->width + static_cast<uint32_t>(x)));
}

template <typename T>
__aicore__ inline float GoodFeaturesToTrack<T>::SobelGx(uint32_t image, int32_t y, int32_t x) const
{
    return -Pixel(image, y - 1, x - 1) + Pixel(image, y - 1, x + 1) - 2.0F * Pixel(image, y, x - 1) +
           2.0F * Pixel(image, y, x + 1) - Pixel(image, y + 1, x - 1) + Pixel(image, y + 1, x + 1);
}

template <typename T>
__aicore__ inline float GoodFeaturesToTrack<T>::SobelGy(uint32_t image, int32_t y, int32_t x) const
{
    return -Pixel(image, y - 1, x - 1) - 2.0F * Pixel(image, y - 1, x) - Pixel(image, y - 1, x + 1) +
           Pixel(image, y + 1, x - 1) + 2.0F * Pixel(image, y + 1, x) + Pixel(image, y + 1, x + 1);
}

template <typename T>
__aicore__ inline float GoodFeaturesToTrack<T>::Response(uint32_t image, int32_t y, int32_t x) const
{
    const int32_t radius = tiling_->blockSize / 2;
    float sumXX = 0.0F;
    float sumYY = 0.0F;
    float sumXY = 0.0F;
    if (useGradCache_) {
        // 从预计算的梯度环读窗口累加（每像素 Sobel 只算一次，见 Process 的 gradBuf_ 填充）。
        // 索引必须与 Process 一致（环下标 = py % gradRingRows_）。
        const int32_t ringRows = (gradRingRows_ < 1) ? 1 : gradRingRows_;
        for (int32_t by = -radius; by <= radius; ++by) {
            for (int32_t bx = -radius; bx <= radius; ++bx) {
                const int32_t py = y + by;
                const int32_t px = x + bx;
                if (py < 0 || px < 0 || py >= static_cast<int32_t>(tiling_->height) ||
                    px >= static_cast<int32_t>(tiling_->width) || py > gradValidHi_) {
                    continue;
                }
                LocalTensor<float>
                    gradRow = gradLocal_[static_cast<uint32_t>(py % ringRows) * 3U * static_cast<uint32_t>(ubStride_)];
                sumXX += gradRow.GetValue(static_cast<uint32_t>(px));
                sumYY += gradRow.GetValue(static_cast<uint32_t>(ubStride_) + px);
                sumXY += gradRow.GetValue(static_cast<uint32_t>(2) * static_cast<uint32_t>(ubStride_) + px);
            }
        }
    } else {
        // 回退：窗口内直接重算 Sobel（超宽行未启用 UB 缓存时）
        for (int32_t by = -radius; by <= radius; ++by) {
            for (int32_t bx = -radius; bx <= radius; ++bx) {
                const int32_t py = y + by;
                const int32_t px = x + bx;
                const float gx = SobelGx(image, py, px);
                const float gy = SobelGy(image, py, px);
                sumXX += gx * gx;
                sumYY += gy * gy;
                sumXY += gx * gy;
            }
        }
    }
    return ResponseFromSum(sumXX, sumYY, sumXY);
}

template <typename T>
__aicore__ inline float GoodFeaturesToTrack<T>::ResponseFromSum(float sumXX, float sumYY, float sumXY) const
{
    if (tiling_->useHarris != 0U) {
        const float trace = sumXX + sumYY;
        return sumXX * sumYY - sumXY * sumXY - tiling_->harrisK * trace * trace;
    }
    const float delta = sumXX - sumYY;
    return 0.5F * (sumXX + sumYY - sqrt(delta * delta + 4.0F * sumXY * sumXY));
}

template <typename T>
__aicore__ inline bool GoodFeaturesToTrack<T>::IsLocalMaximum(uint32_t image, int32_t y, int32_t x, float value) const
{
    for (int32_t dy = -1; dy <= 1; ++dy) {
        for (int32_t dx = -1; dx <= 1; ++dx) {
            if (dy == 0 && dx == 0) {
                continue;
            }
            const int32_t ny = y + dy;
            const int32_t nx = x + dx;
            if (ny >= 0 && nx >= 0 && ny < static_cast<int32_t>(tiling_->height) &&
                nx < static_cast<int32_t>(tiling_->width)) {
                const float other = responseGm_.GetValue(static_cast<uint64_t>(image) * tiling_->imageElements +
                                                         static_cast<uint32_t>(ny) * tiling_->width +
                                                         static_cast<uint32_t>(nx));
                if (other > value || (other == value && (ny < y || (ny == y && nx < x)))) {
                    return false;
                }
            }
        }
    }
    return true;
}

template <typename T>
__aicore__ inline bool GoodFeaturesToTrack<T>::FarEnough(uint32_t image, int32_t y, int32_t x, uint32_t selected) const
{
    const float minDistance2 = tiling_->minDistance * tiling_->minDistance;
    const uint64_t outputBase = static_cast<uint64_t>(image) * tiling_->outputCapacity * 2U;
    for (uint32_t i = 0; i < selected; ++i) {
        // 已选角点按 (x, y) 存放（对齐 OpenCV），故先取 x 再取 y
        const float dx = static_cast<float>(x) - cornersGm_.GetValue(outputBase + 2U * i);
        const float dy = static_cast<float>(y) - cornersGm_.GetValue(outputBase + 2U * i + 1U);
        if (dy * dy + dx * dx < minDistance2) {
            return false;
        }
    }
    return true;
}

template <typename T>
__aicore__ inline uint32_t GoodFeaturesToTrack<T>::SortKeyOf(float value)
{
    // float 位模式 → 单调 uint32（按值升序），再取反 → 按响应降序。
    // 候选值恒 >= 阈值 >= 0，符号位分支对负值同样正确（防御性保留）。
    union {
        float f;
        uint32_t u;
    } pun;
    pun.f = value;
    const uint32_t bits = pun.u;
    const uint32_t asc = ((bits & 0x80000000U) != 0U) ? (~bits) : (bits | 0x80000000U);
    return ~asc;
}

template <typename T>
__aicore__ inline void GoodFeaturesToTrack<T>::RadixPass(GlobalTensor<float> srcVal, GlobalTensor<uint32_t> srcIdx,
                                                         GlobalTensor<float> dstVal, GlobalTensor<uint32_t> dstIdx,
                                                         uint64_t base, uint32_t count, int32_t shift,
                                                         LocalTensor<int32_t> hist)
{
    // 8bit LSD 稳定散列：计数 → 前缀和 → 稳定分布
    for (int32_t b = 0; b < 256; ++b) {
        hist.SetValue(b, 0);
    }
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t bucket = (SortKeyOf(srcVal.GetValue(base + i)) >> shift) & 0xFFU;
        hist.SetValue(bucket, hist.GetValue(bucket) + 1);
    }
    int32_t sum = 0;
    for (int32_t b = 0; b < 256; ++b) {
        const int32_t c = hist.GetValue(b);
        hist.SetValue(b, sum);
        sum += c;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const float v = srcVal.GetValue(base + i);
        const uint32_t bucket = (SortKeyOf(v) >> shift) & 0xFFU;
        const int32_t pos = hist.GetValue(bucket);
        hist.SetValue(bucket, pos + 1);
        dstVal.SetValue(base + static_cast<uint64_t>(pos), v);
        dstIdx.SetValue(base + static_cast<uint64_t>(pos), srcIdx.GetValue(base + i));
    }
}

// ===== 阶段 2a（全核并行）：本核扫描 image 内属于自己的行带，收集 3x3 局部极大候选 =====
// 分区：核 c 负责行带 [c*h/coreNum, (c+1)*h/coreNum)，候选写入候选数组中对应行段
// [bandStart*w, bandEnd*w)；各核区间互不重叠，容量之和恰为 H*W，因此不会越界。
template <typename T>
__aicore__ inline void GoodFeaturesToTrack<T>::FlushCandidates(uint64_t dstBase, uint32_t count)
{
    if (count == 0U) {
        return;
    }
    LocalTensor<int32_t> stageIdx = candStageBuf_.Get<int32_t>();
    LocalTensor<float> stageVal = candStageBuf_.Get<float>()[kCandStage];
    PipeBarrier<PIPE_ALL>(); // 标量写 UB 必须先于 MTE3 读出
    DataCopyParams cp;
    cp.blockCount = 1;
    cp.blockLen = static_cast<uint16_t>(count * sizeof(uint32_t));
    cp.srcStride = 0;
    cp.dstStride = 0;
    DataCopyPad(candIdxI32Gm_[dstBase], stageIdx, cp);
    DataCopyPad(candValGm_[dstBase], stageVal, cp);
    PipeBarrier<PIPE_MTE3>();
}

// 候选记录：写 UB 暂存（全图栅格索引 + 响应值）、累加键高位直方图、满 kCandStage 时 DMA 写出。
// 调用顺序必须与栅格扫描顺序一致（候选数组顺序会通过稳定排序影响并列候选的存活）。
template <typename T>
__aicore__ inline void GoodFeaturesToTrack<T>::PushCandidate(int32_t gy, int32_t x, float value, uint64_t bandCandBase,
                                                             uint32_t& staged, uint32_t& localCount,
                                                             LocalTensor<int32_t> stageIdx, LocalTensor<float> stageVal,
                                                             LocalTensor<float> keyHist)
{
    stageIdx.SetValue(staged, static_cast<int32_t>(static_cast<uint32_t>(gy) * static_cast<uint32_t>(tiling_->width) +
                                                   static_cast<uint32_t>(x)));
    stageVal.SetValue(staged, value);
    {
        const uint32_t kb = (SortKeyOf(value) >> 24) & 0xFFU;
        keyHist.SetValue(static_cast<int32_t>(kb), keyHist.GetValue(static_cast<int32_t>(kb)) + 1.0F);
    }
    ++staged;
    ++localCount;
    if (staged == kCandStage) {
        FlushCandidates(bandCandBase + localCount - staged, staged);
        staged = 0U;
    }
}

// 标量 3×3 局部极大判定：8 邻居全走 UB（图像外邻居跳过）。
// 并列时只有「栅格序更早的邻居」才否决 ⇒ 等价于 key=(value, -栅格序) 的严格最大。
// 由此可推出向量等价形式：上 3 个 + 左 1 个用「严格大于」，右 1 个 + 下 3 个用「大于等于」。
template <typename T>
__aicore__ inline bool GoodFeaturesToTrack<T>::IsLocalMaxStaged(LocalTensor<float> rLocal, int32_t gy, int32_t x,
                                                                int32_t w, int32_t h, int32_t srcY0, float value) const
{
    for (int32_t dy = -1; dy <= 1; ++dy) {
        for (int32_t dx = -1; dx <= 1; ++dx) {
            if (dy == 0 && dx == 0) {
                continue;
            }
            const int32_t ny = gy + dy;
            const int32_t nx = x + dx;
            if (ny < 0 || nx < 0 || ny >= h || nx >= w) {
                continue;
            }
            const float other = rLocal.GetValue(static_cast<uint32_t>(ny - srcY0) * static_cast<uint32_t>(ubStride_) +
                                                static_cast<uint32_t>(nx));
            if (other > value || (other == value && (ny < gy || (ny == gy && nx < x)))) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
__aicore__ inline void GoodFeaturesToTrack<T>::CollectCandidates(uint32_t image, float maxResponse,
                                                                 LocalTensor<float> scanStage, int32_t scanStageRows,
                                                                 uint32_t coreIdx, uint32_t coreNum, float coreMax)
{
    // 本核候选「键高 8 位」直方图（并行 TopK 用）：0 号核归约后定位 b*，
    // 各核再据此过滤自己的分区。局部累加，核间无竞争。
    LocalTensor<float> keyHist = keyHistBuf_.Get<float>();
    for (int32_t b = 0; b < 256; ++b) {
        keyHist.SetValue(b, 0.0F);
    }
    DataCopyParams histParams;
    histParams.blockCount = 1;
    histParams.blockLen = static_cast<uint16_t>(256U * sizeof(int32_t)); // 1KB
    histParams.srcStride = 0;
    histParams.dstStride = 0;
    if (maxResponse <= 0.0F) {
        // 阈值退化为 0（全零/均匀图）：无候选。但仍必须写出一份全零直方图，
        // 否则 0 号核会归约到未初始化的 workspace 垃圾值（b* 随机 → 结果全 0）。
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(keyHistGm_[coreIdx * 256U], keyHist, histParams);
        PipeBarrier<PIPE_MTE3>();
        // 同时把本核在该图的统计槽置零（局部 max 与候选数）：退化图不会走下方写回，
        // 若槽内残留别的图/别的核的值，2b 会读到错误的候选数 ⇒ 该图结果错误。
        WriteCoreMax(image, coreIdx, 0.0F);
        return;
    }
    const int32_t h = static_cast<int32_t>(tiling_->height);
    const int32_t w = static_cast<int32_t>(tiling_->width);
    const float threshold = maxResponse * tiling_->qualityLevel;
    const uint64_t responseBase = static_cast<uint64_t>(image) * tiling_->imageElements;

    // staging 行数必须同时受 UB 容量与单块字节上限约束；否则宽图（w 大 → staging 行数少）
    // 会按 32 行写入而越界踩 UB，触发 MTE 越界 / vector core exception。
    // 此外每块还要额外容纳上下各 1 行 halo：3×3 局部极大的 8 个邻居必须全部来自 UB。
    // 用标量 GetValue 读 GM 取块外邻居会读到陈旧值（跨核 DMA 写与标量读不保证一致），
    // 会使候选集随分块大小变化 → 因此每块实际搬 rowsThis+2 行，邻居索引统一走 UB。
    const int32_t respRowBytes = w * static_cast<int32_t>(sizeof(float));
    const int32_t kMaxBlockRows = 32; // 单次块拷贝行数上限（自身容量上限）
    int32_t maxBlockRows = (respRowBytes <= 255 * 32) ? kMaxBlockRows : 1;
    if (maxBlockRows > scanStageRows - 2) {
        maxBlockRows = scanStageRows - 2; // 预留上下各 1 行 halo
    }
    if (maxBlockRows < 1) {
        maxBlockRows = 1;
    }

    const int32_t bandStart = static_cast<int32_t>(static_cast<int64_t>(coreIdx) * h / static_cast<int64_t>(coreNum));
    const int32_t bandEnd = static_cast<int32_t>(static_cast<int64_t>(coreIdx + 1) * h / static_cast<int64_t>(coreNum));
    const uint64_t bandCandBase = responseBase + static_cast<uint64_t>(bandStart) * w;

    uint32_t localCount = 0U; // 本核累计候选数（含已 flush 的）
    uint32_t staged = 0U;     // 当前 UB 暂存区里的候选数
    LocalTensor<int32_t> stageIdx = candStageBuf_.Get<int32_t>();
    LocalTensor<float> stageVal = candStageBuf_.Get<float>()[kCandStage];
    LocalTensor<float> rLocal = scanStage;
    for (int32_t yBlock = bandStart; yBlock < bandEnd; yBlock += maxBlockRows) {
        const int32_t rowsThis = GftMin(maxBlockRows, bandEnd - yBlock);
        // 连带上下各 1 行 halo 一起搬入 UB（贴图像边界时自然少搬）
        const int32_t srcY0 = GftMax(0, yBlock - 1);
        const int32_t srcY1 = GftMin(h, yBlock + rowsThis + 1);
        const uint32_t stageBase = responseBase + static_cast<uint32_t>(srcY0) * w;
        if (respRowsPadded_) {
            // 行尾需要补齐时逐行拷贝，UB 基址显式写成 r*ubStride_，与索引定义严格一致
            for (int32_t r = srcY0; r < srcY1; ++r) {
                DataCopyParams cp;
                cp.blockCount = 1;
                cp.blockLen = static_cast<uint16_t>(respRowBytes);
                cp.srcStride = 0;
                cp.dstStride = 0;
                DataCopyPad(rLocal[static_cast<uint32_t>(r - srcY0) * static_cast<uint32_t>(ubStride_)],
                            responseGm_[stageBase + static_cast<uint32_t>(r - srcY0) * w], cp, {false, 0, 0, 0});
            }
        } else {
            // 行字节 32B 对齐：UB 行距 == 原始行字节，可一次多行块拷贝
            DataCopyParams cp;
            cp.blockCount = static_cast<uint16_t>(srcY1 - srcY0);
            cp.blockLen = static_cast<uint16_t>(respRowBytes);
            cp.srcStride = 0;
            cp.dstStride = 0;
            DataCopyPad(rLocal, responseGm_[stageBase], cp, {false, 0, 0, 0});
        }
        PipeBarrier<PIPE_MTE2>();
        // 逐像素判 3×3 局部极大：中心与 8 个邻居全部读 UB（含 halo 行），图像外的邻居跳过。
        // 这样判定与分块大小无关，且不会读到跨核陈旧值。
        for (int32_t y = 0; y < rowsThis; ++y) {
            const int32_t gy = yBlock + y;
            const uint32_t rowCenter = static_cast<uint32_t>(gy - srcY0) * static_cast<uint32_t>(ubStride_);
            // ===== 向量路径：整行一次判完（仅 w%8==0 且行不贴图像上下边界）=====
            // 判定可完全向量化：并列时只有「栅格序更早的邻居」否决 ⇒
            //   上 3 个 + 左 1 个（早序）用「严格大于」；右 1 个 + 下 3 个（后序）用「大于等于」。
            // 三张掩码相与后逐字扫描取出候选（每 4 个像素一次标量读，绝大多数直接跳过）。
            // 候选必须按栅格序入队，故顺序为：头边缘 → 向量中段 → 尾边缘。
            const int32_t vecStart = 8;               // 起点 8 的倍数（掩码/矢量地址对齐）
            const int32_t vecEndRaw = (w - 1) & (~7); // ≤ w-1 的最大 8 倍数（保证 x+1 ≤ w-1）
            const int32_t vecEnd = (vecEndRaw > vecStart) ? vecEndRaw : vecStart;
            if ((w & 7) == 0 && w >= 24 && vecEnd > vecStart && gy >= 1 && gy <= h - 2) {
                LocalTensor<float> sv = sobVecBuf_.Get<float>();
                LocalTensor<uint32_t> sobOff = sobOffBuf_.Get<uint32_t>();
                LocalTensor<float> pL = sv[0U];
                LocalTensor<float> pC = sv[1U * kWinChunk];
                LocalTensor<float> pR = sv[2U * kWinChunk];
                LocalTensor<float> cL = sv[3U * kWinChunk];
                LocalTensor<float> cCtr = sv[4U * kWinChunk]; // 中心值（也是候选值来源）
                LocalTensor<float> cR = sv[5U * kWinChunk];
                LocalTensor<float> nL = sv[6U * kWinChunk];
                LocalTensor<float> nC = sv[7U * kWinChunk];
                LocalTensor<float> nR = sv[8U * kWinChunk];
                const uint32_t uWv = static_cast<uint32_t>(ubStride_);
                LocalTensor<float> rowPrev = rLocal[static_cast<uint32_t>(gy - 1 - srcY0) * uWv];
                LocalTensor<float> rowCur = rLocal[static_cast<uint32_t>(gy - srcY0) * uWv];
                LocalTensor<float> rowNext = rLocal[static_cast<uint32_t>(gy + 1 - srcY0) * uWv];
                for (int32_t x = 0; x < vecStart; ++x) { // 头边缘
                    const float value = rLocal.GetValue(rowCenter + static_cast<uint32_t>(x));
                    if (value >= threshold && IsLocalMaxStaged(rLocal, gy, x, w, h, srcY0, value)) {
                        PushCandidate(gy, x, value, bandCandBase, staged, localCount, stageIdx, stageVal, keyHist);
                    }
                }
                for (int32_t x0 = vecStart; x0 < vecEnd; x0 += static_cast<int32_t>(kWinChunk)) {
                    int32_t cnt = vecEnd - x0;
                    if (cnt > static_cast<int32_t>(kWinChunk)) {
                        cnt = static_cast<int32_t>(kWinChunk);
                    }
                    const uint32_t uCnt = static_cast<uint32_t>(cnt);
                    const uint32_t bA = static_cast<uint32_t>((x0 - 1) * 4);
                    const uint32_t bB = static_cast<uint32_t>(x0 * 4);
                    const uint32_t bC = static_cast<uint32_t>((x0 + 1) * 4);
                    Gather(pL, rowPrev, sobOff, bA, uCnt);
                    Gather(pC, rowPrev, sobOff, bB, uCnt);
                    Gather(pR, rowPrev, sobOff, bC, uCnt);
                    Gather(cL, rowCur, sobOff, bA, uCnt);
                    Gather(cCtr, rowCur, sobOff, bB, uCnt);
                    Gather(cR, rowCur, sobOff, bC, uCnt);
                    Gather(nL, rowNext, sobOff, bA, uCnt);
                    Gather(nC, rowNext, sobOff, bB, uCnt);
                    Gather(nR, rowNext, sobOff, bC, uCnt);
                    Max(pL, pL, pC, uCnt); // max(上 3, 左) = 早序邻居最大值
                    Max(pL, pL, pR, uCnt);
                    Max(pL, pL, cL, uCnt);
                    Max(nL, nL, nC, uCnt); // max(右, 下 3) = 后序邻居最大值
                    Max(nL, nL, nR, uCnt);
                    Max(nL, nL, cR, uCnt);
                    Duplicate(pC, threshold, uCnt); // pC 已空出，改作阈值广播
                    // 判定（不依赖掩码类 API，全部用减法 + Max/Maxs/Mins 归一化成 0/1）：
                    //   reject = max( [maxE - c >= 0], [max(maxL - c, th - c) > 0] )
                    // 即「早序邻居并列即否决」+「后序邻居与阈值需严格大于」。
                    Sub(pL, pL, cCtr, uCnt); // dE = maxE - c
                    Sub(nL, nL, cCtr, uCnt); // dL = maxL - c
                    Sub(pC, pC, cCtr, uCnt); // dT = th - c
                    Max(nL, nL, pC, uCnt);
                    Maxs(nL, nL, 0.0F, uCnt);
                    Muls(nL, nL, 1.0E30F, uCnt);
                    Mins(nL, nL, 1.0F, uCnt); // rL = (dL>0 或 dT>0) ? 1 : 0
                    Muls(pL, pL, -1.0F, uCnt);
                    Maxs(pL, pL, 0.0F, uCnt);
                    Muls(pL, pL, 1.0E30F, uCnt);
                    Mins(pL, pL, 1.0F, uCnt); // = (c > maxE) ? 1 : 0
                    Muls(pL, pL, -1.0F, uCnt);
                    Adds(pL, pL, 1.0F, uCnt); // rE = (c <= maxE) ? 1 : 0
                    Max(pL, pL, nL, uCnt);    // reject
                    Muls(pL, pL, -1.0F, uCnt);
                    Adds(pL, pL, 1.0F, uCnt); // keep ∈ {0.0, 1.0}
                    PipeBarrier<PIPE_ALL>();  // 向量写指示值 → 标量逐字读
                    for (uint32_t li = 0U; li < uCnt; ++li) {
                        if (pL.GetValue(li) == 0.0F) {
                            continue; // 非候选：一次标量读跳过
                        }
                        PushCandidate(gy, x0 + static_cast<int32_t>(li), cCtr.GetValue(li), bandCandBase, staged,
                                      localCount, stageIdx, stageVal, keyHist);
                    }
                    PipeBarrier<PIPE_ALL>(); // 标量读完中心值与指示值 → 下一块 Gather 才能覆写槽位
                }
                for (int32_t x = vecEnd; x < w; ++x) { // 尾边缘
                    const float value = rLocal.GetValue(rowCenter + static_cast<uint32_t>(x));
                    if (value >= threshold && IsLocalMaxStaged(rLocal, gy, x, w, h, srcY0, value)) {
                        PushCandidate(gy, x, value, bandCandBase, staged, localCount, stageIdx, stageVal, keyHist);
                    }
                }
                continue;
            }
            for (int32_t x = 0; x < w; ++x) {
                const float value = rLocal.GetValue(rowCenter + static_cast<uint32_t>(x));
                if (value < threshold) {
                    continue;
                }
                bool localMax = true;
                for (int32_t dy = -1; dy <= 1 && localMax; ++dy) {
                    for (int32_t dx = -1; dx <= 1; ++dx) {
                        if (dy == 0 && dx == 0) {
                            continue;
                        }
                        const int32_t ny = gy + dy;
                        const int32_t nx = x + dx;
                        if (ny < 0 || nx < 0 || ny >= h || nx >= w) {
                            continue;
                        }
                        const float other = rLocal.GetValue(static_cast<uint32_t>(ny - srcY0) *
                                                                static_cast<uint32_t>(ubStride_) +
                                                            static_cast<uint32_t>(nx));
                        if (other > value || (other == value && (ny < gy || (ny == gy && nx < x)))) {
                            localMax = false;
                            break;
                        }
                    }
                }
                if (localMax) {
                    // 候选索引用全图扁平索引
                    stageIdx.SetValue(staged,
                                      static_cast<int32_t>(static_cast<uint32_t>(gy) * static_cast<uint32_t>(w) +
                                                           static_cast<uint32_t>(x)));
                    stageVal.SetValue(staged, value);
                    {
                        const uint32_t kb = (SortKeyOf(value) >> 24) & 0xFFU;
                        keyHist.SetValue(static_cast<int32_t>(kb), keyHist.GetValue(static_cast<int32_t>(kb)) + 1.0F);
                    }
                    ++staged;
                    ++localCount;
                    if (staged == kCandStage) {
                        FlushCandidates(bandCandBase + localCount - staged, staged);
                        staged = 0U;
                    }
                }
            }
        }
    }
    // 收尾：把暂存区里剩余的候选写出
    FlushCandidates(bandCandBase + localCount - staged, staged);
    // 直方图搬出：标量写 UB → MTE3 读出必须显式同步
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(keyHistGm_[coreIdx * 256U], keyHist, histParams);
    PipeBarrier<PIPE_MTE3>();

    // 本核候选数用 DMA(MTE3) 写回 StatSlot(image, coreIdx)[1]（与 [0] 的该图局部 max 同一 32B 块），
    // 保证 SyncAll 后对 0 号核可见（标量 SetValue 写 GM 无跨核 cache 一致性保证）。
    {
        LocalTensor<float> maxLocal = respBuf_.Get<float>();
        maxLocal.SetValue(0U, coreMax);
        // AscendC 禁止 float 与无符号整数直接互转，经 int32_t 中转
        maxLocal.SetValue(1U, static_cast<float>(static_cast<int32_t>(localCount)));
        // 标量写 UB 后再由 MTE3 搬出，必须显式同步 PIPE_S → PIPE_MTE3，
        // 否则 DMA 可能读到 respBuf_ 里上一次扫描残留的响应值。
        PipeBarrier<PIPE_ALL>();
        DataCopyParams cntParams;
        cntParams.blockCount = 1;
        cntParams.blockLen = static_cast<uint16_t>(8U * sizeof(float)); // 32B
        cntParams.srcStride = 0;
        cntParams.dstStride = 0;
        DataCopyPad(maxPerCoreGm_[StatSlot(image, coreIdx)], maxLocal, cntParams);
        PipeBarrier<PIPE_MTE3>();
    }
}

// ===== 阶段 2b：键高位直方图定位 TopK 范围 → 收集 → 小集合排序 → 贪心选取 =====
template <typename T>
__aicore__ inline uint64_t GoodFeaturesToTrack<T>::StatSlot(uint32_t image, uint32_t core) const
{
    return (static_cast<uint64_t>(image) * tiling_->coreNum + core) * 8U;
}

template <typename T>
__aicore__ inline void GoodFeaturesToTrack<T>::WriteCoreMax(uint32_t image, uint32_t core, float value)
{
    LocalTensor<float> maxLocal = respBuf_.Get<float>();
    maxLocal.SetValue(0U, value);
    maxLocal.SetValue(1U, 0.0F); // 候选数位先置 0（阶段 2a 会覆盖），避免搬运残留脏值
    PipeBarrier<PIPE_ALL>();     // 标量写 UB 必须先于 MTE3 读出
    DataCopyParams p;
    p.blockCount = 1;
    p.blockLen = static_cast<uint16_t>(8U * sizeof(float)); // 32B
    p.srcStride = 0;
    p.dstStride = 0;
    DataCopyPad(maxPerCoreGm_[StatSlot(image, core)], maxLocal, p);
    PipeBarrier<PIPE_MTE3>();
}

template <typename T>
__aicore__ inline void GoodFeaturesToTrack<T>::CoreRangeForImage(uint32_t image, uint32_t& c0, uint32_t& c1) const
{
    // 全局行按 coreNum 均分（与阶段 1 的 rowStart/rowEnd 同源）：核 c 覆盖
    // [c*totalRows/coreNum, (c+1)*totalRows/coreNum)；该图占全局行 [image*h, (image+1)*h)。
    const int64_t h = tiling_->height;
    const int64_t totalRows = static_cast<int64_t>(tiling_->batch) * h;
    const int64_t imgLo = static_cast<int64_t>(image) * h;
    const int64_t imgHi = imgLo + h;
    const int64_t coreNum = tiling_->coreNum;
    c0 = 0U;
    c1 = 0U;
    bool found = false;
    for (int64_t c = 0; c < coreNum; ++c) {
        const int64_t rStart = c * totalRows / coreNum;
        const int64_t rEnd = (c + 1) * totalRows / coreNum;
        if (rStart >= rEnd || rEnd <= imgLo || rStart >= imgHi) {
            continue; // 空行带的核（H 很小时）或与该图无交集的核
        }
        if (!found) {
            c0 = static_cast<uint32_t>(c);
            found = true;
        }
        c1 = static_cast<uint32_t>(c);
    }
}

template <typename T>
__aicore__ inline LocalTensor<float> GoodFeaturesToTrack<T>::LoadCoreStats(uint32_t image)
{
    LocalTensor<float> stat = statBuf_.Get<float>();
    uint32_t c0 = 0U;
    uint32_t c1 = 0U;
    CoreRangeForImage(image, c0, c1);
    const uint32_t n = (c1 >= c0) ? (c1 - c0 + 1U) : 1U;
    DataCopyParams cp;
    cp.blockCount = 1;
    cp.blockLen = static_cast<uint16_t>(n * 8U * sizeof(float));
    cp.srcStride = 0;
    cp.dstStride = 0;
    DataCopyPad(stat, maxPerCoreGm_[StatSlot(image, c0)], cp, {false, 0, 0, 0});
    PipeBarrier<PIPE_MTE2>();
    return stat;
}

// 2b-A / 2b-C 共用：按 mask（bucket <= maskHi 保留）统计每核计数，写各核前缀偏移。
// mask + Mul + ReduceSum：每核 2 条向量指令。计数 <= H*W 为整数、fp32 精确，
// 且逐核串行前缀和的顺序固定 ⇒ 结果逐位可复现。
// maxPerCore >= 0 时把每核「保留数」钳到该上限（maxPerCore < 0 表示不限制）；无论是否钳制，
// 写出的都是本核区间的**起点**、且前缀推进量 = 实际保留数 ⇒ 前缀偏移始终稠密、无洞。
template <typename T>
__aicore__ inline uint32_t GoodFeaturesToTrack<T>::PerCoreOffsets(GlobalTensor<float> histGm, uint32_t coreNum,
                                                                  LocalTensor<int32_t> meta, uint32_t metaBase,
                                                                  uint32_t offStart, int32_t maskHi, int32_t maxPerCore)
{
    LocalTensor<float> buf = winScrBuf_.Get<float>(); // 阶段 1 之后空闲，4*256 float
    LocalTensor<float> mask = buf;
    LocalTensor<float> prod = buf[kHistRowElems];
    LocalTensor<float> red = buf[2U * kHistRowElems];
    LocalTensor<float> tmpv = buf[3U * kHistRowElems];
    LocalTensor<float> hrow = keyHistBuf_.Get<float>();
    for (int32_t b = 0; b < static_cast<int32_t>(kHistRowElems); ++b) {
        mask.SetValue(b, (b <= maskHi) ? 1.0F : 0.0F);
    }
    PipeBarrier<PIPE_ALL>(); // 标量写 mask → 向量 Mul 读
    DataCopyParams hp;
    hp.blockCount = 1;
    hp.blockLen = 256U * sizeof(float);
    hp.srcStride = 0;
    hp.dstStride = 0;
    uint32_t off = offStart;
    for (uint32_t c = 0U; c < coreNum; ++c) {
        DataCopyPad(hrow, histGm[c * 256U], hp, {false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        Mul(prod, hrow, mask, static_cast<int32_t>(kHistRowElems));
        PipeBarrier<PIPE_V>();
        ReduceSum(red, prod, tmpv, static_cast<int32_t>(kHistRowElems));
        PipeBarrier<PIPE_ALL>();
        const uint32_t cnt = static_cast<uint32_t>(static_cast<int32_t>(red.GetValue(0U)));
        // 只写起点、推进量取钳制后的保留数：否则后续核的起点会跳过被丢弃的洞，
        // 而写出侧（TopKFilterOwn）是按「起点 + 本核前 quota 个」紧凑写的。
        const uint32_t keep = (maxPerCore >= 0) ? static_cast<uint32_t>(GftMin(static_cast<int32_t>(cnt), maxPerCore)) :
                                                  cnt;
        meta.SetValue(metaBase + c, static_cast<int32_t>(off));
        off += keep;
    }
    return off;
}

// ===== 阶段 2b-A（仅 0 号核）：预填 -1 → 向量归约 H1(键 bit24-31) 定位 b* → 算各核「小于组」偏移 =====
// 元数据布局（int32，共 2*coreNum+8）：
//   [0]              = b*  （第 K 小键所在的一级桶；键升序 ⇒ 桶越小响应越大）
//   [1]              = b** （二级桶，只对 bucket==b* 的候选有意义；0xFFFFFFFF 表示「等于组全丢」）
//   [2+c]            = 核 c 在「小于组」(bucket < b*) 的写出偏移
//   [2+coreNum+c]    = 核 c 在「等于组」(bucket==b* 且 sub<=b**) 的写出偏移
//   [2+2*coreNum]    = 小于组总数
//   [2+2*coreNum+1]  = 等于组保留数
//   [2+2*coreNum+2]  = 保留总数（= 基数排序长度）
//   [2+2*coreNum+3]  = 等于组每核配额（0 = 不限制；>0 时 TopKFilterOwn 每核最多写这么多）
//   （元数据共 2*coreNum+8 个 int32，槽位 [2+2*coreNum+3] 在既有容量内，无需改 tiling/workspace 布局）
template <typename T>
__aicore__ inline void GoodFeaturesToTrack<T>::TopKBuildMeta(uint32_t image, uint32_t coreNum)
{
    const uint64_t outputBase = static_cast<uint64_t>(image) * tiling_->outputCapacity * 2U;
    // 预填 -1：容量大且基址 32B 对齐时走「UB 填 -1 + DataCopyPad 分块广播」（DMA 带宽受限）；
    // 其余情况仍用标量，避免 GM 侧非 32B 对齐时 DataCopyPad 报错。
    // candStageBuf_ 在阶段 2b-A 空闲（2a 已 flush 完，2b-B 才复用），共 8 KB = 2048 个 float。
    const uint32_t capElems = 2U * tiling_->outputCapacity;
    const uint32_t fillChunk = 2048U; // 与 candStageBuf_ 容量一致
    if (capElems > 4096U && ((outputBase & 7U) == 0U)) {
        LocalTensor<float> fill = candStageBuf_.Get<float>();
        for (uint32_t i = 0; i < fillChunk; ++i) {
            fill.SetValue(i, -1.0F);
        }
        PipeBarrier<PIPE_ALL>(); // 标量写 UB 必须先于 MTE3 读出
        uint32_t done = 0U;
        for (; done + fillChunk <= capElems; done += fillChunk) {
            // 整块 2048 float = 8 KB（32B 对齐）走普通 DataCopy，比 DataCopyPad 少一层拆分开销
            DataCopy(cornersGm_[outputBase + done], fill, fillChunk);
        }
        PipeBarrier<PIPE_MTE3>();
        for (uint32_t i = done; i < capElems; ++i) { // 尾部不足一块，标量补齐
            cornersGm_.SetValue(outputBase + i, -1.0F);
        }
    } else {
        for (uint32_t i = 0; i < tiling_->outputCapacity; ++i) {
            cornersGm_.SetValue(outputBase + 2U * i, -1.0F);
            cornersGm_.SetValue(outputBase + 2U * i + 1U, -1.0F);
        }
    }
    // 归约各核直方图：直方图存 fp32，可直接用向量 Add。
    // （int32 的 ReduceSum 会退化到 fp16 的 vcadd 而编译失败；Cast(int32→fp32) 实测算错计数。）
    // 跨核生产者-消费者必须走 DMA（标量读 GM 可能命中本核陈旧缓存行）。
    LocalTensor<float> acc = histBuf_.Get<float>();
    LocalTensor<float> row = keyHistBuf_.Get<float>();
    LocalTensor<int32_t> meta = metaBuf_.Get<int32_t>();
    // 正确性：计数 <= H*W <= 2^24，fp32 加法精确；向量 Add 逐 lane 独立、无跨位置累加，
    // 且各核按 c 升序折叠，与标量逐桶累加的结合顺序一致 ⇒ 逐位相同。
    // candStageBuf_ 在 2b-A 此处空闲（2a 已 flush 完，2b-B 才复用）。
    Duplicate(acc, 0.0F, static_cast<int32_t>(kHistRowElems));
    DataCopyParams hp;
    hp.blockCount = 1;
    hp.blockLen = 256U * sizeof(float);
    hp.srcStride = 0;
    hp.dstStride = 0;
    for (uint32_t c = 0U; c < coreNum; ++c) {
        DataCopyPad(row, keyHistGm_[c * 256U], hp, {false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>(); // ① MTE2 写 row → 向量 Add 读 row
        Add(acc, acc, row, static_cast<int32_t>(kHistRowElems));
        PipeBarrier<PIPE_ALL>(); // ② 本轮向量读必须早于下一轮 DMA 覆写 row
    }
    PipeBarrier<PIPE_ALL>(); // ③ 向量写 acc → 后续标量读 acc
    // b*：全局累计计数首次 >= outputCapacity 的桶
    const uint32_t cap = tiling_->outputCapacity;
    int32_t bStar = 255;
    uint32_t cum = 0U;
    for (int32_t b = 0; b < 256; ++b) {
        const uint32_t cntB = static_cast<uint32_t>(static_cast<int32_t>(acc.GetValue(b)));
        if (cum + cntB >= cap) {
            bStar = b;
            break;
        }
        cum += cntB;
    }
    meta.SetValue(0, bStar);
    // 各核保留数（= 该核直方图 bucket<b* 之和）→ 前缀和 = 写出偏移（向量化，见 PerCoreOffsets）
    uint32_t off = PerCoreOffsets(keyHistGm_, coreNum, meta, 2U, 0U, static_cast<int32_t>(bStar) - 1);
    meta.SetValue(2U + 2U * coreNum, static_cast<int32_t>(off));
    // 标量写 UB → MTE3 搬出，必须显式同步（否则 DMA 可能读到 UB 里的旧值）
    PipeBarrier<PIPE_ALL>();
    DataCopyParams mp;
    mp.blockCount = 1;
    mp.blockLen = static_cast<uint16_t>((2U * coreNum + 8U) * sizeof(int32_t));
    mp.srcStride = 0;
    mp.dstStride = 0;
    DataCopyPad(topkMetaGm_, meta, mp);
    PipeBarrier<PIPE_MTE3>();
}

// ===== 阶段 2b-B / 2b-D（全核并行）：mode=0 写「小于组」并统计二级直方图；mode=1 写「等于组」 =====
// 每核只扫自己的分区（n/coreNum 个候选），写到「自己的偏移区间」，因此核间无数据竞争。
// 两趟都保持核内扫描序，组内再按核号顺序拼接 ⇒ 保留集整体等于全图候选序的一个子序列。
// 同一 key 必然同 bucket 同 sub，故等价键不会跨两组；稳定基数排序后次序与「全量保留集」
// 完全一致（逐位等价）。
// 写出必须用 MTE3 DMA（标量 SetValue 写 GM 对其它核不可见），故先攒进 UB staging 再整块搬出。
template <typename T>
__aicore__ inline void GoodFeaturesToTrack<T>::TopKFilterOwn(uint32_t image, uint32_t coreIdx, uint32_t coreNum,
                                                             uint32_t mode)
{
    LocalTensor<int32_t> meta = metaBuf_.Get<int32_t>();
    DataCopyParams mp;
    mp.blockCount = 1;
    mp.blockLen = static_cast<uint16_t>((2U * coreNum + 8U) * sizeof(int32_t));
    mp.srcStride = 0;
    mp.dstStride = 0;
    DataCopyPad(meta, topkMetaGm_, mp, {false, 0, 0, 0});
    PipeBarrier<PIPE_MTE2>();
    const uint32_t bStar = static_cast<uint32_t>(meta.GetValue(0));
    const uint32_t bStar2 = static_cast<uint32_t>(meta.GetValue(1U));
    // 等于组每核配额（0 = 不限制）：由 2b-C 在并列悬崖时写入，只对 mode=1（写等于组）生效
    const uint32_t quota = (mode == 1U) ? static_cast<uint32_t>(meta.GetValue(2U + 2U * coreNum + 3U)) : 0U;
    const uint32_t dstBase = static_cast<uint32_t>(
        meta.GetValue((mode == 0U) ? (2U + coreIdx) : (2U + coreNum + coreIdx)));
    const int32_t h = static_cast<int32_t>(tiling_->height);
    const int32_t w = static_cast<int32_t>(tiling_->width);
    const uint64_t candBase = static_cast<uint64_t>(image) * tiling_->imageElements;
    const int32_t ya = static_cast<int32_t>(static_cast<int64_t>(coreIdx) * h / static_cast<int64_t>(coreNum));
    const uint64_t src = candBase + static_cast<uint64_t>(ya) * w;
    // 本核候选数从「每核统计量」DMA 读回（不能用 keyHistBuf_：0 号核在归约时已把它覆写成
    // 最后一个核的直方图）。statBuf_ 容量足够，此处只取本核那 8 个 float。
    LocalTensor<float> ownStat = statBuf_.Get<float>();
    DataCopyParams sp;
    sp.blockCount = 1;
    sp.blockLen = static_cast<uint16_t>(8U * sizeof(float));
    sp.srcStride = 0;
    sp.dstStride = 0;
    DataCopyPad(ownStat, maxPerCoreGm_[StatSlot(image, coreIdx)], sp, {false, 0, 0, 0});
    PipeBarrier<PIPE_MTE2>();
    const uint32_t cntAll = static_cast<uint32_t>(static_cast<int32_t>(ownStat.GetValue(1U)));
    LocalTensor<int32_t> stageIdx = candStageBuf_.Get<int32_t>();
    LocalTensor<float> stageVal = candStageBuf_.Get<float>()[kCandStage];
    LocalTensor<float> hist2 = subHistBuf_.Get<float>();
    if (mode == 0U) {
        for (int32_t b = 0; b < 256; ++b) {
            hist2.SetValue(b, 0.0F);
        }
    }
    const bool keepNothing = (bStar2 > 255U);
    uint32_t staged = 0U;
    uint32_t written = 0U; // 已写出的保留候选数（= 相对本核偏移区间的写位置）
    for (uint32_t i = 0U; i < cntAll; ++i) {
        const float val = candValGm_.GetValue(src + i);
        const uint32_t key = SortKeyOf(val);
        const uint32_t kb = (key >> 24) & 0xFFU;
        if (mode == 0U) {
            if (kb > bStar) {
                continue; // 键更大，不可能进前 K
            }
            if (kb == bStar) {
                // 落在边界桶：只统计二级直方图，去留交给 mode=1 那趟决定
                const uint32_t sb = (key >> 16) & 0xFFU;
                hist2.SetValue(static_cast<int32_t>(sb), hist2.GetValue(static_cast<int32_t>(sb)) + 1.0F);
                continue;
            }
        } else if (kb != bStar || keepNothing || (((key >> 16) & 0xFFU) > bStar2)) {
            continue;
        }
        // 配额钳制（仅等于组）：按扫描序取前 quota 个，写满即停。
        // 必须截断，否则本核写出会越过「起点 + quota」区间、踩到后一核的偏移区间。
        if (quota > 0U && written + staged >= quota) {
            break;
        }
        stageIdx.SetValue(staged, candIdxI32Gm_.GetValue(src + i));
        stageVal.SetValue(staged, val);
        ++staged;
        if (staged == kCandStage) {
            PipeBarrier<PIPE_ALL>();
            DataCopyParams cp;
            cp.blockCount = 1;
            cp.blockLen = kCandStage * sizeof(int32_t);
            cp.srcStride = 0;
            cp.dstStride = 0;
            DataCopyPad(candIdx2I32Gm_[candBase + dstBase + written], stageIdx, cp);
            DataCopyPad(candVal2Gm_[candBase + dstBase + written], stageVal, cp);
            PipeBarrier<PIPE_MTE3>();
            written += staged;
            staged = 0U;
        }
    }
    if (staged > 0U) {
        PipeBarrier<PIPE_ALL>();
        DataCopyParams cp;
        cp.blockCount = 1;
        cp.blockLen = staged * sizeof(int32_t);
        cp.srcStride = 0;
        cp.dstStride = 0;
        DataCopyPad(candIdx2I32Gm_[candBase + dstBase + written], stageIdx, cp);
        DataCopyPad(candVal2Gm_[candBase + dstBase + written], stageVal, cp);
        PipeBarrier<PIPE_MTE3>();
    }
    if (mode == 0U) {
        // 二级直方图要跨核可见（下一阶段 0 号核归约），必须走 MTE3
        PipeBarrier<PIPE_ALL>();
        DataCopyParams h2p;
        h2p.blockCount = 1;
        h2p.blockLen = static_cast<uint16_t>(256U * sizeof(int32_t));
        h2p.srcStride = 0;
        h2p.dstStride = 0;
        DataCopyPad(subHistGm_[coreIdx * 256U], hist2, h2p);
        PipeBarrier<PIPE_MTE3>();
    }
}

// ===== 阶段 2b-C（仅 0 号核）：二级细化 —— 归约 H2 定位 b**、算「等于组」偏移与保留总数 =====
// 作用：把边界桶 b* 里的候选压到 ≈K，避免基数排序长度被显著放大。
template <typename T>
__aicore__ inline void GoodFeaturesToTrack<T>::TopKRefineMeta(uint32_t image, uint32_t coreNum)
{
    (void)image; // 元数据与镜像无关（每张图独立跑完整 2b）
    LocalTensor<float> acc = histBuf_.Get<float>();
    LocalTensor<float> row = keyHistBuf_.Get<float>();
    LocalTensor<int32_t> meta = metaBuf_.Get<int32_t>();
    DataCopyParams mp;
    mp.blockCount = 1;
    mp.blockLen = static_cast<uint16_t>((2U * coreNum + 8U) * sizeof(int32_t));
    mp.srcStride = 0;
    mp.dstStride = 0;
    DataCopyPad(meta, topkMetaGm_, mp, {false, 0, 0, 0});
    PipeBarrier<PIPE_MTE2>();
    const uint32_t totalLess = static_cast<uint32_t>(meta.GetValue(2U + 2U * coreNum));
    const uint32_t cap = tiling_->outputCapacity;
    // 同 2b-A：二级直方图归约同样走向量 Add。
    Duplicate(acc, 0.0F, static_cast<int32_t>(kHistRowElems));
    DataCopyParams hp;
    hp.blockCount = 1;
    hp.blockLen = 256U * sizeof(float);
    hp.srcStride = 0;
    hp.dstStride = 0;
    for (uint32_t c = 0U; c < coreNum; ++c) {
        DataCopyPad(row, subHistGm_[c * 256U], hp, {false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>(); // ① MTE2 写 row → 向量 Add 读 row
        Add(acc, acc, row, static_cast<int32_t>(kHistRowElems));
        PipeBarrier<PIPE_ALL>(); // ② 本轮向量读必须早于下一轮 DMA 覆写 row
    }
    PipeBarrier<PIPE_ALL>(); // ③ 向量写 acc → 后续标量读 acc
    // b**：让「小于组 + 等于组保留」首次 >= K。totalLess >= K 时用哨兵 0xFFFFFFFF
    // 表示「等于组全丢」（此时保留集立刻回落到 ≈K）。
    uint32_t bStar2 = 0xFFFFFFFFU;
    if (totalLess < cap) {
        const uint32_t need = cap - totalLess;
        uint32_t cum = 0U;
        bStar2 = 255U;
        for (int32_t b = 0; b < 256; ++b) {
            const uint32_t cntB = static_cast<uint32_t>(static_cast<int32_t>(acc.GetValue(b)));
            if (cum + cntB >= need) {
                bStar2 = static_cast<uint32_t>(b);
                break;
            }
            cum += cntB;
        }
    }
    // 各核「等于组保留数」= 本核 sub <= b** 的计数之和 → 前缀和（起点 = 小于组总数）
    // 先按「不限制」算一遍，得到等于组总量，据此判断是否需要加每核配额（并列悬崖修复）。
    const int32_t subMask = (bStar2 <= 255U) ? static_cast<int32_t>(bStar2) : -1;
    uint32_t off = PerCoreOffsets(subHistGm_, coreNum, meta, 2U + coreNum, totalLess, subMask);
    const uint32_t equalTotal = off - totalLess;
    // 并列悬崖：当同一个 (b*, b**) 子桶里的候选数远大于 K 时（noise=0 纯棋盘实测 ≈31k），
    // 子桶粒度截断会把整个子桶都保留下来，使 0 号核的 4 趟基数排序长度膨胀一个数量级。
    // 这里给「等于组」加每核配额，把保留总数压回 max(4096, outputCapacity) 量级。
    // equalTotal <= equalCap（正常输入）⇒ quota = 0 且不重算 ⇒ 与加配额前逐位一致。
    const uint32_t equalCap = static_cast<uint32_t>(GftMax(4096, static_cast<int32_t>(cap)));
    uint32_t quota = 0U;
    if (equalTotal > equalCap) {
        quota = (equalCap + coreNum - 1U) / coreNum;
        // 用配额重算「等于组」前缀偏移（覆盖 meta[2+coreNum+c]）：PerCoreOffsets 内部
        // 写的仍是起点、推进量 = min(cnt, quota) ⇒ 前缀稠密，与写出侧截断口径一致。
        off = PerCoreOffsets(subHistGm_, coreNum, meta, 2U + coreNum, totalLess, subMask, static_cast<int32_t>(quota));
    }
    meta.SetValue(1U, static_cast<int32_t>(bStar2));
    meta.SetValue(2U + 2U * coreNum + 1U, static_cast<int32_t>(off - totalLess));
    meta.SetValue(2U + 2U * coreNum + 2U, static_cast<int32_t>(off));
    meta.SetValue(2U + 2U * coreNum + 3U, static_cast<int32_t>(quota)); // 0 = 不限制
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(topkMetaGm_, meta, mp);
    PipeBarrier<PIPE_MTE3>();
}

// ===== 阶段 2b-E（仅 0 号核）：保留集（≈K）基数排序 + 贪心选取 =====
template <typename T>
__aicore__ inline void GoodFeaturesToTrack<T>::SelectFinal(uint32_t image, uint32_t coreNum)
{
    LocalTensor<int32_t> meta = metaBuf_.Get<int32_t>();
    DataCopyParams mp;
    mp.blockCount = 1;
    mp.blockLen = (coreNum + 2U) * sizeof(int32_t);
    mp.srcStride = 0;
    mp.dstStride = 0;
    DataCopyPad(meta, topkMetaGm_, mp, {false, 0, 0, 0});
    PipeBarrier<PIPE_MTE2>();
    const uint32_t candCount = static_cast<uint32_t>(meta.GetValue(2U + 2U * coreNum + 2U));
    const int32_t w = static_cast<int32_t>(tiling_->width);
    const uint64_t candBase = static_cast<uint64_t>(image) * tiling_->imageElements;
    const uint64_t outputBase = static_cast<uint64_t>(image) * tiling_->outputCapacity * 2U;
    // 保留集在 2 号缓冲，4 趟 8bit LSD（稳定）后回到 2 号缓冲：2→1→2→1→2
    if (candCount > 1U) {
        LocalTensor<int32_t> hist = histBuf_.Get<int32_t>();
        RadixPass(candVal2Gm_, candIdx2Gm_, candValGm_, candIdxGm_, candBase, candCount, 0, hist);
        RadixPass(candValGm_, candIdxGm_, candVal2Gm_, candIdx2Gm_, candBase, candCount, 8, hist);
        RadixPass(candVal2Gm_, candIdx2Gm_, candValGm_, candIdxGm_, candBase, candCount, 16, hist);
        RadixPass(candValGm_, candIdxGm_, candVal2Gm_, candIdx2Gm_, candBase, candCount, 24, hist);
    }
    // greedy 从高到低选取（FarEnough minDistance 抑制）
    // minDistance <= 1 时：角点坐标是整数，(dy,dx) 非全零 ⇒ dy²+dx² >= 1 >= minDistance²，
    // FarEnough 恒为真，可直接跳过——省掉每次 O(selected) 的 GM 读。
    const bool noSuppress = (tiling_->minDistance <= 1.0F);
    uint32_t selected = 0U;
    for (uint32_t c = 0U; c < candCount && selected < tiling_->outputCapacity; ++c) {
        const uint32_t idx = candIdx2Gm_.GetValue(candBase + c);
        const int32_t y = static_cast<int32_t>(idx / static_cast<uint32_t>(w));
        const int32_t x = static_cast<int32_t>(idx - static_cast<uint32_t>(y) * w);
        if (noSuppress || FarEnough(image, y, x, selected)) {
            // 对齐 OpenCV cv::goodFeaturesToTrack：每行按 (x, y) 写出
            cornersGm_.SetValue(outputBase + 2U * selected, static_cast<float>(x));
            cornersGm_.SetValue(outputBase + 2U * selected + 1U, static_cast<float>(y));
            ++selected;
        }
    }
    numCornersGm_.SetValue(image, static_cast<int32_t>(selected));
}
// 取 3 行邻域的 fp32 视图：fp16 输入先 Cast 到 fp32 暂存（scr 需 3*w 个 float，
// 复用阶段 1 空闲的 vsumBuf_）。
// 用函数模板特化做编译期分派：fp32 走特化版（零开销），同时避免对 fp32 生成非法的 half→float Cast。
template <typename TT>
__aicore__ inline void RowViewsFp32(LocalTensor<float> scr, LocalTensor<TT> a, LocalTensor<TT> b, LocalTensor<TT> c,
                                    uint32_t w, LocalTensor<float>& pa, LocalTensor<float>& pb, LocalTensor<float>& pc)
{
    Cast(scr[0U], a, RoundMode::CAST_NONE, w);
    Cast(scr[w], b, RoundMode::CAST_NONE, w);
    Cast(scr[2U * w], c, RoundMode::CAST_NONE, w);
    PipeBarrier<PIPE_V>();
    pa = scr[0U];
    pb = scr[w];
    pc = scr[2U * w];
}

template <>
__aicore__ inline void RowViewsFp32<float>(LocalTensor<float> /*scr*/, LocalTensor<float> a, LocalTensor<float> b,
                                           LocalTensor<float> c, uint32_t /*w*/, LocalTensor<float>& pa,
                                           LocalTensor<float>& pb, LocalTensor<float>& pc)
{
    pa = a;
    pb = b;
    pc = c;
}

template <typename T>
__aicore__ inline void GoodFeaturesToTrack<T>::Process()
{
    const int32_t h = static_cast<int32_t>(tiling_->height);
    const int32_t wAbs = static_cast<int32_t>(tiling_->width); // 图像绝对宽度（响应分支内部会另起 w）
    const int32_t radius = tiling_->blockSize / 2;
    const int32_t halo = radius + 1; // Response 的 Sobel 邻域最大行偏移
    const int32_t totalRows = static_cast<int32_t>(tiling_->batch) * h;
    const uint32_t coreIdx = GetBlockIdx();
    const int32_t coreNum = static_cast<int32_t>(tiling_->coreNum);

    // UB 预算：图像行带 + 梯度(3通道) + 响应暂存。
    // 单行长度上限：本文件的行搬运一律走 `DataCopyPad` + `DataCopyParams`，其 `blockLen` 以
    // **字节** 为单位、上限 65535；`DataCopy`（非 Pad）的 blockLen 才是以 32B 为单位。
    // 该上限的强制性由 Host Tiling 保证：`4*W > 65535`（即 W > 16383）时 Tiling 直接返回失败，
    // 因此本文件所有按行搬运的 `blockLen`（4*W 字节）都不会被 uint16 截断——包括 useUb=false
    // 的回退路径（该路径没有其他宽度防护，宽度正确性不能依赖它）。
    // 下面的 UB 预算只决定 useUb 分支是否可行，不承担宽度正确性：
    //   r=1 时每行带需 (coreRows+2*halo)*imgRow + (coreRows+2*r)*gradRow + coreRows*respRow + vsumRow
    //   ⇒ fp32 上限 w≈2275、fp16≈2642
    const int32_t kUbBudgetBytes = 160 * 1024;
    const int32_t kMaxBlockBytes = 65535;

    // ===================== 阶段 1 的两套几何 =====================
    // 方案 A（整行）：整行常驻 UB，行宽 = 对齐后的 wAbs。w 大到放不下时转方案 B。
    // 方案 B（列分块）：把行按列切成 colTileW 宽的小块，每块左右各带 colHalo 列 halo；
    //   块内坐标 u ∈ [0, colTileW + 2*colHalo)，u=0 对应图像列 colBase = tx0 - colHalo。
    //   图像外的列一律补 0 ⇒ 块内的 Sobel/窗口与「图像外=0」语义严格等价，
    //   因此块内**不需要任何 isEdge 分支**。
    const int32_t colHalo = ((radius + 9) + 7) & (~7); // 8 的倍数且 >= radius+9

    const int32_t fullRowBytes = wAbs * static_cast<int32_t>(sizeof(T));
    int32_t fullCoreRows = 0;
    {
        const int32_t fb = (fullRowBytes + 31) & (~31);
        const int32_t fs = fb / static_cast<int32_t>(sizeof(T));
        const int32_t fg = 3 * fs * static_cast<int32_t>(sizeof(float));
        const int32_t fr = fs * static_cast<int32_t>(sizeof(float));
        fullCoreRows = (kUbBudgetBytes - (2 * halo * fb + 2 * radius * fg + fg)) / (fb + fg + fr);
        if (fullCoreRows < 1 || fullRowBytes > kMaxBlockBytes) {
            fullCoreRows = 0; // 整行放不下（超宽图）：转列分块
        }
    }

    // 列分块必须先扣掉 2a staging 的需求（2a 复用 gradBuf_，且必须整行宽 ⇒ 3 行 × wAbs × 4B；
    // 不扣会让 2a 越界读写）。再从大到小挑 tileW：先求 coreRows>=4 的档
    // （Sobel 只多算 (coreRows+2r)/coreRows 倍），不行再退而求其次。
    const int32_t stage2aBytes = 3 * wAbs * static_cast<int32_t>(sizeof(float));
    int32_t colTileW = 0;
    int32_t colCoreRows = 0;
    if (fullCoreRows == 0 && stage2aBytes < kUbBudgetBytes) {
        const int32_t colBudget = kUbBudgetBytes - stage2aBytes;
        for (int32_t pass = 0; pass < 2 && colTileW == 0; ++pass) {
            const int32_t needRows = (pass == 0) ? 4 : 1;
            for (int32_t cand = 2048; cand >= 128; cand -= 128) {
                const int32_t cb = ((cand + 2 * colHalo) * static_cast<int32_t>(sizeof(T)) + 31) & (~31);
                const int32_t cs = cb / static_cast<int32_t>(sizeof(T));
                const int32_t cg = 3 * cs * static_cast<int32_t>(sizeof(float));
                const int32_t cr = cs * static_cast<int32_t>(sizeof(float));
                const int32_t rows = (colBudget - (2 * halo * cb + 2 * radius * cg + cg)) / (cb + cg + cr);
                if (rows >= needRows) {
                    colTileW = cand;
                    colCoreRows = rows;
                    break;
                }
            }
        }
    }
    // 列分块对不齐 32B 时没有意义（tileW 与 colHalo 都是 8 的倍数 ⇒ 行字节天然 32B 对齐）
    const bool colTiled = (colTileW > 0);
    const int32_t wUb = colTiled ? (colTileW + 2 * colHalo) : wAbs; // UB 行宽（元素）
    int32_t coreRowsMax = colTiled ? colCoreRows : fullCoreRows;

    const int32_t rowBytes = wUb * static_cast<int32_t>(sizeof(T));
    const int32_t respRowBytes = wUb * static_cast<int32_t>(sizeof(float));
    // UB 中的行距：DataCopyPad 搬运时会把不足 32B 的尾部补齐到 32B，所以 UB 行距恒为 align32(行字节)。
    // 所有 UB 内索引都用 ubStride_（元素单位），而不是 w ⇒ w 非 8 对齐时同样可以走 UB 快路径
    const int32_t ubRowBytes = (rowBytes + 31) & (~31);
    const int32_t ubStride = ubRowBytes / static_cast<int32_t>(sizeof(T));
    ubStride_ = ubStride;
    // 两者决定 DMA 走「整块多行拷贝」还是「逐行拷贝」（见成员注释）
    imgRowsPadded_ = (ubRowBytes != rowBytes);
    respRowsPadded_ = (ubStride * static_cast<int32_t>(sizeof(float)) != respRowBytes);
    const int32_t gradRowBytes = 3 * ubStride * static_cast<int32_t>(sizeof(float)); // gx²,gy²,gxy
    const int32_t respUbRowBytes = ubStride * static_cast<int32_t>(sizeof(float));
    const int32_t vsumRowBytes = 3 * ubStride * static_cast<int32_t>(sizeof(float)); // 垂直窗口和（单行 3 通道）
    // 行带缓冲按「实际所需行数」分配，而不是三块都按同一上限行数分配：
    //   图像需 coreRows + 2*halo 行、梯度需 coreRows + 2*radius 行、响应仅需 coreRows 行、vsum 恒 1 行。
    const bool ubUsable = (coreRowsMax >= 1);
    const int32_t imageRows = ubUsable ? (coreRowsMax + 2 * halo) : 1;
    const int32_t gradRows = ubUsable ? (coreRowsMax + 2 * radius) : 1;
    const int32_t respRowsAlloc = ubUsable ? coreRowsMax : 1;
    // 梯度缓冲容量：既要放阶段 1 的梯度环，又要能当 2a 候选扫描的 staging。
    // 列分块下 ubStride_ ≈ tileW ≪ wAbs，梯度环所需字节数会小于「3 行整行宽」的 staging 需求
    // ⇒ 取两者较大值；否则 scanStageRows 折算出的行数装不下、2a 会越界读写。
    int32_t gradBytes = gradRows * gradRowBytes;
    if (colTiled && gradBytes < stage2aBytes) {
        gradBytes = stage2aBytes;
    }
    pipe.InitBuffer(imageBuf_, static_cast<uint32_t>(imageRows * ubRowBytes));
    pipe.InitBuffer(gradBuf_, static_cast<uint32_t>(gradBytes));
    pipe.InitBuffer(respBuf_, static_cast<uint32_t>(respRowsAlloc * respUbRowBytes));
    pipe.InitBuffer(vsumBuf_, static_cast<uint32_t>(vsumRowBytes));
    // 候选扫描的 staging 复用阶段 1 结束后已死的梯度缓冲，按「wAbs 个 float 一行」折算行数。
    // 必须按字节反算（不能写死 3*gradRows）：列分块下 ubStride_ ≠ wAbs，写死会超出缓冲。
    const int32_t scanStageRows = colTiled ? (gradBytes / (wAbs * static_cast<int32_t>(sizeof(float)))) :
                                             (3 * gradRows);
    // 列分块修正：垂直窗口需要 coreRows+2*radius 行梯度同时在环里，否则会读到被后续行
    // 覆盖的槽 ⇒ 响应串扰 ⇒ 角点成倍增多。把每块行数夹到环深能容纳的范围内。
    if (colTiled) {
        const int32_t ringDepth = gradBytes / (wAbs * static_cast<int32_t>(sizeof(float)));
        const int32_t ringNeed = coreRowsMax + 2 * radius;
        if (ringNeed > ringDepth) {
            coreRowsMax = (ringDepth > 2 * radius) ? (ringDepth - 2 * radius) : 1;
        }
        if (coreRowsMax < 1) {
            coreRowsMax = 1;
        }
    }
    pipe.InitBuffer(histBuf_, 256U * sizeof(int32_t));
    // 并行 TopK：本核候选键高 8 位直方图（阶段 2a 累加）+ 跨核元数据中转（b*/偏移，MTE2 读回）
    pipe.InitBuffer(keyHistBuf_, 256U * sizeof(float));
    // 元数据 2*coreNum+8 个 int32（8*coreNum+32 字节，本身即 32B 对齐）
    pipe.InitBuffer(metaBuf_, (2U * static_cast<uint32_t>(coreNum) + 8U) * sizeof(int32_t));
    // 二级直方图（键 bit16-23，只统计 bucket==b* 的候选；fp32）
    pipe.InitBuffer(subHistBuf_, 256U * sizeof(float));
    // ReduceSum 的 fp32 结果与临时工作区（count <= 256）
    pipe.InitBuffer(histRedBuf_, 32U * sizeof(float));
    pipe.InitBuffer(histWorkBuf_, 256U * sizeof(float));
    // 每核 8 个 float 的统计量（局部 max + 候选数）MTE2 中转缓冲
    pipe.InitBuffer(statBuf_, static_cast<uint32_t>(coreNum) * 8U * sizeof(float));
    // 候选 DMA 暂存：kCandStage 个 uint32(idx) + kCandStage 个 float(val)
    pipe.InitBuffer(candStageBuf_, kCandStage * (sizeof(uint32_t) + sizeof(float)));
    // 响应向量化：3 个窗口和分块缓冲 + 暂存（delta / trace / maxVec）
    pipe.InitBuffer(winBuf_, 3U * kWinChunk * sizeof(float));
    pipe.InitBuffer(winScrBuf_, 4U * kWinChunk * sizeof(float));
    // Sobel 向量化：8 个 Gather 目标槽 + gx/gy（pB 兼临时）
    pipe.InitBuffer(sobVecBuf_, 9U * kWinChunk * sizeof(float));
    pipe.InitBuffer(sobOffBuf_, kWinChunk * sizeof(uint32_t));
    // Gather 的字节偏移表 off[i] = i*4（元素 i 的字节偏移），只填一次
    {
        LocalTensor<uint32_t> sobOff = sobOffBuf_.Get<uint32_t>();
        for (uint32_t i = 0U; i < kWinChunk; ++i) {
            sobOff.SetValue(i, i * static_cast<uint32_t>(sizeof(float)));
        }
        PipeBarrier<PIPE_ALL>();
    }

    // 响应计算：每核负责自己的行带 [rowStart, rowEnd)，配合行带 UB 缓存加速标量读；
    // 响应先暂存 UB，再 DataCopyPad(DMA) 批量写回 GM → 跨核可见
    const int32_t rowStart = static_cast<int32_t>(static_cast<int64_t>(coreIdx) * totalRows / coreNum);
    const int32_t rowEnd = static_cast<int32_t>(static_cast<int64_t>(coreIdx + 1) * totalRows / coreNum);

    // ---- 梯度环（跨 block 持久）----
    // grad 行 py 常驻下标 (py % gradRingRows)，响应的垂直窗口按环取用。
    // 环深取 coreRowsMax + 2*radius 的两个理由：
    //   (a) 单个 block 的 grad 范围恰为 coreRows + 2*radius 行（<= 环深），可同时驻留；
    //   (b) 相邻 block 需要复用的旧行，行号差最大 coreRows-1+2*radius < 环深，不会被新写入覆盖。
    const int32_t gradRingRows = (coreRowsMax < 1) ? 1 : (coreRowsMax + 2 * radius);
    gradRingRows_ = gradRingRows;
    gradValidHi_ = -1; // 环中已计算的最大 image 行（-1 = 尚未计算任何行）

    // 逐图统计：行带可能跨多张图，进入新 image 时先把上一张图的局部 max 写回统计槽
    float coreMax = 0.0F;   // 当前 image 的局部最大响应
    uint32_t maxImage = 0U; // coreMax 当前归属的 image
    bool maxInited = false; // 是否已有归属（首个 image 之前为 false）
    int32_t row = rowStart;
    while (row < rowEnd) {
        const int32_t image = row / h;
        if (!maxInited || static_cast<uint32_t>(image) != maxImage) {
            if (maxInited) {
                WriteCoreMax(maxImage, coreIdx, coreMax);
            }
            // 梯度环缓存以「图内 y」为键（见 gradValidHi_ 定义）：换图后 y 从 0 重新开始，
            // 不失效的话 image 1 会被判定为「0..h-1 都已算过」⇒ 直接复用上一张图的梯度，
            // 导致第二张图起输出与第一张完全相同（batch>1 且行带跨图时必现）。
            gradValidHi_ = -1;
            ubCacheOn_ = false;
            maxImage = static_cast<uint32_t>(image);
            coreMax = 0.0F;
            maxInited = true;
        }
        const int32_t yBase = row - image * h;
        // 本 image 内本核可处理的最大行数（不跨 image）
        const int32_t maxInImage = GftMin(rowEnd, (image + 1) * h) - row;
        int32_t coreRows = GftMin(maxInImage, coreRowsMax);
        if (coreRows < 1) {
            coreRows = 1;
        }
        const int32_t cacheY0 = GftMax(0, yBase - halo);
        const int32_t cacheY1 = GftMin(h, yBase + coreRows + halo);
        const int32_t cacheRows = cacheY1 - cacheY0;

        // 行带能放入 UB 且单行长度满足块拷贝上限时用 UB 缓存，否则回退 GM。
        // 缓冲区已按所需行数分配 ⇒ cacheRows <= coreRows + 2*halo <= imageRows 恒成立。
        // 行宽不要求 32B 对齐：DataCopyPad 会把每行尾部补齐到 32B，UB 行距 = ubRowBytes，
        // 索引统一用 ubStride_（见其定义），因此 w 非 8 对齐时同样走快路径。
        const bool useUb = ubUsable;
        // 列分块下 Pixel()/SobelGx/SobelGy 的 UB 缓存分支按「整行」坐标工作、不认分块列偏移
        // ⇒ 关掉（只影响 y=0 与 y=h-1 两行，代价可忽略）。
        ubCacheOn_ = useUb && !colTiled;
        ubImage_ = image;
        ubRowStart_ = cacheY0;
        ubRows_ = cacheRows;
        // ===================== 列分块主循环 =====================
        // 方案 A：tileStep = wAbs ⇒ 恰好循环一次。
        // 方案 B：按 colTileW 逐块处理，每块独立完成「图像拷贝 → Sobel → 垂直窗口 → 响应 → 写回」。
        //   块内列坐标 u：0 ─ colHalo ─(本块输出 wT 列)─ colHalo+wT ─ wUb-1
        //   对应图像列 colBase+u（colBase = tx0 - colHalo）；图像外的列一律补 0。
        const int32_t tileStep = colTiled ? colTileW : wAbs;
        for (int32_t tx0 = 0; tx0 < wAbs; tx0 += tileStep) {
            const int32_t wT = GftMin(wAbs - tx0, tileStep);
            const int32_t colBase = colTiled ? (tx0 - colHalo) : 0;
            // 本块「计算宽度」：方案 A = 图像宽；方案 B = 载入缓冲宽度
            // （= colTileW + 2*colHalo，必为 8 的倍数 ⇒ 恒走向量 Sobel）。
            const int32_t w = colTiled ? ubStride : wAbs;
            // 本块输出列在缓冲中的范围 [uLo, uHi)。方案 A 时 = [0, w)。
            // 响应/max 只在 [uLo,uHi) 上做 ⇒ 列分块时 halo 列的无效值不会污染 maxResponse；
            // uLo = colHalo 是 8 的倍数 ⇒ 向量响应的 respLocal[yOff+uLo] 仍满足 32B 对齐
            // （yOff = 行号×ubStride，ubStride 也是 8 的倍数）。
            const int32_t uLo = colTiled ? colHalo : 0;
            const int32_t uHi = colTiled ? (colHalo + wT) : w;
            if (colTiled) {
                gradValidHi_ = -1; // 换列后环里是上一块的列数据，必须重算
                // 上一块对 imgLocal_/gradLocal_/respLocal 的读写（向量 Sobel 的 Gather、标量窗口、
                // MTE3 写回）必须全部结束，才允许本块的 DMA/向量写覆盖同一批缓冲（WAR）。
                // 少了这一条会读到上一块的残值 ⇒ 响应整片异常。
                PipeBarrier<PIPE_ALL>();
            }
            if (useUb) {
                imgLocal_ = imageBuf_.Get<T>();
                // 行字节数本身 32B 对齐时（UB 行距 == 原始行字节），可一次多行块拷贝；
                // 否则（UB 行距 > 原始行字节，即行尾需要补齐）改用**逐行拷贝**：
                // 多行 DataCopyPad 在 UB 侧的落地行距语义不明确（是否按 32B 补齐取决于实现），
                // 逐行拷贝把每行的 UB 基址显式写成 r*ubStride_，与索引定义严格一致，无歧义。
                const uint32_t imgBase = static_cast<uint32_t>(image) * tiling_->imageElements +
                                         static_cast<uint32_t>(cacheY0) * tiling_->width;
                if (colTiled) {
                    // 列分块：逐行只搬本块覆盖的有效列 [lx0, lx1)，缓冲内落不到有效数据的列（图像外）补 0。
                    // 补 0 后块内**无需任何 isEdge 判断**：块内 Sobel/窗口与「图像外=0」语义严格等价。
                    const int32_t lx0 = GftMax(0, colBase);
                    const int32_t lx1 = GftMin(wAbs, colBase + ubStride);
                    const int32_t lw = lx1 - lx0;
                    const int32_t uStart = lx0 - colBase; // 有效列在缓冲中的起始列
                    if (lw > 0) {
                        for (int32_t r = 0; r < cacheRows; ++r) {
                            DataCopyParams rowParams;
                            rowParams.blockCount = 1;
                            rowParams.blockLen = static_cast<uint16_t>(lw * static_cast<int32_t>(sizeof(T)));
                            rowParams.srcStride = 0;
                            rowParams.dstStride = 0;
                            DataCopyPad(
                                imgLocal_[static_cast<uint32_t>(r) * static_cast<uint32_t>(ubStride_) +
                                          static_cast<uint32_t>(uStart)],
                                imageGm_[imgBase + static_cast<uint32_t>(r) * static_cast<uint32_t>(tiling_->width) +
                                         static_cast<uint32_t>(lx0)],
                                rowParams, {false, 0, 0, 0});
                        }
                    }
                    PipeBarrier<PIPE_MTE2>(); // 等 DMA 落地
                    if (uStart > 0 || (uStart + lw) < ubStride) {
                        for (int32_t r = 0; r < cacheRows; ++r) {
                            const uint32_t rowOff = static_cast<uint32_t>(r) * static_cast<uint32_t>(ubStride_);
                            for (int32_t u = 0; u < uStart; ++u) {
                                imgLocal_.SetValue(rowOff + static_cast<uint32_t>(u), static_cast<T>(0));
                            }
                            for (int32_t u = uStart + lw; u < ubStride; ++u) {
                                imgLocal_.SetValue(rowOff + static_cast<uint32_t>(u), static_cast<T>(0));
                            }
                        }
                        PipeBarrier<PIPE_ALL>(); // 标量写 + MTE2 写都要对后续向量读可见
                    }
                } else if (imgRowsPadded_) {
                    for (int32_t r = 0; r < cacheRows; ++r) {
                        DataCopyParams rowParams;
                        rowParams.blockCount = 1;
                        rowParams.blockLen = static_cast<uint16_t>(rowBytes);
                        rowParams.srcStride = 0;
                        rowParams.dstStride = 0;
                        DataCopyPad(imgLocal_[static_cast<uint32_t>(r) * static_cast<uint32_t>(ubStride_)],
                                    imageGm_[imgBase + static_cast<uint32_t>(r) * tiling_->width], rowParams,
                                    {false, 0, 0, 0});
                    }
                } else {
                    DataCopyParams copyParams;
                    copyParams.blockCount = static_cast<uint16_t>(cacheRows); // 每行一个 block
                    copyParams.blockLen = static_cast<uint16_t>(rowBytes);    // 单行字节数（≤8160B）
                    copyParams.srcStride = 0;                                 // GM 行间连续，无间隔
                    copyParams.dstStride = 0;                                 // UB 行间连续，无间隔
                    DataCopyPad(imgLocal_, imageGm_[imgBase], copyParams, {false, 0, 0, 0});
                }
                PipeBarrier<PIPE_MTE2>(); // 等待 MTE2 拷贝完成后再标量读取
            }
            useGradCache_ = useUb;
            gradLocal_ = gradBuf_.Get<float>();
            // 本块所需的 grad 行范围：输出行 y∈[yBase, yBase+coreRows-1] 的垂直窗口需要
            // grad 行 [y-radius, y+radius]，合并即 [yBase-radius, yBase+coreRows-1+radius]。
            // 这些行在环中的下标为 py % gradRingRows（不是 py - 基址），因此上一 block 已算过的行
            // 仍然有效，只需补算 gradValidHi_+1 之后的新行。
            const int32_t gradRowStart = GftMax(0, yBase - radius);
            const int32_t gradRowLast = GftMin(h - 1, yBase + coreRows - 1 + radius);
            const int32_t gradFrom = GftMax(gradRowStart, gradValidHi_ + 1);
            // 预计算新增的 Sobel 梯度平方（gx²/gy²/gxy），供响应垂直窗口(vsum)累加复用。
            // 每像素只算一次，避免 Response 窗口内 blockSize² 次重复 Sobel。
            // 优化：非贴图像上下边界（py-1 与 py+1 均在图像内）时，3×3 邻域全在 imgLocal_ 内，
            //      内联直读避免 Pixel 的逐像素边界分支 + 函数调用；贴边界的行用 Pixel 版 Sobel
            //      （图像外行按 0 处理，与原 Response 语义一致）。
            // 在 910B 要求 32B 对齐，偏移视图触发 vector core exception；Copy/Gather calCount 版不支持 2201。
            if (useUb) {
                const uint32_t uW = static_cast<uint32_t>(ubStride_);
                const uint32_t gStride = 3U * uW;
                for (int32_t py = gradFrom; py <= gradRowLast; ++py) {
                    LocalTensor<float> gradRow = gradLocal_[static_cast<uint32_t>(py % gradRingRows) * gStride];
                    if (py >= 1 && py <= h - 2) {
                        // 邻域行 py-1/py/py+1 全在图像内（也全在本次 imgLocal_ 缓存内）：内联直读
                        const int32_t ly = py - cacheY0;
                        LocalTensor<T> rPrev = imgLocal_[static_cast<uint32_t>(ly - 1) * uW];
                        LocalTensor<T> rCur = imgLocal_[static_cast<uint32_t>(ly) * uW];
                        LocalTensor<T> rNext = imgLocal_[static_cast<uint32_t>(ly + 1) * uW];
                        if ((w & 7) == 0 && w >= 24) {
                            // ===== 向量化 Sobel：Gather 取 8 个移位副本，一次算一块 =====
                            // 与水平滑窗不同：Sobel 每个输出像素彼此独立（无跳位置累加），
                            // 且向量算式的结合顺序与标量逐字一致 ⇒ 结果逐位不变。
                            // 要求 w % 8 == 0：保证 3 个通道目标地址与分块起点都是 32B 对齐；
                            // 向量段取 [vecStart,vecEnd) ⊆ [1, w-2] ⇒ Gather 的字节下标永远不会越出本行。
                            LocalTensor<float> prevF;
                            LocalTensor<float> curF;
                            LocalTensor<float> nextF;
                            RowViewsFp32<T>(vsumBuf_.Get<float>(), rPrev, rCur, rNext, uW, prevF, curF, nextF);
                            LocalTensor<uint32_t> sobOff = sobOffBuf_.Get<uint32_t>();
                            LocalTensor<float> sv = sobVecBuf_.Get<float>();
                            LocalTensor<float> pA = sv[0U];
                            LocalTensor<float> pB = sv[1U * kWinChunk]; // 兼作 Muls 临时
                            LocalTensor<float> pC = sv[2U * kWinChunk];
                            LocalTensor<float> cA = sv[3U * kWinChunk];
                            LocalTensor<float> cC = sv[4U * kWinChunk];
                            LocalTensor<float> nA = sv[5U * kWinChunk];
                            LocalTensor<float> nC = sv[6U * kWinChunk];
                            LocalTensor<float> gxv = sv[7U * kWinChunk];
                            LocalTensor<float> gyv = sv[8U * kWinChunk];
                            const int32_t vecStart = 8;
                            const int32_t vecEndRaw = (w - 1) & (~7); // ≤ w-1 的最大 8 倍数
                            const int32_t vecEnd = (vecEndRaw > vecStart) ? vecEndRaw : vecStart;
                            for (int32_t pass = 0; pass < 2; ++pass) { // 两端标量像素（每行 ≤ 16 个）
                                const int32_t xb = (pass == 0) ? 0 : vecEnd;
                                const int32_t xe = (pass == 0) ? vecStart : w;
                                for (int32_t x = xb; x < xe; ++x) {
                                    const uint32_t ux = static_cast<uint32_t>(x);
                                    const bool hasL = (x > 0);
                                    const bool hasR = (x < w - 1);
                                    const float v00 = hasL ? static_cast<float>(rPrev.GetValue(ux - 1U)) : 0.0F;
                                    const float v01 = static_cast<float>(rPrev.GetValue(ux));
                                    const float v02 = hasR ? static_cast<float>(rPrev.GetValue(ux + 1U)) : 0.0F;
                                    const float v10 = hasL ? static_cast<float>(rCur.GetValue(ux - 1U)) : 0.0F;
                                    const float v12 = hasR ? static_cast<float>(rCur.GetValue(ux + 1U)) : 0.0F;
                                    const float v20 = hasL ? static_cast<float>(rNext.GetValue(ux - 1U)) : 0.0F;
                                    const float v21 = static_cast<float>(rNext.GetValue(ux));
                                    const float v22 = hasR ? static_cast<float>(rNext.GetValue(ux + 1U)) : 0.0F;
                                    const float gx = -v00 + v02 - 2.0F * v10 + 2.0F * v12 - v20 + v22;
                                    const float gy = -v00 - 2.0F * v01 - v02 + v20 + 2.0F * v21 + v22;
                                    gradRow.SetValue(ux, gx * gx);
                                    gradRow.SetValue(uW + ux, gy * gy);
                                    gradRow.SetValue(2U * uW + ux, gx * gy);
                                }
                            }
                            PipeBarrier<PIPE_ALL>(); // 标量写边缘 → 向量写同一行带
                            for (int32_t x0 = vecStart; x0 < vecEnd; x0 += static_cast<int32_t>(kWinChunk)) {
                                int32_t cnt = vecEnd - x0;
                                if (cnt > static_cast<int32_t>(kWinChunk)) {
                                    cnt = static_cast<int32_t>(kWinChunk);
                                }
                                const uint32_t uCnt = static_cast<uint32_t>(cnt);
                                const uint32_t bA = static_cast<uint32_t>((x0 - 1) * 4); // P/C/N 的 x-1 列
                                const uint32_t bB = static_cast<uint32_t>(x0 * 4);       // x 列
                                const uint32_t bC = static_cast<uint32_t>((x0 + 1) * 4); // x+1 列
                                Gather(pA, prevF, sobOff, bA, uCnt);
                                Gather(pC, prevF, sobOff, bC, uCnt);
                                Gather(cA, curF, sobOff, bA, uCnt);
                                Gather(cC, curF, sobOff, bC, uCnt);
                                Gather(nA, nextF, sobOff, bA, uCnt);
                                Gather(nC, nextF, sobOff, bC, uCnt);
                                // gx = -P[x-1] + P[x+1] - 2*C[x-1] + 2*C[x+1] - N[x-1] + N[x+1]
                                Muls(gxv, pA, -1.0F, uCnt);
                                Add(gxv, gxv, pC, uCnt);
                                Muls(pB, cA, 2.0F, uCnt);
                                Sub(gxv, gxv, pB, uCnt);
                                Muls(pB, cC, 2.0F, uCnt);
                                Add(gxv, gxv, pB, uCnt);
                                Sub(gxv, gxv, nA, uCnt);
                                Add(gxv, gxv, nC, uCnt);
                                // cA/cC 已用完，改用它们装 P[x] / N[x]
                                Gather(cA, prevF, sobOff, bB, uCnt);
                                Gather(cC, nextF, sobOff, bB, uCnt);
                                // gy = -P[x-1] - 2*P[x] - P[x+1] + N[x-1] + 2*N[x] + N[x+1]
                                Muls(gyv, pA, -1.0F, uCnt);
                                Muls(pB, cA, 2.0F, uCnt);
                                Sub(gyv, gyv, pB, uCnt);
                                Sub(gyv, gyv, pC, uCnt);
                                Add(gyv, gyv, nA, uCnt);
                                Muls(pB, cC, 2.0F, uCnt);
                                Add(gyv, gyv, pB, uCnt);
                                Add(gyv, gyv, nC, uCnt);
                                Mul(gradRow[2U * uW + static_cast<uint32_t>(x0)], gxv, gyv, uCnt); // gx*gy
                                Mul(gradRow[static_cast<uint32_t>(x0)], gxv, gxv, uCnt);           // gx²
                                Mul(gradRow[uW + static_cast<uint32_t>(x0)], gyv, gyv, uCnt);      // gy²
                                PipeBarrier<PIPE_V>();
                            }
                        } else {
                            // ---- 行内滑动复用：UB load 从 8 次/像素降到 4 次/像素 ----
                            // x 处需要 P[x-1],P[x],P[x+1] / C[x-1],C[x+1] / N[x-1],N[x],N[x+1]；
                            // 推进到 x+1 需要 P[x],P[x+1],P[x+2] / C[x],C[x+2] / N[x],N[x+1],N[x+2]。
                            // ⇒ P、N 各复用 2 个（只需新读 P[x+2]、N[x+2]）；
                            //    C 需要的列号 x-1→x、x+1→x+2 奇偶翻转，4 个下标两两不同，无法复用（新读 2 个）。
                            // 越界列（x-1<0 或 x+1>=w）用预置 0 表示，等价于 hasL/hasR 三元判断，
                            // 顺带省掉每像素 2 次比较 + 8 次选择。
                            // gx/gy 表达式保持原始结合顺序 ⇒ 浮点结果逐位不变。
                            float p0 = 0.0F, p1 = 0.0F, p2 = 0.0F; // P[x-1], P[x], P[x+1]
                            float c0 = 0.0F, c2 = 0.0F;            // C[x-1], C[x+1]
                            float n0 = 0.0F, n1 = 0.0F, n2 = 0.0F; // N[x-1], N[x], N[x+1]
                            p1 = static_cast<float>(rPrev.GetValue(0U));
                            n1 = static_cast<float>(rNext.GetValue(0U));
                            if (w > 1) {
                                p2 = static_cast<float>(rPrev.GetValue(1U));
                                c2 = static_cast<float>(rCur.GetValue(1U));
                                n2 = static_cast<float>(rNext.GetValue(1U));
                            }
                            for (int32_t x = 0; x < w; ++x) {
                                const uint32_t ux = static_cast<uint32_t>(x);
                                // SobelGx 核 [-1 0 1; -2 0 2; -1 0 1]
                                const float gx = -p0 + p2 - 2.0F * c0 + 2.0F * c2 - n0 + n2;
                                // SobelGy 核 [-1 -2 -1; 0 0 0; 1 2 1]
                                const float gy = -p0 - 2.0F * p1 - p2 + n0 + 2.0F * n1 + n2;
                                gradRow.SetValue(ux, gx * gx);
                                gradRow.SetValue(uW + ux, gy * gy);
                                gradRow.SetValue(2U * uW + ux, gx * gy);
                                // 推进到 x+1
                                p0 = p1;
                                p1 = p2;
                                n0 = n1;
                                n1 = n2;
                                c0 = static_cast<float>(rCur.GetValue(ux)); // C[x] → 下一轮的 C[(x+1)-1]
                                if (x + 2 < w) {
                                    const uint32_t u2 = ux + 2U;
                                    p2 = static_cast<float>(rPrev.GetValue(u2));
                                    c2 = static_cast<float>(rCur.GetValue(u2));
                                    n2 = static_cast<float>(rNext.GetValue(u2));
                                } else {
                                    p2 = 0.0F;
                                    c2 = 0.0F;
                                    n2 = 0.0F;
                                }
                            }
                        }
                    } else {
                        // 贴图像上/下边界：邻域行 py-1 或 py+1 越界，按 0 处理
                        for (int32_t x = 0; x < w; ++x) {
                            const int32_t absX = colBase + x;
                            float gx = 0.0F;
                            float gy = 0.0F;
                            if (absX >= 0 && absX < wAbs) { // 列分块的 halo 列可能落到图像外 ⇒ 视为 0
                                gx = SobelGx(image, py, absX);
                                gy = SobelGy(image, py, absX);
                            }
                            const uint32_t ux = static_cast<uint32_t>(x);
                            gradRow.SetValue(ux, gx * gx);
                            gradRow.SetValue(uW + ux, gy * gy);
                            gradRow.SetValue(2U * uW + ux, gx * gy);
                        }
                    }
                }
                if (gradRowLast > gradValidHi_) {
                    gradValidHi_ = gradRowLast; // 记录环中已算到的最大 image 行，下一个 block 从其后继续
                }
            }
            // 计算本块响应：先暂存 UB，再 DataCopyPad(DMA) 批量写回 GM（DMA 走 MTE 管道保证跨核可见）
            LocalTensor<float> respLocal = respBuf_.Get<float>();
            if (useUb) {
                // 滑动窗口优化：垂直方向 3 行窗口和用向量累加；水平方向滑动窗口 O(1) 递推，
                // 避免每像素 blockSize² 次窗口重读（标量读从 27 次/像素 降到 ~6 次/像素）
                const int32_t radius = tiling_->blockSize / 2;
                const uint32_t uW = static_cast<uint32_t>(ubStride_);
                const int32_t vsumCount = 3 * ubStride; // 3 通道一起向量化（gx²,gy²,gxy）
                LocalTensor<float> vsumLocal = vsumBuf_.Get<float>();
                for (int32_t y = yBase; y < yBase + coreRows; ++y) {
                    // 垂直窗口和（向量）：vsum = Σ_{by∈[-r,r]} grad[y+by]，仅累加图像内行。
                    // 顶部/底部行带 y±radius 可能越出图像边界，越界行贡献 0（对齐原 Response 的 continue）。
                    // 注：水平方向不对 vsum 做 x 截断（保留全宽），由后续水平滑动窗口的越界判断处理。
                    Duplicate(vsumLocal, 0.0F, vsumCount); // 清零
                    for (int32_t by = -radius; by <= radius; ++by) {
                        const int32_t py = y + by;
                        if (py < 0 || py >= h) {
                            continue; // 图像外行：贡献 0
                        }
                        // 环中有效性判断按 image 行号：py 必须已算过且未被后续写入覆盖。
                        // 覆盖不会发生（见 gradRingRows 推导：复用行与新写入行的行号差 < 环深），
                        // 下面的 py > gradValidHi_ 是防御性检查。
                        if (py > gradValidHi_) {
                            continue;
                        }
                        Add(vsumLocal, vsumLocal, gradLocal_[static_cast<uint32_t>(py % gradRingRows) * 3U * uW],
                            vsumCount);
                    }
                    PipeBarrier<PIPE_V>(); // 等待向量写 vsumLocal 完成，再标量读
                    const uint32_t yOff = static_cast<uint32_t>(y - yBase) * uW;
                    if (tiling_->hasMask == 0U && tiling_->useHarris == 0U) {
                        // ===== 向量化响应（无 mask / 非 Harris 分支）=====
                        // 标量滑动窗口只负责把三个窗口和写进 UB（O(1) 递推里 1 元素移位无法向量化），
                        // 响应本身整块用向量算：Sub/Mul/Add/Sqrt/Muls。
                        // mask 与 Harris 分支走标量实现，浮点结果逐位不变。
                        LocalTensor<float> winBuf = winBuf_.Get<float>();
                        LocalTensor<float> wXX = winBuf[0];
                        LocalTensor<float> wYY = winBuf[kWinChunk];
                        LocalTensor<float> wXY = winBuf[2U * kWinChunk];
                        LocalTensor<float> scr = winScrBuf_.Get<float>();
                        LocalTensor<float> scrA = scr[0];
                        LocalTensor<float> scrB = scr[kWinChunk];
                        LocalTensor<float> maxVec = scr[2U * kWinChunk];
                        Duplicate(maxVec, 0.0F, kWinChunk);
                        float winXX = 0.0F, winYY = 0.0F, winXY = 0.0F;
                        // 窗口初值取列 uLo（非分块时 uLo=0，与原实现逐字等价：负偏移越界=0）
                        for (int32_t bu = uLo - radius; bu <= uLo + radius; ++bu) {
                            if (bu < 0 || bu >= w) {
                                continue; // 图像外列：贡献 0
                            }
                            winXX += vsumLocal.GetValue(static_cast<uint32_t>(bu));
                            winYY += vsumLocal.GetValue(uW + static_cast<uint32_t>(bu));
                            winXY += vsumLocal.GetValue(2U * uW + static_cast<uint32_t>(bu));
                        }
                        for (int32_t x0 = uLo; x0 < uHi; x0 += static_cast<int32_t>(kWinChunk)) {
                            const int32_t cnt = GftMin(static_cast<int32_t>(kWinChunk), uHi - x0);
                            // 块内分三段：左边界（x-r < 0）/ 内部（两个方向都合法）/ 右边界。
                            // 内部段去掉每像素 3 个分支 + 6 次边界比较，且严格保持
                            // 「先减 subIdx 再加 addIdx」的原始顺序 ⇒ 浮点结果逐位不变。
                            int32_t iLo = radius - x0;
                            if (iLo < 0) {
                                iLo = 0;
                            }
                            int32_t iHi = w - radius - 1 - x0;
                            if (iHi > cnt) {
                                iHi = cnt;
                            }
                            if (iHi < iLo) {
                                iHi = iLo;
                            }
                            // 左边界：x-r 越界（不减）；x+r+1 仍用 addIdx<w 判断，与原实现一致
                            for (int32_t i = 0; i < iLo; ++i) {
                                const int32_t x = x0 + i;
                                wXX.SetValue(static_cast<uint32_t>(i), winXX);
                                wYY.SetValue(static_cast<uint32_t>(i), winYY);
                                wXY.SetValue(static_cast<uint32_t>(i), winXY);
                                const int32_t addIdx = x + radius + 1;
                                if (addIdx < w) {
                                    winXX += vsumLocal.GetValue(static_cast<uint32_t>(addIdx));
                                    winYY += vsumLocal.GetValue(uW + static_cast<uint32_t>(addIdx));
                                    winXY += vsumLocal.GetValue(2U * uW + static_cast<uint32_t>(addIdx));
                                }
                            }
                            // 内部段：无任何边界分支
                            for (int32_t i = iLo; i < iHi; ++i) {
                                const int32_t x = x0 + i;
                                wXX.SetValue(static_cast<uint32_t>(i), winXX);
                                wYY.SetValue(static_cast<uint32_t>(i), winYY);
                                wXY.SetValue(static_cast<uint32_t>(i), winXY);
                                const uint32_t uSub = static_cast<uint32_t>(x - radius);
                                const uint32_t uAdd = static_cast<uint32_t>(x + radius + 1);
                                winXX -= vsumLocal.GetValue(uSub);
                                winXX += vsumLocal.GetValue(uAdd);
                                winYY -= vsumLocal.GetValue(uW + uSub);
                                winYY += vsumLocal.GetValue(uW + uAdd);
                                winXY -= vsumLocal.GetValue(2U * uW + uSub);
                                winXY += vsumLocal.GetValue(2U * uW + uAdd);
                            }
                            // 右边界：保留原实现的两个越界判断
                            for (int32_t i = iHi; i < cnt; ++i) {
                                const int32_t x = x0 + i;
                                wXX.SetValue(static_cast<uint32_t>(i), winXX);
                                wYY.SetValue(static_cast<uint32_t>(i), winYY);
                                wXY.SetValue(static_cast<uint32_t>(i), winXY);
                                if (x < w - 1) {
                                    const int32_t subIdx = x - radius;
                                    const int32_t addIdx = x + radius + 1;
                                    if (subIdx >= 0) {
                                        winXX -= vsumLocal.GetValue(static_cast<uint32_t>(subIdx));
                                        winYY -= vsumLocal.GetValue(uW + static_cast<uint32_t>(subIdx));
                                        winXY -= vsumLocal.GetValue(2U * uW + static_cast<uint32_t>(subIdx));
                                    }
                                    if (addIdx < w) {
                                        winXX += vsumLocal.GetValue(static_cast<uint32_t>(addIdx));
                                        winYY += vsumLocal.GetValue(uW + static_cast<uint32_t>(addIdx));
                                        winXY += vsumLocal.GetValue(2U * uW + static_cast<uint32_t>(addIdx));
                                    }
                                }
                            }
                            PipeBarrier<PIPE_ALL>(); // 标量写 UB → 向量读
                            // resp = 0.5*(XX+YY - sqrt((XX-YY)^2 + 4*XY^2))
                            Sub(scrA, wXX, wYY, cnt);   // delta
                            Mul(scrA, scrA, scrA, cnt); // delta^2
                            Mul(scrB, wXY, wXY, cnt);   // XY^2
                            Muls(scrB, scrB, 4.0F, cnt);
                            Add(scrA, scrA, scrB, cnt);
                            Sqrt(scrA, scrA, cnt);
                            Add(scrB, wXX, wYY, cnt); // trace
                            Sub(scrA, scrB, scrA, cnt);
                            Max(maxVec, maxVec, scrA, cnt); // 跨 chunk 保持行内最大值
                            Muls(respLocal[yOff + static_cast<uint32_t>(x0)], scrA, 0.5F, cnt);
                            PipeBarrier<PIPE_V>();
                        }
                        PipeBarrier<PIPE_ALL>(); // maxVec 向量写 → 标量读
                        for (uint32_t i = 0U; i < kWinChunk; ++i) {
                            const float v = maxVec.GetValue(i);
                            if (v > coreMax) {
                                coreMax = v;
                            }
                        }
                    } else {
                        // 水平滑动窗口（标量 O(1) 递推）：win[x] = sum_{bx∈[-r,r]} vsum[x+bx]
                        float winXX = 0.0F, winYY = 0.0F, winXY = 0.0F;
                        for (int32_t bu = uLo - radius; bu <= uLo + radius; ++bu) {
                            if (bu < 0 || bu >= w) {
                                continue; // 图像外列：贡献 0
                            }
                            winXX += vsumLocal.GetValue(static_cast<uint32_t>(bu));
                            winYY += vsumLocal.GetValue(uW + static_cast<uint32_t>(bu));
                            winXY += vsumLocal.GetValue(2U * uW + static_cast<uint32_t>(bu));
                        }
                        for (int32_t x = uLo; x < uHi; ++x) {
                            const uint32_t ux = static_cast<uint32_t>(x);
                            const int32_t absX = colBase + x; // 列分块的 halo 列可能落到图像外
                            float resp = 0.0F;
                            if (absX >= 0 && absX < wAbs &&
                                (tiling_->hasMask == 0U ||
                                 maskGm_.GetValue(static_cast<uint64_t>(image) * tiling_->imageElements +
                                                  static_cast<uint32_t>(y) * tiling_->width +
                                                  static_cast<uint32_t>(absX)) != 0U)) {
                                resp = ResponseFromSum(winXX, winYY, winXY);
                            }
                            respLocal.SetValue(yOff + ux, resp);
                            if (resp > coreMax) {
                                coreMax = resp;
                            }
                            // 窗口右移一位：win[x+1] = win[x] - vsum[x-r] + vsum[x+r+1]（越界=0）
                            if (x < w - 1) {
                                const int32_t subIdx = x - radius;
                                const int32_t addIdx = x + radius + 1;
                                if (subIdx >= 0) {
                                    winXX -= vsumLocal.GetValue(static_cast<uint32_t>(subIdx));
                                    winYY -= vsumLocal.GetValue(uW + static_cast<uint32_t>(subIdx));
                                    winXY -= vsumLocal.GetValue(2U * uW + static_cast<uint32_t>(subIdx));
                                }
                                if (addIdx < w) {
                                    winXX += vsumLocal.GetValue(static_cast<uint32_t>(addIdx));
                                    winYY += vsumLocal.GetValue(uW + static_cast<uint32_t>(addIdx));
                                    winXY += vsumLocal.GetValue(2U * uW + static_cast<uint32_t>(addIdx));
                                }
                            }
                        }
                    }
                }
            } else {
                // 回退（超宽行未启用 UB 缓存）：逐像素 Response 窗口重算 Sobel
                for (int32_t y = yBase; y < yBase + coreRows; ++y) {
                    for (int32_t x = 0; x < w; ++x) {
                        const uint32_t localIdx = static_cast<uint32_t>(y - yBase) * static_cast<uint32_t>(ubStride_) +
                                                  static_cast<uint32_t>(x);
                        float resp = 0.0F;
                        if (tiling_->hasMask == 0U ||
                            maskGm_.GetValue(static_cast<uint64_t>(image) * tiling_->imageElements +
                                             static_cast<uint32_t>(y) * tiling_->width + static_cast<uint32_t>(x)) !=
                                0U) {
                            resp = Response(static_cast<uint32_t>(image), y, x);
                        }
                        respLocal.SetValue(localIdx, resp);
                        if (resp > coreMax) {
                            coreMax = resp;
                        }
                    }
                }
            }
            // DMA 批量写回 GM：每行一个 block，行间连续（srcStride/dstStride=0）。
            // UB 行距与原始行字节不等（行尾补齐）时改逐行写，避免多行块拷贝在 UB 侧落地行距的歧义。
            const uint32_t respBase = static_cast<uint32_t>(image) * tiling_->imageElements +
                                      static_cast<uint32_t>(yBase) * tiling_->width;
            if (colTiled) {
                // 列分块：只写回本块输出的列段 [uLo, uHi) → GM 列 [tx0, tx0+wT)。
                // UB 源偏移 uLo*sizeof(float) = 64B（colHalo=16 时）满足 DataCopyPad 的 32B 起点要求。
                for (int32_t r = 0; r < coreRows; ++r) {
                    DataCopyParams writeParams;
                    writeParams.blockCount = 1;
                    writeParams.blockLen = static_cast<uint16_t>(wT * static_cast<int32_t>(sizeof(float)));
                    writeParams.srcStride = 0;
                    writeParams.dstStride = 0;
                    DataCopyPad(
                        responseGm_[respBase + static_cast<uint32_t>(r) * tiling_->width + static_cast<uint32_t>(tx0)],
                        respLocal[static_cast<uint32_t>(r) * static_cast<uint32_t>(ubStride_) +
                                  static_cast<uint32_t>(uLo)],
                        writeParams);
                }
            } else if (respRowsPadded_) {
                for (int32_t r = 0; r < coreRows; ++r) {
                    DataCopyParams writeParams;
                    writeParams.blockCount = 1;
                    writeParams.blockLen = static_cast<uint16_t>(respRowBytes);
                    writeParams.srcStride = 0;
                    writeParams.dstStride = 0;
                    DataCopyPad(responseGm_[respBase + static_cast<uint32_t>(r) * tiling_->width],
                                respLocal[static_cast<uint32_t>(r) * static_cast<uint32_t>(ubStride_)], writeParams);
                }
            } else {
                DataCopyParams writeParams;
                writeParams.blockCount = static_cast<uint16_t>(coreRows);
                writeParams.blockLen = static_cast<uint16_t>(respRowBytes);
                writeParams.srcStride = 0;
                writeParams.dstStride = 0;
                DataCopyPad(responseGm_[respBase], respLocal, writeParams);
            }
            PipeBarrier<PIPE_MTE3>(); // 等待 DMA 写回完成（MTE3），保证后续跨核 SyncAll 后数据可见
        } // ===================== 列分块主循环结束 =====================
        ubCacheOn_ = false;
        row += coreRows;
    }
    // 最后一张图的局部 max 写回（DMA 单块 32B，保证跨核可见）
    if (maxInited) {
        WriteCoreMax(maxImage, coreIdx, coreMax);
    }
    // 无参 SyncAll = 硬件跨核 barrier（AscendC 接口 SyncAll<isAIVOnly=true>()）：不访问 GM/UB
    // workspace、不依赖 MIX task type，只要求每个已启动的核都执行到此处；前置条件是 DMA 写
    // 已通过 PipeBarrier<MTE3> 落盘，barrier 后跨核可见。
    SyncAll();
    // 阶段 2/3：候选收集全核并行（各核扫自己的行带），合并+排序+贪心在 0 号核。
    // 每个 image 两级 barrier：SyncAll(候选分区就绪) → 0 号核处理 → SyncAll(分区可复用)。
    for (uint32_t image = 0; image < tiling_->batch; ++image) {
        // 全局 max = 所有核局部 max 的最大值（只读 coreNum 个值，避免全图扫描）；
        // 跨核读必须走 MTE2（标量读 GM 可能命中本核陈旧缓存行）。
        // 逐图阈值：只归约「与该图行区间相交」的核（CoreRangeForImage 与阶段 1 同源公式）
        uint32_t statC0 = 0U;
        uint32_t statC1 = 0U;
        CoreRangeForImage(image, statC0, statC1);
        LocalTensor<float> stat = LoadCoreStats(image);
        float maxResponse = 0.0F;
        for (uint32_t c = 0; c < (statC1 - statC0 + 1U); ++c) {
            const float v = stat.GetValue(c * 8U);
            if (v > maxResponse) {
                maxResponse = v;
            }
        }
        // 候选扫描 staging 复用阶段 1 已结束、当前空闲的梯度缓冲（见 scanStageRows 推导）
        // 2a 的 staging 是「整行宽」（gradBuf_ 由 stage2aBytes=3*wAbs*4 保证），
        // 而 IsLocalMaxStaged 用 ubStride_ 作行距索引 ⇒ 列分块下 ubStride_ 是 strip
        // 步长，与 staging 的整行步长不符 ⇒ 邻域索引错位 ⇒ 候选成倍增多。
        const int32_t ubStrideSave = ubStride_;
        if (colTiled) {
            ubStride_ = ((wAbs * static_cast<int32_t>(sizeof(T)) + 31) & (~31)) / static_cast<int32_t>(sizeof(T));
        }
        CollectCandidates(image, maxResponse, gradBuf_.Get<float>(), scanStageRows, coreIdx,
                          static_cast<uint32_t>(coreNum), coreMax);
        ubStride_ = ubStrideSave; // 恢复（后续 image 的 phase 1 仍按 strip 步长工作）
        SyncAll();
        // 阶段 2b 拆成五段（A/B/C/D/E），把 0 号核的 O(n) 串行工作并行掉：
        //   A(0 号核) 向量归约 H1 → b* 与各核「小于组」偏移；B(全核) 写小于组 + 统计 H2；
        //   C(0 号核) 归约 H2 → b**（二级细化，把保留集压回 ≈K）与「等于组」偏移；
        //   D(全核) 写等于组；E(0 号核) 只对保留集（≈K）排序 + 贪心。
        if (coreIdx == 0U) {
            TopKBuildMeta(image, static_cast<uint32_t>(coreNum));
        }
        SyncAll();
        TopKFilterOwn(image, coreIdx, static_cast<uint32_t>(coreNum), 0U);
        SyncAll();
        if (coreIdx == 0U) {
            TopKRefineMeta(image, static_cast<uint32_t>(coreNum));
        }
        SyncAll();
        TopKFilterOwn(image, coreIdx, static_cast<uint32_t>(coreNum), 1U);
        SyncAll();
        if (coreIdx == 0U) {
            SelectFinal(image, static_cast<uint32_t>(coreNum));
        }
        SyncAll();
    }
}
} // namespace NsGoodFeaturesToTrack
#endif
