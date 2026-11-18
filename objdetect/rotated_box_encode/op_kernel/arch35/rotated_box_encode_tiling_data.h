/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ROTATED_BOX_ENCODE_TILING_DATA_H_
#define ROTATED_BOX_ENCODE_TILING_DATA_H_
#include <cstdint>

constexpr int64_t BOX_CHANNELS = 5; // box 5 维通道固定（x0, y0, x1, y1, θ_deg），spec.yaml shape[1]==5

struct RotatedBoxEncodeTilingData {
    int64_t dim0;        // box 总数 = B × N（展平为 1D 线性处理的语义单元数，非元素数）
    int64_t N;           // box 数 per batch = shape[2]，用于 5 通道 GM stride：element(b,c,n) = b*5N + c*N + n
    int32_t coreNum;     // 实际使用的核数；空 tensor（dim0==0）时置 0
    int64_t blockFormer; // 每核基础 box 数（对齐到 ELEM_ALIGN_FACTOR=512 box）
    int64_t blockNum;    // 虚拟 block 数量 = ceil(dim0 / blockFormer)
    int64_t ubFormer;    // UB tile 大小（box 数，对齐到 ALIGN_256_BYTES=256 / (BOX_CHANNELS × sizeof(T))）
    int64_t ubLoopOfFormerBlock; // 首 block 的 UB 循环次数 = ceil(blockFormer / ubFormer)
    int64_t ubTailOfFormerBlock; // 首 block 的尾部 box 数 = blockFormer - (ubLoopOfFormerBlock - 1) × ubFormer
    int64_t ubLoopOfTailBlock;   // 尾 block 的 UB 循环次数 = ceil(blockTail / ubFormer)
    int64_t ubTailOfTailBlock;   // 尾 block 的尾部 box 数 = blockTail - (ubLoopOfTailBlock - 1) × ubFormer
    float weight[BOX_CHANNELS];  // 编码权重 [wx, wy, ww, wh, wa]（attr weight 透传，缺省 [1,1,1,1,1]）
    int64_t slotElems; // 聚合布局：每 batch 每通道的 UB slot 元素数 A = ceilAlign(N*sizeof(T), 32B)/sizeof(T)；
                       // 0 = 非聚合（per-batch 路径）。tile 内 batch 数 K = ubFormer/N，通道 stride = K*A。
                       // 新增字段只追加在结构体末尾（保持历史字段偏移不变，兼容缓存 tiling blob）
};
#endif // ROTATED_BOX_ENCODE_TILING_DATA_H_
