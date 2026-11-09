/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BOUNDING_BOX_DECODE_TILING_DATA_H_
#define BOUNDING_BOX_DECODE_TILING_DATA_H_

#include <cstdint>

// bounding_box_decode TilingData — Elementwise 按 box 数切分（tiling.md §四/§七）+ 属性标量
// 非模板：固定 rank=2，无 broadcast，无 Group；IS_EMPTY 属 TilingKey（§6）不进本结构
constexpr int64_t kElemsPerBox = 4; // C=4，每 box 4 元素 (x1,y1,x2,y2) / (dx',dy',dw',dh')

struct BoundingBoxDecodeTilingData {
    // —— 多核切分（按 box 数，§9 多核切分；tiling.md §一/§七）—— 单位均为 box 数（除 coreNum 为核数）
    int64_t dim0;        // box 总数 N（= anchor_box.shape[0]；非元素数，元素数 = dim0 × kElemsPerBox）
    int32_t coreNum;     // 实际参与核数 = min(CeilDiv(dim0×minDtypeBits, MIN_TILING_BITS), availableCoreNum)
    int64_t blockFormer; // 每核基础 box 数（512 box 对齐，ELEM_ALIGN_FACTOR=512）
    int64_t blockNum;    // 虚拟 block 数 = CeilDiv(dim0, blockFormer)
    // —— UB 切分（按 box 数，§9 UB 切分；tiling.md §二/§七）—— 单位均为 box 数
    int64_t ubFormer; // 每 UB 块基础 box 数（256B 对齐；alignFactor = 256 / (kElemsPerBox × sizeof(T))，fp16=32 box /
                      // fp32=16 box）
    int64_t ubLoopOfFormerBlock; // 首 block 的 UB 循环次数 = CeilDiv(blockFormer, ubFormer)
    int64_t ubTailOfFormerBlock; // 首 block 尾部 box 数 = blockFormer - (ubLoopOfFormerBlock-1)×ubFormer
    int64_t ubLoopOfTailBlock;   // 尾 block 的 UB 循环次数 = CeilDiv(blockTail, ubFormer)
    int64_t ubTailOfTailBlock;   // 尾 block 尾部 box 数 = blockTail - (ubLoopOfTailBlock-1)×ubFormer
    // —— 属性标量（GE 图属性传入——本算子 aclnn_exclude，kernel Compute 消费）——
    float means[4]; // deltas 反标准化均值 m0..m3（spec.yaml attributes.means，默认 0）
    float stds[4];  // deltas 反标准化标准差 s0..s3（各元素 ≠ 0，Host 校验保证；默认 1）
    int64_t maxShapeH; // max_shape[0] = H，原样透传；kernel 内实际裁剪上界为 maxShapeH-1（最后有效像素）
    int64_t maxShapeW; // max_shape[1] = W，原样透传；kernel 内实际裁剪上界为 maxShapeW-1（最后有效像素）
    // wh_ratio_clip 对齐 A2 基线（canndev tbe clamp_denorm_detal）：Host 预计算
    // whMaxRatio = |ln(wh_ratio_clip)|，kernel 在反标准化后将 dw/dh 裁剪到 ±whMaxRatio 再进入 exp。
    // （旧版注释"wh_ratio_clip 不参与核心公式"与 A2 基线及 proto.h 声明不符，已按基线补齐。）
    float whMaxRatio;
    // 注：空 tensor 标记不进 TilingData 也不进 TilingKey——kernel 以 td.dim0==0 运行期
    // 短路（Task 41：BOOL TPL 参数曾致框架 e2e tiling buffer 分配失败已移除；
    // TilingKey 仅按 dtype 分档 FP32=0/FP16=1）
    // 注：empty 路径（IS_EMPTY=true）kernel 短路，上述计算字段填 0 不被消费
};

#endif // BOUNDING_BOX_DECODE_TILING_DATA_H_
