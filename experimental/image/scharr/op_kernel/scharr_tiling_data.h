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
 * \file scharr_tiling_data.h
 * \brief tiling data struct for Scharr
 */

#ifndef __SCHARR_TILING_DATA_H__
#define __SCHARR_TILING_DATA_H__

#include <cstdint>

struct ScharrTilingData {
    int32_t totalLength;         // 输入/输出张量的总元素个数
    int32_t H;                   // 每个二维切片的高度，即倒数第二维
    int32_t W;                   // 每个二维切片的宽度，即最后一维
    int32_t batchCount;          // 前置维度乘积，即独立 HxW 切片数量
    int32_t usedCoreNum;         // 实际启用的 AIV 逻辑核数
    int32_t splitMode;           // 多核切分模式：0按输出行，1按输出像素，2按完整HxW切片
    int32_t baseRowsPerCore;     // 每核至少处理的输出行数（按 batchCount*H 扁平均分）
    int32_t extraRowCoreCount;   // 前多少核额外多处理1行
    int32_t basePixelsPerCore;   // 像素切分模式下每核至少处理的输出像素数
    int32_t extraPixelCoreCount; // 像素切分模式下前多少核额外多处理1个像素
    int32_t basePlanesPerCore;   // 切片模式下每核至少处理的完整HxW切片数
    int32_t extraPlaneCoreCount; // 切片模式下前多少核额外多处理1个完整切片
    int32_t blockW;              // fallback 路径单次处理的最大列数
    int32_t tileH;               // multirow 路径单块输出行数；0 表示走 fallback 路径
    int32_t compRows;            // 软件补偿链一次扁平处理的行数K（>=2；旧回退布局为0）
    int32_t compPitch;           // 补偿暂存区使用的行跨度
    int32_t bandW;               // 列分块 slab：每带输出列数；>0 表示启用列分块路径
    int32_t rawPitch;            // fallback 输入行槽跨度（输入类型元素）
    int32_t vectorPitch;         // scratch 中一个 float 向量槽的跨度（float 元素）
    int32_t inputSlabPitch;      // multirow 原始输入 slab 每行跨度（输入类型元素）
    int32_t floatSlabPitch;      // multirow 转换后 float slab 每行跨度（float 元素）
    int32_t outputSlabPitch;     // multirow 输出 slab 每行跨度（float 元素）
    uint32_t inputBufferBytes;   // 输入 Queue 中单个 UB Buffer 的字节数
    uint32_t outputBufferBytes;  // 输出 Queue 中单个 UB Buffer 的字节数
    uint32_t floatSlabBytes;     // 非 float32 输入转换后 float slab 的 UB 字节数
    uint32_t scratchBytes;       // 整个 scratch arena 的 UB 字节数
    uint32_t slotsOffset;        // scratch 内三个通用向量槽的起始字节偏移
    uint32_t work0Offset;        // scratch 内第一个临时计算向量的字节偏移
    uint32_t work1Offset;        // scratch 内第二个临时计算向量的字节偏移
    uint32_t comp0Offset;        // 软件补偿计算高位中间量的字节偏移
    uint32_t comp1Offset;        // 软件补偿计算低位中间量的字节偏移
    uint32_t comp2Offset;        // 软件补偿计算主结果中间量的字节偏移
    uint32_t fallbackOffset;     // 非有限值等特殊数据回退结果向量的字节偏移
    uint32_t maskOffset;         // 有限值判定掩码在 scratch 中的字节偏移
    uint32_t castHalfOffset;     // uint8 转 float32 时所用 half 中转区的字节偏移
    uint32_t offset0Offset;      // Gather 偏移表0的字节偏移
    uint32_t offset1Offset;      // Gather 偏移表1的字节偏移
    uint32_t coefficientOffset;  // 常驻系数向量字节偏移
    uint32_t tailBatchOffset;    // Y奇数末列批量快照及索引的UB字节偏移，0表示关闭
    uint32_t bandTableOffset;    // 列分块 Gather 表 B/C 的字节偏移
    int32_t direction;           // 0 = X, 1 = Y
    int32_t border;              // 0 = REFLECT_101, 1 = REPLICATE, 2 = CONSTANT
};

#endif // __SCHARR_TILING_DATA_H__
