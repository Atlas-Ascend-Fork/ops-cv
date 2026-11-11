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
 * \file resize_bilinear_v2.cpp
 * \brief resize_bilinear_v2
 */
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5162)
#include "./arch51/resize_bilinear_v2_simd_nhwc.h"
#else
#include "./arch35/resize_bilinear_v2_base.h"
#include "./arch35/resize_bilinear_v2_all_copy.h"
#include "./arch35/resize_bilinear_v2_point_copy.h"
#include "./arch35/resize_bilinear_v2_broadcast_nchw.h"
#include "./arch35/resize_bilinear_v2_broadcast_nhwc.h"
#include "./arch35/resize_bilinear_v2_c_parallel.h"
#include "./arch35/resize_bilinear_v2_nc.h"
#include "./arch35/resize_bilinear_v2_simt_nchw.h"
#include "./arch35/resize_bilinear_v2_simt_nhwc.h"
#include "./arch35/resize_bilinear_v2_simt_nhwc.h"
#include "./arch35/resize_bilinear_v2_simt_hw.h"
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5162)
#define TILING_KEY_SIMD_NHWC 50000
#else
#define TILING_KEY_C_PARALLEL 10000
#define TILING_KEY_HW_CACHE 20000
#define TILING_KEY_SIMT_NHWC 30000
#define TILING_KEY_SIMT_NCHW 30001
#define TILING_KEY_SIMT_NHWC_IDX64 30002
#define TILING_KEY_SIMT_NCHW_IDX64 30003
#define TILING_KEY_SIMT_HW 30004
#define TILING_KEY_SIMT_HW_IDX64 30005
#define TILING_KEY_ALL_COPY 40000
#define TILING_KEY_POINT_COPY 40001
#define TILING_KEY_NCHW_BROADCAST 40002
#define TILING_KEY_NHWC_BROADCAST 40003
#endif

using namespace AscendC;

extern "C" __global__ __aicore__ void resize_bilinear_v2(GM_ADDR x, GM_ADDR size, GM_ADDR y, GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tilingData, tiling);

    TPipe pipe;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5162)
    if (TILING_KEY_IS(TILING_KEY_SIMD_NHWC)) {
        ResizeBilinearV2::ResizeBilinearV2SimdNhwc<DTYPE_X, DTYPE_Y> op;
        op.Init(x, size, y, &pipe, &tilingData);
        op.Process();
    }
#else
    if (TILING_KEY_IS(TILING_KEY_ALL_COPY)) {
        ResizeBilinearV2::ResizeBilinearV2AllCopy<DTYPE_X> op;
        op.Init(x, size, y, &pipe, &tilingData);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_POINT_COPY)) {
        ResizeBilinearV2::ResizeBilinearV2PointCopy<DTYPE_X> op;
        op.Init(x, size, y, &pipe, &tilingData);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_NCHW_BROADCAST)) {
        ResizeBilinearV2::ResizeBilinearV2BroadcastNCHW<DTYPE_X> op;
        op.Init(x, size, y, &pipe, &tilingData);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_NHWC_BROADCAST)) {
        ResizeBilinearV2::ResizeBilinearV2BroadcastNHWC<DTYPE_X> op;
        op.Init(x, size, y, &pipe, &tilingData);
        op.Process();
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_C_PARALLEL)) {
        if (tilingData.cFactor < tilingData.lenC) {
            ResizeBilinearV2::ResizeBilinearV2Nc<DTYPE_X, DTYPE_Y> op;
            op.Init(x, size, y, &pipe, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2::ResizeBilinearV2CParallel<DTYPE_X, DTYPE_Y> op;
            op.Init(x, size, y, &pipe, &tilingData);
            op.Process();
        }
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_NCHW)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2::ResizeBilinearV2SimtNCHW<DTYPE_X, DTYPE_Y, true, 0, uint32_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
            return;
        }
        if (tilingData.lenSrcH == tilingData.lenDesH && tilingData.lenSrcW == tilingData.lenDesW) {
            ResizeBilinearV2::ResizeBilinearV2SimtNCHW<DTYPE_X, DTYPE_Y, false, 1, uint32_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        } else if (tilingData.lenSrcH == 1 && tilingData.lenSrcW == 1) {
            ResizeBilinearV2::ResizeBilinearV2SimtNCHW<DTYPE_X, DTYPE_Y, false, 2, uint32_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        } else if (tilingData.lenDesH == 1 && tilingData.lenDesW == 1) {
            ResizeBilinearV2::ResizeBilinearV2SimtNCHW<DTYPE_X, DTYPE_Y, false, 3, uint32_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2::ResizeBilinearV2SimtNCHW<DTYPE_X, DTYPE_Y, false, 0, uint32_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        }

        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_NHWC)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2::ResizeBilinearV2SimtNHWC<DTYPE_X, DTYPE_Y, true, 0, uint32_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
            return;
        }
        if (tilingData.lenSrcH == tilingData.lenDesH && tilingData.lenSrcW == tilingData.lenDesW) {
            ResizeBilinearV2::ResizeBilinearV2SimtNHWC<DTYPE_X, DTYPE_Y, false, 1, uint32_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        } else if (tilingData.lenSrcH == 1 && tilingData.lenSrcW == 1) {
            ResizeBilinearV2::ResizeBilinearV2SimtNHWC<DTYPE_X, DTYPE_Y, false, 2, uint32_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        } else if (tilingData.lenDesH == 1 && tilingData.lenDesW == 1) {
            ResizeBilinearV2::ResizeBilinearV2SimtNHWC<DTYPE_X, DTYPE_Y, false, 3, uint32_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2::ResizeBilinearV2SimtNHWC<DTYPE_X, DTYPE_Y, false, 0, uint32_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_NCHW_IDX64)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2::ResizeBilinearV2SimtNCHW<DTYPE_X, DTYPE_Y, true, 0, uint64_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
            return;
        }
        if (tilingData.lenSrcH == tilingData.lenDesH && tilingData.lenSrcW == tilingData.lenDesW) {
            ResizeBilinearV2::ResizeBilinearV2SimtNCHW<DTYPE_X, DTYPE_Y, false, 1, uint64_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        } else if (tilingData.lenSrcH == 1 && tilingData.lenSrcW == 1) {
            ResizeBilinearV2::ResizeBilinearV2SimtNCHW<DTYPE_X, DTYPE_Y, false, 2, uint64_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        } else if (tilingData.lenDesH == 1 && tilingData.lenDesW == 1) {
            ResizeBilinearV2::ResizeBilinearV2SimtNCHW<DTYPE_X, DTYPE_Y, false, 3, uint64_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2::ResizeBilinearV2SimtNCHW<DTYPE_X, DTYPE_Y, false, 0, uint64_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_NHWC_IDX64)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2::ResizeBilinearV2SimtNHWC<DTYPE_X, DTYPE_Y, true, 0, uint64_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
            return;
        }
        if (tilingData.lenSrcH == tilingData.lenDesH && tilingData.lenSrcW == tilingData.lenDesW) {
            ResizeBilinearV2::ResizeBilinearV2SimtNHWC<DTYPE_X, DTYPE_Y, false, 1, uint64_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        } else if (tilingData.lenSrcH == 1 && tilingData.lenSrcW == 1) {
            ResizeBilinearV2::ResizeBilinearV2SimtNHWC<DTYPE_X, DTYPE_Y, false, 2, uint64_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        } else if (tilingData.lenDesH == 1 && tilingData.lenDesW == 1) {
            ResizeBilinearV2::ResizeBilinearV2SimtNHWC<DTYPE_X, DTYPE_Y, false, 3, uint64_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2::ResizeBilinearV2SimtNHWC<DTYPE_X, DTYPE_Y, false, 0, uint64_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        }
        return;
    }

    if (TILING_KEY_IS(TILING_KEY_SIMT_HW)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2::ResizeBilinearV2SimtHW<DTYPE_X, DTYPE_Y, true, 0, uint32_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2::ResizeBilinearV2SimtHW<DTYPE_X, DTYPE_Y, false, 0, uint32_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        }
        return;
    }
    if (TILING_KEY_IS(TILING_KEY_SIMT_HW_IDX64)) {
        if (tilingData.halfPixelCenters > 0) {
            ResizeBilinearV2::ResizeBilinearV2SimtHW<DTYPE_X, DTYPE_Y, true, 0, uint64_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        } else {
            ResizeBilinearV2::ResizeBilinearV2SimtHW<DTYPE_X, DTYPE_Y, false, 0, uint64_t> op;
            op.Init(x, size, y, &tilingData);
            op.Process();
        }
        return;
    }
#endif
}
