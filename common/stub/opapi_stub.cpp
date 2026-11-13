/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/slice.h"

namespace l0op {
const aclTensor* Contiguous(const aclTensor* x, aclOpExecutor* executor)
{
    (void)x;
    (void)executor;
    return x;
}

const aclTensor* Transpose(const aclTensor* x, const aclIntArray* perm, aclOpExecutor* executor)
{
    (void)x;
    (void)perm;
    (void)executor;
    return x;
}

const aclTensor* ViewCopy(const aclTensor* x, const aclTensor* y, aclOpExecutor* executor)
{
    (void)x;
    (void)y;
    (void)executor;
    return y;
}

const aclTensor* UnsqueezeNd(const aclTensor* x, const aclIntArray* dim, aclOpExecutor* executor)
{
    (void)x;
    (void)dim;
    (void)executor;
    return x;
}

const aclTensor* UnsqueezeNd(const aclTensor* x, int64_t dim, aclOpExecutor* executor)
{
    (void)x;
    (void)dim;
    (void)executor;
    return x;
}

const aclTensor* SqueezeNd(const aclTensor* x, const aclIntArray* dim, aclOpExecutor* executor)
{
    (void)x;
    (void)dim;
    (void)executor;
    return x;
}

const aclTensor* SqueezeNd(const aclTensor* x, int64_t dim, aclOpExecutor* executor)
{
    (void)x;
    (void)dim;
    (void)executor;
    return x;
}

const aclTensor* ReFormat(const aclTensor* x, const op::Format& format, aclOpExecutor* executor = nullptr)
{
    (void)x;
    (void)format;
    (void)executor;
    return x;
}

const aclTensor* TransDataSpecial(const aclTensor* x, op::Format dstPrimaryFormat, int64_t groups,
                                  aclOpExecutor* executor)
{
    (void)x;
    (void)dstPrimaryFormat;
    (void)groups;
    (void)executor;
    return x;
}

const aclTensor* Cast(const aclTensor* self, op::DataType dstDtype, aclOpExecutor* executor)
{
    (void)self;
    (void)dstDtype;
    (void)executor;
    return self;
}

const aclTensor* CastOnlyForConvBackward(const aclTensor* self, op::DataType dstDtype, aclOpExecutor* executor)
{
    (void)self;
    (void)dstDtype;
    (void)executor;
    return self;
}

const aclTensor* TransData(const aclTensor* x, op::Format dstPrimaryFormat, int64_t groups, aclOpExecutor* executor)
{
    (void)x;
    (void)dstPrimaryFormat;
    (void)groups;
    (void)executor;
    return x;
}

const aclTensor* Reshape(const aclTensor* x, const op::Shape& shape, aclOpExecutor* executor)
{
    (void)x;
    (void)shape;
    (void)executor;
    return x;
}

const aclTensor* Reshape(const aclTensor* x, const aclIntArray* shape, aclOpExecutor* executor)
{
    (void)x;
    (void)shape;
    (void)executor;
    return x;
}

const aclTensor* Fill(const aclTensor* dims, const aclTensor* value, const aclIntArray* outShape,
                      aclOpExecutor* executor)
{
    (void)dims;
    (void)value;
    (void)outShape;
    (void)executor;
    return value;
}

const aclTensor* Slice(const aclTensor* x, const aclTensor* y, const aclTensor* offset, const aclTensor* size,
                       aclOpExecutor* executor)
{
    (void)x;
    (void)y;
    (void)offset;
    (void)size;
    (void)executor;
    return x;
}

const aclTensor* Slice(const aclTensor* x, const aclIntArray* offsets, const aclIntArray* size, aclOpExecutor* executor)
{
    (void)x;
    (void)offsets;
    (void)size;
    (void)executor;
    return x;
}

aclTensor* ConcatD(const aclTensorList* inputs, int64_t dim, op::DataType outDtype, aclOpExecutor* executor)
{
    (void)inputs;
    (void)dim;
    (void)outDtype;
    (void)executor;
    return nullptr;
}

aclTensor* ConcatD(const aclTensorList* inputs, int64_t dim, aclOpExecutor* executor)
{
    (void)inputs;
    (void)dim;
    (void)executor;
    return nullptr;
}

} // namespace l0op
