/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <limits>
#include <vector>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"
#include "../../../op_kernel/clahe_tiling_data.h"

TEST(ClaheInferShape, PreservesStaticAndDynamicShapes)
{
    for (const auto& dims :
         std::vector<std::vector<int64_t>>{{64, 65}, {256, 3, 5}, {2, 1, 17, 19}, {-1, -1}, {-1, 1, -1, -1}, {-2}}) {
        gert::Shape shape;
        for (auto dim : dims) {
            shape.AppendDim(dim);
        }
        gert::StorageShape storage;
        storage.MutableOriginShape() = shape;
        storage.MutableStorageShape() = shape;
        gert::InfershapeContextPara para("Clahe", {{storage, ge::DT_UINT16, ge::FORMAT_ND}},
                                         {{{{}, {}}, ge::DT_UINT16, ge::FORMAT_ND}},
                                         std::vector<gert::InfershapeContextPara::OpAttr>{});
        ExecuteTestCase(para, ge::GRAPH_SUCCESS, std::vector<std::vector<int64_t>>{dims});
    }
}
