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
#include <map>
#include <vector>
#include "ge_api.h"
#include "ge_ir_build.h"
#include "graph.h"
#include "tensor.h"
#include "types.h"
#include "array_ops.h"
#include "../../op_graph/non_max_suppression_v7_proto.h"

using namespace ge;
int main()
{
    std::map<AscendString, AscendString> options = {{"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
    if (GEInitialize(options) != 0)
        return 1;
    int ret = 0;
    {
        Graph graph("non_max_suppression_v7_graph");
        TensorDesc bd(Shape({1, 4, 4}), FORMAT_ND, DT_FLOAT);
        bd.SetPlacement(kPlacementHost);
        TensorDesc sd(Shape({1, 2, 4}), FORMAT_ND, DT_FLOAT);
        sd.SetPlacement(kPlacementHost);
        op::Data boxes("boxes");
        boxes.set_attr_index(0);
        boxes.update_input_desc_x(bd);
        boxes.update_output_desc_y(bd);
        op::Data scores("scores");
        scores.set_attr_index(1);
        scores.update_input_desc_x(sd);
        scores.update_output_desc_y(sd);
        auto nms = op::NonMaxSuppressionV7("non_max_suppression_v7");
        nms.set_input_boxes(boxes).set_input_scores(scores);
        nms.set_attr_center_point_box(0).set_attr_max_boxes_size(4);
        nms.update_output_desc_selected_indices(TensorDesc(Shape({4, 3}), FORMAT_ND, DT_INT32));
        graph.SetInputs({boxes, scores}).SetOutputs({nms});
        Session session(std::map<AscendString, AscendString>{});
        ret = session.AddGraph(0, graph);
        if (ret == 0)
            std::cout << "NonMaxSuppressionV7 graph built successfully\n";
    }
    const int finalizeRet = GEFinalize();
    return ret == 0 && finalizeRet == 0 ? 0 : 1;
}
