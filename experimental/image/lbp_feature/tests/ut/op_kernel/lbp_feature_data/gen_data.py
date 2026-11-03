#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ============================================================
# LBPFeature kernel UT 数据/golden 生成
# 基准：cv::elbp（OpenCV 原版语义），严格 float32 逐位模拟
# 与 ACLNN 版的差异：
#   ① 输出平铺到当前目录，文件名对齐 compare_data.py 的 glob 规则
#   ② 默认参数改为 kernel UT 典型用例（调用方会显式传参）
#   ③ 内层双循环向量化（逐元素 float32 舍入与标量版位级一致）
# ============================================================

import math
import time
import argparse
import numpy as np


# ============================================================
# 与 C++ OpenCV 中的 CV_PI 保持一致
# ============================================================

CV_PI = 3.14159265358979323846


# ============================================================
# float32 常量
# ============================================================

FLOAT_EPSILON = np.float32(np.finfo(np.float32).eps)

FLOAT_ONE = np.float32(1.0)


# ============================================================
# float32 运算辅助函数
#
# 显式 np.float32()，模拟 C++ float 每一步运算完成后的
# float32 舍入，而不是让 NumPy 偷偷提升到 float64。
# ============================================================


def f32(value):
    return np.float32(value)


def f32_mul(a, b):
    return np.float32(np.float32(a) * np.float32(b))


def f32_add(a, b):
    return np.float32(np.float32(a) + np.float32(b))


def f32_sub(a, b):
    return np.float32(np.float32(a) - np.float32(b))


def f32_abs(a):
    return np.float32(np.abs(np.float32(a)))


# ============================================================
# 严格按照 cv::elbp 计算一个 neighbor 的采样参数
# （与 ACLNN 版逐行一致）
# ============================================================


def get_neighbor_param(radius, neighbors, n):
    neighbors_float = np.float32(neighbors)

    angle = 2.0 * CV_PI * float(n) / float(neighbors_float)

    x = np.float32(float(radius) * math.cos(angle))

    y = np.float32(-float(radius) * math.sin(angle))

    fx = int(math.floor(float(x)))

    fy = int(math.floor(float(y)))

    cx = int(math.ceil(float(x)))

    cy = int(math.ceil(float(y)))

    tx = np.float32(x - np.float32(fx))

    ty = np.float32(y - np.float32(fy))

    one_minus_tx = np.float32(FLOAT_ONE - tx)

    one_minus_ty = np.float32(FLOAT_ONE - ty)

    w1 = f32_mul(one_minus_tx, one_minus_ty)

    w2 = f32_mul(tx, one_minus_ty)

    w3 = f32_mul(one_minus_tx, ty)

    w4 = f32_mul(tx, ty)

    return (
        x,
        y,
        fx,
        fy,
        cx,
        cy,
        tx,
        ty,
        w1,
        w2,
        w3,
        w4,
    )


def build_neighbor_params(radius, neighbors):
    params = []

    for n in range(neighbors):
        params.append(get_neighbor_param(radius, neighbors, n))

    return params


def pixel_to_float32(value):
    """
    uint8 经整数提升后参与 float 运算，最终转换成 float。
    uint8 0~255 能被 float32 精确表示。
    """

    return np.float32(value)


# ============================================================
# 对单张 2D 图像计算严格 OpenCV ELBP Golden
#
# 向量化说明：
#   neighbor 参数与像素位置无关，对每个 n，内层双循环
#   （for i / for j）可整体换成 numpy 切片。
#   逐元素 float32 运算的舍入与标量完全相同；
#   插值仍按 C++ 左结合顺序：
#       ((w1*p1 + w2*p2) + w3*p3) + w4*p4
#   因此结果与标量版位级一致。
# ============================================================


def compute_lbp(image2d, radius, neighbors):
    H, W = image2d.shape

    outH = H - 2 * radius
    outW = W - 2 * radius

    if outH <= 0 or outW <= 0:
        raise ValueError("H/W 必须大于 2 * radius")

    result = np.zeros((outH, outW), dtype=np.int32)

    neighbor_params = build_neighbor_params(radius, neighbors)

    # 中心像素（提前转 float32，一次完成）
    center = image2d[radius : H - radius, radius : W - radius].astype(np.float32)

    # 循环顺序保持 cv::elbp 的 for n 外层
    for n in range(neighbors):
        (
            x,
            y,
            fx,
            fy,
            cx,
            cy,
            tx,
            ty,
            w1,
            w2,
            w3,
            w4,
        ) = neighbor_params[n]

        # ============================================
        # 四个采样平面（切片向量化，对应
        # src.at<_Tp>(i+fy,j+fx) 等四个访问）
        # ============================================

        p1 = image2d[
            radius + fy : H - radius + fy, radius + fx : W - radius + fx
        ].astype(np.float32)

        p2 = image2d[
            radius + fy : H - radius + fy, radius + cx : W - radius + cx
        ].astype(np.float32)

        p3 = image2d[
            radius + cy : H - radius + cy, radius + fx : W - radius + fx
        ].astype(np.float32)

        p4 = image2d[
            radius + cy : H - radius + cy, radius + cx : W - radius + cx
        ].astype(np.float32)

        # ============================================
        # float t = w1*p1 + w2*p2 + w3*p3 + w4*p4;
        # 左结合：((w1*p1 + w2*p2) + w3*p3) + w4*p4
        # 每步 np.float32() 保证逐位与 C++ float 一致
        # ============================================

        t = f32_add(f32_mul(w1, p1), f32_mul(w2, p2))

        t = f32_add(t, f32_mul(w3, p3))

        t = f32_add(t, f32_mul(w4, p4))

        # ============================================
        # cv::elbp 比较条件：
        #   (t > center) || (|t - center| < FLT_EPSILON)
        # 等价于 kernel 的 t - center + epsilon > 0
        # ============================================

        diff = f32_sub(t, center)

        abs_diff = f32_abs(diff)

        cond = (t > center) | (abs_diff < FLOAT_EPSILON)

        # dst += condition << n
        result += cond.astype(np.int32) * np.int32(1 << n)

    return result


# ============================================================
# 主 Golden 生成函数
# ============================================================


def gen_golden_data_simple():
    parser = argparse.ArgumentParser(
        description=("生成严格仿照 cv::elbp 的 LBPFeature kernel UT 测试数据")
    )

    parser.add_argument(
        "--shape", type=str, default="6,70", help=("输入形状，例如 6,70")
    )

    parser.add_argument("--radius", type=int, default=1, help="LBP radius")

    parser.add_argument("--neighbors", type=int, default=8, help="LBP neighbors")

    parser.add_argument(
        "--dtype",
        type=str,
        default="uint8",
        choices=["uint8", "float32"],
        help="输入 dtype",
    )

    parser.add_argument(
        "--min_val", type=float, default=0.0, help=("float32 随机输入最小值")
    )

    parser.add_argument(
        "--max_val", type=float, default=255.0, help=("float32 随机输入最大值")
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help=("随机种子，-1表示随机（默认），>=0表示固定种子用于可复现测试"),
    )

    args = parser.parse_args()

    # ========================================================
    # 参数解析与校验（与算子守卫一致）
    # ========================================================

    radius = args.radius
    neighbors = args.neighbors

    shape = list(map(int, args.shape.split(",")))

    dtype = args.dtype

    if len(shape) < 2:
        raise ValueError("shape 至少需要二维")

    H = shape[-2]
    W = shape[-1]

    if radius < 0:
        raise ValueError("radius 不能小于 0")

    if neighbors < 1:
        raise ValueError("neighbors 必须 >= 1")

    if neighbors > 31:
        raise ValueError("当前 int32 LBP code 要求 neighbors <= 31")

    if H <= 2 * radius:
        raise ValueError(f"H={H} <= 2*radius={2 * radius}")

    if W <= 2 * radius:
        raise ValueError(f"W={W} <= 2*radius={2 * radius}")

    # ========================================================
    # 随机种子设置
    # ========================================================

    if args.seed < 0:
        seed = int(time.time() * 1000) % 1000000
    else:
        seed = args.seed

    np.random.seed(seed)

    print(f"随机种子: {seed}")

    # ========================================================
    # 输入生成
    # ========================================================

    if dtype == "uint8":
        input_x = np.random.uniform(0, 255, shape).astype(np.uint8)

        print(f"输入形状: {tuple(shape)}, dtype: uint8")

    else:
        input_x = np.random.uniform(args.min_val, args.max_val, shape).astype(
            np.float32
        )

        print(f"输入形状: {tuple(shape)}, dtype: float32")

        print(f"数据范围: [{args.min_val}, {args.max_val}]")

    print(f"参数: radius={radius}, neighbors={neighbors}")

    # ========================================================
    # 输出 shape
    # ========================================================

    outH = H - 2 * radius

    outW = W - 2 * radius

    out_shape = shape[:-2] + [outH, outW]

    golden = np.zeros(out_shape, dtype=np.int32)

    # ========================================================
    # 前导维度展平，逐张计算
    # ========================================================

    batch_count = 1

    for dim in shape[:-2]:
        batch_count *= dim

    input_reshaped = input_x.reshape(batch_count, H, W)

    golden_reshaped = golden.reshape(batch_count, outH, outW)

    for b in range(batch_count):
        golden_reshaped[b] = compute_lbp(input_reshaped[b], radius, neighbors)

    # ========================================================
    # 保存：平铺到当前目录
    # 文件名规则与 compare_data.py 的 glob 对齐：
    #   golden 必须含 "golden"，输出必须含 "output"，
    #   且都以 "int32_" 开头（compare 按 dtype 名匹配配对）
    # ========================================================

    input_x.tofile("input_lbp_feature.bin")

    golden.tofile("int32_golden_lbp_feature.bin")

    # ========================================================
    # 输出信息
    # ========================================================

    print(f"输出形状: {golden.shape}, dtype: int32")

    print(f"批次数: {batch_count} (自动展开)")

    print(f"golden 样例值 (batch0): {golden_reshaped[0, 0, :10]}...")

    print("INFO: generate input data success!")


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    gen_golden_data_simple()
