/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_test_utils.h"
#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include <limits>

template <typename T>
class AicpuCompareReal : public testing::Test {};
using RealTypes = testing::Types<float, double, Eigen::half>;
TYPED_TEST_SUITE(AicpuCompareReal, RealTypes);

TYPED_TEST(AicpuCompareReal, RejectsNanAndMismatchedInfinities)
{
    using T = TypeParam;
    T one = T(1);
    T nan = std::numeric_limits<T>::quiet_NaN();
    T inf = std::numeric_limits<T>::infinity();
    T negativeInf = -inf;
    EXPECT_FALSE(CompareResult(&nan, &one, 1));
    EXPECT_FALSE(CompareResult(&one, &nan, 1));
    EXPECT_FALSE(CompareResult(&nan, &nan, 1));
    EXPECT_FALSE(CompareResult(&inf, &one, 1));
    EXPECT_FALSE(CompareResult(&one, &inf, 1));
    EXPECT_FALSE(CompareResult(&negativeInf, &inf, 1));
}

TYPED_TEST(AicpuCompareReal, MatchesSignedInfinitiesAndFiniteControls)
{
    using T = TypeParam;
    T inf = std::numeric_limits<T>::infinity();
    T negativeInf = -inf;
    T one = T(1);
    T two = T(2);
    T zero = T(0);
    T negativeZero = T(-0.0);
    EXPECT_TRUE(CompareResult(&inf, &inf, 1));
    EXPECT_TRUE(CompareResult(&negativeInf, &negativeInf, 1));
    EXPECT_TRUE(CompareResult(&one, &one, 1));
    EXPECT_TRUE(CompareResult(&zero, &negativeZero, 1));
    EXPECT_FALSE(CompareResult(&two, &one, 1));
}

template <typename T>
void CheckComplexNonfinite()
{
    using C = std::complex<T>;
    const T inf = std::numeric_limits<T>::infinity();
    const T nan = std::numeric_limits<T>::quiet_NaN();
    C finite(1, 1);
    for (C value : {C(nan, 1), C(1, nan), C(inf, 1), C(1, inf)}) {
        EXPECT_FALSE(CompareResult(&value, &finite, 1));
        EXPECT_FALSE(CompareResult(&finite, &value, 1));
    }
    C same(inf, 1);
    C near(inf, std::nextafter(T(1), T(2)));
    C wrong(inf, 2);
    C opposite(-inf, 1);
    C nanValue(nan, 1);
    EXPECT_TRUE(CompareResult(&same, &same, 1));
    ASSERT_NE(same.imag(), near.imag());
    EXPECT_TRUE(CompareResult(&same, &near, 1));
    EXPECT_FALSE(CompareResult(&same, &wrong, 1));
    EXPECT_FALSE(CompareResult(&same, &opposite, 1));
    EXPECT_FALSE(CompareResult(&nanValue, &nanValue, 1));
}

TEST(AicpuCompareResult, ComplexNonfiniteComponents)
{
    CheckComplexNonfinite<float>();
    CheckComplexNonfinite<double>();
}

TEST(AicpuCompareResult, FiniteToleranceControls)
{
    float f = 1, fnear = 1 + 5e-7F;
    double d = 1, dnear = 1 + 5e-11;
    Eigen::half h(1), hnear(1.0009765625F);
    EXPECT_TRUE(CompareResult(&f, &fnear, 1));
    EXPECT_TRUE(CompareResult(&d, &dnear, 1));
    EXPECT_TRUE(CompareResult(&h, &hnear, 1));
}

TEST(AicpuCompareResult, AllCloseNonfinitePolicy)
{
    float one = 1, near = 1.00005F, wrong = 2;
    float inf = std::numeric_limits<float>::infinity();
    float negativeInf = -inf;
    float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(CompareResultAllClose(&nan, &one, 1));
    EXPECT_FALSE(CompareResultAllClose(&one, &nan, 1));
    EXPECT_FALSE(CompareResultAllClose(&nan, &nan, 1));
    EXPECT_FALSE(CompareResultAllClose(&inf, &one, 1));
    EXPECT_FALSE(CompareResultAllClose(&one, &inf, 1));
    EXPECT_FALSE(CompareResultAllClose(&negativeInf, &inf, 1));
    EXPECT_TRUE(CompareResultAllClose(&inf, &inf, 1));
    EXPECT_TRUE(CompareResultAllClose(&negativeInf, &negativeInf, 1));
    EXPECT_TRUE(CompareResultAllClose(&one, &near, 1));
    EXPECT_FALSE(CompareResultAllClose(&one, &wrong, 1));
}
