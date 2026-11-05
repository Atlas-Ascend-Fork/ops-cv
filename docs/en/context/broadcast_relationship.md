# Broadcast Relationship

## Broadcast Concept

Broadcast describes how operators handle tensors (or arrays) of different shapes during computation. In most cases, tensors (or arrays) of different shapes are allowed to automatically expand their shapes during element-wise operations so that their dimensions become compatible with each other. Typically, the smaller tensor (or array) is "broadcast" to match the larger tensor (or array).

Currently, many CANN operator API parameters support broadcast in their shapes, which can improve computational efficiency and reduce memory usage (especially in large-scale data scenarios). For a more detailed introduction to broadcast techniques, refer to the [NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html) official website.

## Broadcast Rules

When performing broadcast computation, the following rules generally apply:

- Rule 1: If the number of dimensions differs between arrays, all arrays are aligned to the array with the longest shape. Dimensions that are insufficient are padded with 1 on the **left** until the number of dimensions is equal.

  > NOTE
  > - Example 1: The number of dimensions refers to the dimensionality of the shape of a tensor (or array). For instance, if x.shape = (1, 1, 2, 4), the number of dimensions is 4.
  > - Example 2: For computing a + b, where a.shape = (2, 2, 3) and b.shape = (2, 3), array b will be broadcast to b.shape = (1, 2, 3).

- Rule 2: If the number of dimensions is equal between arrays and a certain dimension of one array is 1, that array with the dimension of 1 will be stretched to match the corresponding dimension shape of the other array.

  > **NOTE**
  > In this scenario, you only need to ensure that broadcast can be performed along a certain dimension. For example, when computing a+b where a.shape=\(1, 3\) and b.shape=\(3, 1\), the two arrays will be broadcast to a.shape=\(3, 3\) and b.shape=\(3, 3\).

- Rule 3: After left-padding with 1 according to Rule 1, broadcasting fails if any pair of corresponding dimensions has unequal sizes and neither size is 1.

Based on the above rules, the broadcast process generally first expands dimensions according to **Rule 1**, and then stretches shapes according to **Rule 2**. A specific example is as follows:

```text
Assume a.shape=(2,2,3). Values:
[[[1 2 3],[4 5 6]],
 [[1 2 3],[4 5 6]]]
Assume b.shape=(2,3). Values:
[[1 2 3],
 [-1 -2 -3]]
Expand dimensions according to rule 1. b.shape=(1,2,3). Values:
[[[1 2 3],
  [-1 -2 -3]]]
Stretch shapes according to rule 2. b.shape=(2,2,3). Values:
[[[1 2 3],[-1 -2 -3]],
 [[1 2 3],[-1 -2 -3]]]
Calculate a+b and get:
 [[[2 4 6],[3 3 3]],
  [[2 4 6],[3 3 3]]]
```

## Restrictions

When the data types or the derived data types of two inputs a and b that satisfy the broadcast relationship fall within COMPLEX64, COMPLEX128, DOUBLE, INT16, UINT16, or UINT64, the following condition must be met in addition to the broadcast rules described above. Otherwise, the broadcast fails and the operator execution reports an error.
Condition: After merging consecutive axes that require broadcasting and consecutive axes that do not require broadcasting, the total number of resulting dimensions must be less than 6.
Example:

- When a.shape=\(5, 1, 5, 1, 5, 1\) and b.shape=\(5, 5, 5, 5, 5, 5\), there are no axes to merge, and the final number of dimensions is 6. The broadcast reports an error.
- When a.shape=\(5, 1, 5, 5, 1, 1\) and b.shape=\(5, 5, 5, 5, 5, 5\), the 2nd and 3rd dimensions do not require broadcasting, while the 4th and 5th dimensions both require broadcasting. After merging consecutive axes respectively, the total number of dimensions is 4. The broadcast succeeds.
