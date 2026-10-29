# Rasterizer

English | [简体中文](./README.md)

## Product Support

|Product             |  Supported  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     ×    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function Description

- Operator function: performs rasterization. Based on the given points and faces in 3D space, the operator obtains the minimum depth of each pixel on the screen and the index of the corresponding face, and computes the perspective-corrected interpolation of the barycentric coordinates of that face.

- Calculation formula:

  $findices$ records the face index corresponding to the minimum depth of each pixel, and $barycentric$ records the perspective-corrected interpolation of the barycentric coordinates of each vertex with respect to the face recorded in $findices$.
  The Z-Buffer used in the calculation records the minimum depth $z_{\min}(x, y)$ of each pixel $(x, y)$ and the triangle face index $\text{face\_idx}(x, y)$ corresponding to that depth.

  The calculation procedure is as follows. For each triangle face $f$ in space:

  1. Convert the three vertex coordinates $(v_0, v_1, v_2)$ of $f$ into screen coordinates $(v_{s0}, v_{s1}, v_{s2})$
  2. Calculate the rectangular range bounding $f$ based on $(v_{s0}, v_{s1}, v_{s2})$
  3. For each pixel $v_i = (x_i, y_i)$ in the rectangle, perform the following operations:

     a. Calculate the pixel center coordinate $v_c$
     b. Calculate the barycentric coordinates $(\alpha, \beta, \gamma)$ of $v_c$ with respect to triangle $f$
     c. Determine whether $v_c$ is inside the triangle based on $(\alpha, \beta, \gamma)$. If $v_c$ is not inside the triangle, process the next pixel in the rectangle; otherwise, perform the following steps
     d. Use $(\alpha, \beta, \gamma)$ and $(v_{s0}, v_{s1}, v_{s2})$ to obtain the depth value of the current pixel
     e. If the depth prior is enabled, calculate the depth threshold depth_thres using the depth prior map; otherwise, directly perform the next step "Z-Buffer update"

        - Calculate the depth threshold depth_thres using the depth prior map
        - If depth < depth_thres, process the next pixel in the rectangle; otherwise, perform the following steps

     f. Z-Buffer update:

        - If $depth < z_{\min}(x_i, y_i)$:

          $$
          \quad z_{\min}(x_i, y_i) \gets \text{depth} \\
          \quad \text{face\_idx}(x_i, y_i) \gets f
          $$

        - If $depth = z_{\min}(x_i, y_i)$:

          $$
          \quad \text{face\_idx}(x_i, y_i) \gets \min(\text{face\_idx}(x_i, y_i),\ f)
          $$

  After all triangle faces in space are processed by the preceding steps, for each pixel $v_i = (x_i, y_i)$ on the screen of size $height * width$:

  1. Take the face index $f_{idx}$ corresponding to $v_i$ from the Z-Buffer, $findices (x_i, y_i) \gets f_{idx}$
  2. Convert the three vertex coordinates $(v_0, v_1, v_2)$ of $f$ into screen coordinates $(v_{s0}, v_{s1}, v_{s2})$
  3. Calculate the center point coordinate $v_c$ of $v_i$
  4. Calculate the barycentric coordinates $(\alpha, \beta, \gamma)$ of $v_c$ with respect to triangle $f$
  5. Use $(\alpha, \beta, \gamma)$ to calculate the perspective-corrected interpolation $(\tilde{\alpha}, \tilde{\beta}, \tilde{\gamma})$
  6. $barycentric(x_i, y_i) \gets (\tilde{\alpha}, \tilde{\beta}, \tilde{\gamma})$

  The specific calculation methods involved are as follows:

  - Convert the vertex $v = (x, y, z, w)$ into screen coordinates $v_s = (x_s, y_s, z_s)$

    $$
    x_s = (x / w * 0.5 + 0.5) * (width - 1) + 0.5\\
    y_s = (0.5 + 0.5 * y / w) * (height - 1) + 0.5\\
    z_s = z / w * 0.49999 + 0.5
    $$

  - Barycentric coordinates $(\alpha, \beta, \gamma)$ of the point $v$ with respect to the triangle $(v_0, v_1, v_2)$

    1. Calculate the signed areas $area$, $beta\_tri$, and $gamma\_tri$ of the triangles $(v_0, v_1, v_2)$, $(v_0, v, v_2)$, and $(v_0, v_1, v)$ respectively
    2. If $area$ is 0, then $\alpha = \beta = \gamma = -1$; otherwise

      $$
      \beta = beta\_tri / area\\
      \gamma = gamma\_tri / area\\
      \alpha = 1 - \beta - \gamma
      $$

  - Signed area of the triangle formed by the vertices $v_0 = (x_0, y_0, z_0)$, $v_1 = (x_1, y_1, z_1)$, and $v_2 = (x_2, y_2, z_2)$

    $$
    area = (x_2 - x_0) * (y_1 - y_0) - (x_1 - x_0) * (y_2 - y_0)
    $$

  - Calculate the depth $depth$ of the pixel $v = (x, y)$ by combining the barycentric coordinates $(\alpha, \beta, \gamma)$ with the triangle screen coordinates $(v_0 = (x_0, y_0, z_0), v_1 = (x_1, y_1, z_1), v_2 = (x_2, y_2, z_2))$

    $$
    depth = \alpha * z_0 + \beta * z_1 + \gamma * z_2
    $$

  - Calculate the depth threshold $depth\_thres$ of the point $v = (x, y)$ by combining the depth map $d$ with the occlusion truncation $occlusion\_truncation$

    $$
    depth\_thres = d(x, y) * 0.49999 + 0.5 + occlusion\_truncation
    $$

  - Determine whether the point is inside the triangle based on the barycentric coordinates $(\alpha, \beta, \gamma)$. If $\alpha >= 0$, $\beta >= 0$, and $\gamma >= 0$, the point is inside the triangle (including on the edges of the triangle); otherwise, the point is not inside the triangle.
  - Calculate the perspective-corrected interpolation $(\lambda_0^{corrected}, \lambda_1^{corrected}, \lambda_2^{corrected})$ by combining the barycentric coordinates $(\lambda_0, \lambda_1, \lambda_2)$ with the three vertex coordinates of the triangle $v_0 = (x_0, y_0, z_0, w_0)$, $v_1 = (x_1, y_1, z_1, w_1)$, and $v_2 = (x_2, y_2, z_2, w_2)$

    $$
    \lambda_i^{corrected} = \frac{\lambda_i / w_i} { \sum (\lambda_j / w_j)}
    $$

## Parameter Description

- **Parameter description**:

  <table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attr</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>v</td>
      <td>Input</td>
      <td>Represents the input tensor of vertex coordinates in space, corresponding to `v` in the formula description, with a size of 2.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>f</td>
      <td>Input</td>
      <td>Represents the input tensor of faces in space, corresponding to `f` in the formula description, with a size of 2.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>d</td>
      <td>Optional input</td>
      <td><ul><li>Represents the input tensor of the depth map, used to calculate the depth threshold; this parameter does not take effect.</li><li>The default value is empty.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>height</td>
      <td>Input</td>
      <td><ul><li>Represents the screen height.</li><li>The value range is [1,4096].</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>width</td>
      <td>Input</td>
      <td><ul><li>Represents the screen width.</li><li>The value range is [1,4096].</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>occlusion_truncation</td>
      <td>Input</td>
      <td><ul><li>Occlusion truncation, used to calculate the depth threshold; this parameter does not take effect but must be passed in.</li><li>The default value is 0.0, and currently only 0.0 is supported.</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_depth_prior</td>
      <td>Input</td>
      <td><ul><li>Represents whether to apply the depth prior; this parameter does not take effect but must be passed in.</li><li>The default value is 0, and currently only 0 is supported.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>findices</td>
      <td>Output</td>
      <td>Represents the face index corresponding to the minimum depth of each pixel on the screen, corresponding to `findices` in the formula description, with a size of 2.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>barycentric</td>
      <td>Output</td>
      <td>Represents the output tensor of the perspective-corrected interpolation of barycentric coordinates solved based on the face with the minimum depth for each pixel on the screen, corresponding to `barycentric` in the formula description, with a shape of (height, width, 3); the last dimension corresponds to the perspective-corrected barycentric coordinates of the three vertices.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Only the input scenario with useDepthPrior set to 0 is supported, and the parameters d, occlusionTruncation, and useDepthPrior do not take effect in the actual calculation.

## Invocation Description

| Invocation Method   | Sample Code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_rasterizer](examples/test_aclnn_rasterizer.cpp) | Call the Rasterizer operator through the [aclnnRasterizer](docs/aclnnRasterizer_en.md) interface. |
