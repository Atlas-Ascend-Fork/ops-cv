# ResizeDFusionPass

## Fusion Pattern

This fusion rewrites each matched `ResizeD` operator (dynamic spatial-size resize, whose prototype is registered in the CANN built-in IR) into a concrete target operator according to `mode` and `data_format`. The rule is registered at the `CustomPassStage::kAfterBuiltinFusionPass` stage and is scheduled automatically during graph compilation; when the CANN built-in legacy pass of the same name fuses first, this rule naturally becomes a no-op, and it takes over the slot automatically once the legacy pass is removed.

The pattern is a single node matched by the `ResizeD` node type (`sizes` is a required attr).

**Scenario 1: `linear` (`data_format == "NCHW"`, input `[N, C, 1, W]`)**

The `sizes` attr becomes an int32 const input; the dimension is converted between `[N, C, 1, W]` and `[N, C, W]` by a Squeeze/Unsqueeze pair:

```text
        x                             x
        |                             |
      ResizeD        ==>          Squeeze(axis={-2})
        |                             |
        y                        ResizeLinear(scale=scales[0])
                                      |
                                  Unsqueeze(axes={-2})
                                      |
                                      y
```

- The first element of `scales` becomes the `scale` scalar of `ResizeLinear`.
- The Squeeze/Unsqueeze attr value keeps the legacy `{-2}` (the negative index of the squeezed H dim).

**Scenario 2: `cubic` + NCHW**

Rewritten directly to `ResizeBicubicV2` (with `scales` passed through as ListFloat).

**Scenario 3: `cubic` + HWNC**

A `Transpose(perm=[2, 3, 0, 1])` is inserted first to convert the input to NCHW, then the node is rewritten to `ResizeBicubicV2`:

```text
        x                             x
        |                             |
      ResizeD        ==>          Transpose(perm=[2,3,0,1])
        |                             |
        y                        ResizeBicubicV2(scales)
                                      |
                                      y
```

**Attr semantics**

- The target operator gets `align_corners=true` only when `coordinate_transformation_mode == "align_corners"`; everything else maps to false.
- The `sizes` attr (1 target W element for `linear`, 2 target H/W elements for `cubic`) becomes an int32 const `size` input.

## Constraints

- Platform constraint: the `NpuArch` is read via `rtGetSocSpec` and the fusion applies on `DAV_3510` (Ascend 950 series) only; on every other platform the rule skips after the match and `ResizeD` runs unfused, with its native kernel as provided by the platform op stores (verified on the Atlas A2 series). The gate replicates the legacy semantics of the code-level `IsRegBase()` gate intersected with the registration whitelist that contains the 3510 entry only.
- Structural constraints: `mode ∈ {linear, cubic}` (the default `nearest` is not fused); a known-rank `x`/`y` is 4-dim, and an unknown rank (`[-2]`) skips the rank checks and still fuses.
- Attr constraints:
  - `data_format`: NCHW only for `linear`; NCHW or HWNC for `cubic`.
  - `sizes`: exactly 1 positive element for `linear`, exactly 2 positive elements for `cubic`; `scales`: exactly 1 element for `linear`, exactly 2 for `cubic` (a missing attr is checked with its default value).
  - In `linear` mode the input H dim is 1 (`[N, C, 1, W]`, guaranteed by the built-in infershape), and a known-rank `x` has non-zero C/H/W dims.
- Dtype constraints: `x` is in {FLOAT16, FLOAT, BFLOAT16} (as the legacy pass); the `y` dtype/format must equal those of `x`.
- Version constraint: the fusion takes effect when the GE compiler version is at least 8.5.0 (`CustomPassStage::kAfterBuiltinFusionPass` is supported since 8.5.0).

## Applicable Products

<!-- npu="950" id1 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id1 -->
