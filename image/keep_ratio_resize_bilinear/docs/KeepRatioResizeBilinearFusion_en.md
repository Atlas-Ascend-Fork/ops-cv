# KeepRatioResizeBilinearFusion

## Fusion Pattern

This fusion rewrites each matched `KeepRatioResizeBilinear` operator (aspect-ratio-preserving bilinear resizing driven by `min_dimension`/`max_dimension`, whose prototype is registered in the CANN built-in IR) into `ResizeBilinearV2` + an int32 const `size`. The rule is registered at the `CustomPassStage::kAfterBuiltinFusionPass` stage and is scheduled automatically during graph compilation; when the CANN built-in legacy pass of the same name fuses first, this rule naturally becomes a no-op, and it takes over the slot automatically once the legacy pass is removed.

The pattern is a single node matched by the `KeepRatioResizeBilinear` node type.

**Target size computation (keep-ratio, line-by-line identical to the legacy pass; `floor` rounds toward negative infinity)**

```text
scale_min = min(H, W) / min_dimension
candidate1 = { floor(H / scale_min + 0.5), floor(W / scale_min + 0.5) }

scale_max = max(H, W) / max_dimension
candidate2 = { floor(H / scale_max + 0.5), floor(W / scale_max + 0.5) }

newH, newW = (longer side of candidate1 > max_dimension) ? candidate2 : candidate1
```

That is, the candidate is first computed by "scaling the shorter side up to `min_dimension`"; if its longer side exceeds `max_dimension`, the size is recomputed by "scaling the longer side up to `max_dimension`" instead. H/W comes from the `images` input: `dims[1,2]` under NHWC, `dims[2,3]` under any other format (including NCHW).

**Rewrite product**

```text
        images                          images
          |                               |
  KeepRatioResizeBilinear   ==>   ResizeBilinearV2
          |                        /    \       (size is an int32 const
          y                       y      Const size {newH, newW})
```

- The `align_corners`/`half_pixel_centers` attrs are passed through (default `false`).
- Before the rewrite a `ResizeBilinearV2D` probe (via `GeUtils::CheckNodeSupportOnAicore`) is performed: when supported the product stays in the V2D form (`size` as an attr, no const input). V2D currently has no kernel on any target platform, so the fallback to V2 + const always applies, matching the legacy const2attr attempt-and-fallback behavior.
- The target node keeps the source node's images/y descs (the `y` dtype is fixed to FLOAT, per the source operator IR).

## Constraints

- Platform constraint: the `NpuArch` is read via `rtGetSocSpec` and the rule is active on all chips except `DAV_5102` (MC62CM12A) - replicating the legacy registration mechanism (unconditional registration on the standard archs, plus a registration whitelist whose 3510 entry includes this pass and whose 5102 entry does not; every other arch is not governed by the whitelist at all).
- Structural constraints: an empty or non-4-dim `images` shape is deliberately let through and fails the whole build during this pass (unreachable in valid graphs: the built-in infershape intercepts it first); the fusion is skipped when the H/W dims are unknown (-1 or -2).
- Attr constraints: `min_dimension`/`max_dimension` must exist and be non-zero, otherwise the fusion is skipped (a missing attr keeps the initial 0 and hits the ==0 check, matching the legacy path that ignores the GetInt return value).
- Version constraint: the fusion takes effect when the GE compiler version is at least 8.5.0 (`CustomPassStage::kAfterBuiltinFusionPass` is supported since 8.5.0).

## Applicable Products

This fusion rule is active on all chips except `DAV_5102` (MC62CM12A); the applicable products follow the op-store availability of the rewrite target operator `ResizeBilinearV2`:

<!-- npu="950" id1 -->
<term>Ascend 950PR&950DT products</term>
<!-- end id1 -->

<!-- npu="A3" id2 -->
<term>Atlas A3 products</term>
<!-- end id2 -->

<!-- npu="910b" id3 -->
<term>Atlas A2 products</term>
<!-- end id3 -->

<!-- npu="310p" id4 -->
<term>Atlas inference products</term>
<!-- end id4 -->

<!-- npu="910" id5 -->
<term>Atlas training products</term>
<!-- end id5 -->

On the Atlas 200I/500 A2 inference products `ResizeBilinearV2` has no kernel and the fusion product is not executable, so they are out of scope.
