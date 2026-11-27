# ResizeFusionPass

## Fusion Pattern

This fusion rewrites each matched `Resize` operator (the GE adaptation of the ONNX Resize operator, whose prototype is registered in the CANN built-in IR) into a concrete target operator according to the `mode` attr, and prunes the redundant inputs. The rule is registered at the `CustomPassStage::kAfterBuiltinFusionPass` stage and is scheduled automatically during graph compilation; when the CANN built-in legacy pass of the same name fuses first, this rule matches no `Resize` node and naturally becomes a no-op, and it takes over the slot automatically once the legacy pass is removed.

The pattern matches by the `Resize` node type: the IR is 1 required input `x` plus 3 optional inputs `roi`/`scales`/`sizes`, and one pattern is built per possible number of connected optional inputs (4 forms in total).

**Scenario 1: static output shape (const branch)**

The `size` is extracted from the output shape according to the input format and fed as an int32 const input (NHWC -> `dims[1,2]`, NCHW -> `dims[2,3]`, NCDHW -> `dims[2,3,4]`; a spatial format outside this set produces a 0-element `size` const, line-by-line identical to the legacy behavior):

```text
        x                                x
        |                                |
      Resize      ==>   ResizeNearestNeighborV2 / ResizeBilinearV2
        |                                |   (size comes from an int32 const node)
        y                                y
```

- `nearest` -> `ResizeNearestNeighborV2`: a `ResizeNearestNeighborV2D` probe (via `GeUtils::CheckNodeSupportOnAicore`) is performed first; when supported the product stays in the V2D form (`size` as an attr, no const input). V2D currently has no kernel on any target platform, so the fallback to V2 + const always applies, matching the legacy `CheckOpSupported` behavior.
- `linear` -> `ResizeBilinearV2`.

**Scenario 2: dynamic output shape (from `resize_original_type == "onnx_resize"`), or linear mode with a 5-dim input (Slice branch)**

A `Slice(offsets={2}, size={2 or 3})` is inserted before the last connected input (`sizes`, falling back to `scales`/`roi` when absent) to extract the spatial dims:

```text
        x     roi  scales  sizes        x                sizes
         \     |     |     /            |                  |
          \    |     |    /     ==>    |    Slice(offsets=2) /
            Resize                     ResizeNearestNeighborV2 / ResizeBilinearV2 / ResizeTrilinear
              |                          |
              y                          y
```

- `nearest` -> `ResizeNearestNeighborV2`.
- `linear` 4-dim -> `ResizeBilinearV2`.
- `linear` 5-dim -> `ResizeTrilinear` (provided by the AICPU op store, verified executable on Ascend 950).

**Attr and input semantics**

- `coordinate_transformation_mode == "pytorch_half_pixel"` -> `half_pixel_centers=true`; `== "align_corners"` -> `align_corners=true`; everything else maps to both false.
- The target operator's `size` input desc is set explicitly to int32 (the Slice branch keeps the source input dtype).
- The redundant inputs such as `roi`/`scales` are kept as dangling boundary inputs of the replacement graph (equivalent to the legacy `DeleteInput` semantics, eliminated by the GE dead-code pass).
- The control edges entering the source node are preserved by the framework subgraph replacement mechanism (relayed through an inserted NoOp node).

## Constraints

- Structural constraints: matched by the `Resize` node type with all 1-to-4 input forms; no platform gate, active on all chips.
- Skip conditions:
  - Nodes specified to run on the DVPP engine (`_specified_engine_name == "DNN_VM_DVPP"` or `_specified_kernel_lib_name == "dvpp_ops_kernel"`) are not rewritten.
  - Nodes whose output `y` has fewer than 4 dims, whose `mode == "cubic"`, or whose dynamic shape does not come from `onnx_resize` are skipped.
  - In `nearest` mode the node is skipped when the `x` dtype is neither FLOAT16 nor FLOAT.
- A missing or invalid `mode` attr is deliberately let through and fails the whole build during this pass (keeping the legacy failure point).
- Product executability: whether the rewrite product is executable depends on the target operators (`ResizeNearestNeighborV2`/`ResizeBilinearV2`/`ResizeTrilinear`) being available in the platform op stores; the 5-dim `linear` product `ResizeTrilinear` is currently executable on Ascend 950 only and fails to build on the other platforms (identical to the legacy behavior).
- Version constraint: the fusion takes effect when the GE compiler version is at least 8.5.0 (`CustomPassStage::kAfterBuiltinFusionPass` is supported since 8.5.0).

## Applicable Products

This fusion rule has no platform gate; the applicable products follow the op-store availability of the rewrite target operators:

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

On the Atlas 200I/500 A2 inference products the target operators above have no kernel and the fusion product is not executable, so they are out of scope.
