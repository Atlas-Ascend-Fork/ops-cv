# Fusion boundary

GammaCorrection is already a single fused kernel for Cast, Log, scalar multiply, Exp and Cast.
No graph rewrite is registered here because replacing arbitrary framework Pow nodes would change
the public operator contract and attribute validation semantics.
