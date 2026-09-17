# MatMul–ReLU fusion

## Problem

Separate GEMM and ReLU launch twice and materialize an unactivated intermediate that is read by ReLU.

## Hypothesis

A GEMM epilogue may reduce latency by removing a launch and the intermediate write/read when the raw result has no other observer.

## Baseline

Separate GEMM and ReLU, compared against the fused specialization of the same GEMM kernel. Compare naive with naive and tiled with tiled; do not confound fusion with a kernel change.

## Implementation

Compile-time CUDA epilogue specialization. Graph fusion is allowed only for a sole ReLU consumer when the raw GEMM result is not marked as output. Replan storage on the transformed dependencies.

## Benchmark methodology

Run the full correctness suite before forge_fusion_compare. The shared harness checks every output against double-precision GEMM followed by ReLU, uses identical data/shapes, and reports metadata, warmups, sample counts, event intervals and host durations. Transfers/reference work are excluded. Event intervals can contain host-submission gaps. Run order is naive then tiled, and separate then fused within each kernel; repeat runs to assess drift. Planned launch counts are structural. Logical intermediate bytes represent a write plus read of the unactivated Float32 output, not profiler-measured DRAM traffic. Use Nsight later for actual traffic attribution.

## Results

Host fusion/lifetime tests passed. GPU correctness: NOT YET RUN. Performance: NOT YET MEASURED.

## Interpretation

Removing a launch and intermediate does not establish a measured speedup. Small shapes, event overhead, GPU clocks and kernel behavior can affect the outcome.

## Conclusion

Candidate implemented and opt-in. Performance acceptance awaits GPU results.
