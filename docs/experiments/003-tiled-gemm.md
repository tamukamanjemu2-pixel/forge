# Shared-memory tiled GEMM

## Problem

The naive kernel loads operands from global memory for each output, with no explicit block-level reuse.

## Hypothesis

Cooperative tile loading and shared-memory reuse may improve Float32 GEMM performance for some matrix sizes, at the cost of barriers and edge padding.

## Baseline

Use the current naive implementation in the same binary and on the same GPU as the tiled candidate. Preserve historical T4 results as historical evidence; do not use them as if they were collected under this new harness.

## Implementation

16×16 shared tiles, one output per thread, zero-padded edge loads, synchronization before consumption and before tile reuse. Explicit selection; naive remains default. No Tensor Core use.

## Benchmark methodology

Build Release and run the full correctness suite first. Run forge_matmul_compare, which uses deterministic signed inputs and checks every output against a double-precision CPU reference before timing. Defaults cover square sizes 64, 256, 512 and rectangular M=127, N=131, K=67. It records GPU model/capability, CUDA headers/runtime/driver, host compiler, shape, dtype, warmups and sample counts.

Ten warmups and fifty measured iterations are defaults; CLI allows overrides. Transfers, allocation, reference calculation, and result validation are excluded. CUDA events bracket operator submission on one explicit stream. These intervals may include GPU idle time caused by host validation/dispatch. Host timing includes event calls and completion wait. Neither substitutes for profiler kernel-only timing or end-to-end model latency.

Report event minimum/median/p95, host median, effective event GFLOP/s, max absolute error and the ratio of naive/tiled event medians. Ratios are computed only from the current run. Execution order is fixed (naive then tiled), clocks are not locked, and multiple runs are needed to assess drift and variance. Record the commit alongside saved output.

## Results

CUDA correctness: NOT YET RUN. Performance: NOT YET MEASURED.

## Interpretation

No performance conclusion is available. Edge padding, synchronization cost and host dispatch overhead can dominate small shapes. Nsight measurements will be needed to attribute any observed change to the kernel.

## Conclusion

Candidate implemented; not accepted as a measured optimization. Naive remains the default until correctness and repeatable comparisons support changing it.
