# Lifetime-based storage reuse

## Problem

Dedicated storage for every intermediate retains allocations that are no longer needed after their last consumer.

## Hypothesis

Assigning non-overlapping lifetimes to shared storage slots should reduce requested allocation capacity for graphs with reusable intermediates.

## Baseline

Compile the same graph with reuse_memory=false. Both modes pad capacities to 256 bytes and use the same operators and execution order.

## Implementation

Inclusive lifetimes; outputs pinned through completion; deterministic best-fit free-slot selection. Allocate slots once per executable. No allocation/free during execution.

## Benchmark methodology

On NVIDIA hardware, record commit, GPU model, toolkit/compiler, graph shapes, batch size, both modes' statistics and numerical outputs. Compare reserved_bytes and allocation_count, while separately sampling GPU allocation overhead if desired. Do not interpret static peak_live_bytes as device telemetry. Timing experiments must use warmups and repeated measurements, and separate compilation from execution.

## Results

Host planner correctness tests passed, including independent slot-content simulation. Actual GPU results: NOT YET MEASURED.

## Interpretation

Planned capacity and tensor payload are distinct; padding can exceed the payload for tiny tensors. The best-fit method is not a global minimum. Single-stream ordering is essential to the reuse proof.

## Conclusion

Implemented and host-tested. GPU correctness and practical memory/performance effects remain to be measured.
