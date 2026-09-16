# Milestone 6: lifetime-based storage reuse

## Objective

Reuse owned CUDA buffers when tensor lifetimes do not overlap, while preserving graph outputs and keeping operator execution allocation-free.

## Files

- include/forge/memory_plan.h and src/memory_plan.cpp: pure host planning, lifetime intervals, best-fit slot assignment and checked statistics.
- include/forge/runtime.h and src/runtime.cpp: reusable slot allocation, memory statistics, and no-reuse compile option.
- tests/memory_plan_test.cpp: independent slot-content simulation, chains, branches, outputs, padding, overflow and deterministic generated DAGs.
- tests/runtime_test.cpp: both reuse modes, repeated execution and retained early outputs.
- examples/inference.cpp: report reserved capacity and reuse assignments.
- CMakeLists.txt, README.md, docs/architecture.md: integration and contracts.
- docs/experiments/004-lifetime-memory-plan.md: experiment methodology and pending GPU results.

## Decisions

Per-executable ownership; no global pool. Inclusive lifetimes prevent same-operation input/output aliasing. Marked outputs remain live to the end. Best-fit slots are rounded to 256 bytes; allocation happens only at compile time. The existing compile(graph) API enables reuse; compile(graph, {.reuse_memory = false}) provides a dedicated-storage baseline.

## Validation

Mac Release build with the installed 26.5 SDK passed. CTest passed 6/6, including memory_planning. The planner tests simulate slot contents across 32 deterministic 80-operation DAGs, verifying consumers see the original producer and marked outputs are retained. They also test known chain lifetimes, disabled reuse, mixed sizes, borrowed inputs, capacity overflow and invalid alignment. Runtime/example C++20 syntax checks passed with warnings enabled. `git diff --check` passed.

```sh
cmake -S . -B build/host-sdk26 -DFORGE_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
cmake --build build/host-sdk26 -j
ctest --test-dir build/host-sdk26 --output-on-failure --timeout 20
```

## Deferred GPU checks

GPU compilation/execution remains deferred by user request. Milestones 4–6: NOT YET GPU VALIDATED. No measured GPU memory savings or latency improvements are claimed.

```bash
%%bash
set -e
cd /content/forge
cmake -S . -B build/cuda -DFORGE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/cuda -j
ctest --test-dir build/cuda --output-on-failure --timeout 60
./build/cuda/forge_inference
```

Expected suite: ten tests, including memory_planning and runtime_cuda. Runtime tests run both reuse modes against the host reference and check that early outputs survive later operations and repeated executions.

## Commit message

feat: reuse graph storage based on tensor lifetimes
