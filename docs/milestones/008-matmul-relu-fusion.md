# Milestone 8: opt-in MatMul–ReLU fusion

## Objective

Fuse a semantically safe MatMul–ReLU pair into a single CUDA launch and eliminate the raw intermediate allocation without changing graph output identities.

## Files

- include/forge/graph_plan.h, src/graph_plan.cpp: pure host graph transformation and planned launch counts.
- include/forge/graph.h: internal fused/elided operation tags.
- include/forge/memory_plan.h, src/memory_plan.cpp: plan lifetimes/storage from transformed dependencies.
- include/forge/runtime.h, src/runtime.cpp: fusion option, plan statistics and fused dispatch.
- include/forge/ops/matmul.h, src/ops/matmul.cpp: fused operator entry points and shared validation.
- cuda/matmul.h, cuda/matmul.cu, cuda/matmul_tiled.cu: fused epilogue specializations for both kernels.
- tests/fusion_plan_test.cpp: eligibility, protected outputs, branching, transformed lifetimes and slot-content checks.
- tests/matmul_test.cpp, tests/runtime_test.cpp, tests/operator_validation_test.cpp: fused/unfused reference paths, options and host rejection.
- benchmarks/matmul_compare.cpp, CMakeLists.txt: shared benchmark built as GEMM and fusion comparisons.
- README.md, docs/architecture.md, docs/status.md and experiment note: contracts, status and measurement plan.

## Decisions

Fusion remains off by default. Multi-consumer or observable raw MatMul results prevent fusion. Preserve original indices with an elided marker. Schedule fused work at the ReLU position and replan lifetimes accordingly. Report planned launch counts and analytical intermediate bytes separately from measured timing or profiler traffic.

## Local validation

Mac Release build using SDK 26.5 passed; CTest passed 7/7 including fusion_planning. Host syntax checks for runtime/example code passed with warnings enabled. `git diff --check` passed. NVIDIA compilation and execution are deferred; milestones 4–8 remain GPU-unvalidated.

```sh
cmake -S . -B build/host-sdk26 -DFORGE_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
cmake --build build/host-sdk26 -j
ctest --test-dir build/host-sdk26 --output-on-failure --timeout 20
```

## Later NVIDIA checks

```bash
%%bash
set -e
cd /content/forge
cmake -S . -B build/cuda -DFORGE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/cuda -j
ctest --test-dir build/cuda --output-on-failure --timeout 60
./build/cuda/forge_fusion_compare
```

Full suite: eleven tests. MatMul/runtime tests span both kernels, both fusion choices, memory reuse options and stream paths. No speedup or measured traffic reduction is claimed.

## Commit message

feat: add opt-in matmul relu fusion and comparison benchmarks
