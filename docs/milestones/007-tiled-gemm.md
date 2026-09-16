# Milestone 7: selectable tiled GEMM and comparison harness

## Objective

Introduce a replaceable shared-memory GEMM candidate without changing tensor/runtime APIs or claiming an unmeasured speedup.

## Files

- cuda/matmul_tiled.cu and cuda/matmul.h: tiled kernel and launcher.
- include/forge/ops/matmul.h and src/ops/matmul.cpp: kernel selection overloads and validation.
- include/forge/runtime.h and src/runtime.cpp: graph-wide explicit MatMul implementation option.
- benchmarks/matmul_compare.cpp: reference-checked naive/tiled timing harness.
- tests/matmul_test.cpp, tests/runtime_test.cpp: both implementations, stream paths, tile boundaries, reuse modes and inference batches.
- tests/operator_validation_test.cpp, tests/graph_test.cpp: invalid kernel selection and host-only rejection.
- CMakeLists.txt, README.md, docs/architecture.md: targets and usage.
- docs/experiments/003-tiled-gemm.md: methodology and pending results.

## Decisions

Naive stays the default. The tiled candidate uses 16×16 shared-memory tiles with padding and barriers, Float32 accumulation, no Tensor Cores. Direct operators and compiled graphs support explicit selection. Existing signatures remain available. Performance acceptance awaits measured before/after results.

## Local validation

Mac Release build with SDK 26.5 passed; CTest passed 6/6. Runtime/example host syntax checks passed. `git diff --check` passed. CUDA code, CUDA-dependent benchmark and GPU tests cannot be compiled on this Mac and remain unverified.

```sh
cmake -S . -B build/host-sdk26 -DFORGE_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
cmake --build build/host-sdk26 -j
ctest --test-dir build/host-sdk26 --output-on-failure --timeout 20
```

## Deferred NVIDIA validation

Milestones 4–7 remain GPU-unvalidated, per the user's deferred-testing workflow. Full suite remains ten tests; MatMul/runtime tests now cover both kernels.

```bash
%%bash
set -e
cd /content/forge
cmake -S . -B build/cuda -DFORGE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/cuda -j
ctest --test-dir build/cuda --output-on-failure --timeout 60
./build/cuda/forge_matmul_compare
```

GPU correctness: NOT YET RUN. Performance: NOT YET MEASURED. No speedup is claimed and the default kernel is unchanged.

## Commit message

feat: add selectable tiled gemm and comparison benchmarks
