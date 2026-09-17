# Milestone 9: FP16 execution with Float32 accumulation

## Objective

Execute existing operators and compiled graphs using FP16 storage while retaining Float32 accumulation/reductions and explicit numerical contracts. This is not Tensor Core execution.

## Files

- cuda/fp16.h, cuda/fp16.cu: naive/tiled GEMM, fused ReLU, standalone ReLU and stable Softmax.
- src/ops/matmul.cpp, src/ops/activations.cpp: dtype validation and dispatch.
- src/graph.cpp: Float16 inputs, output inference and mixed-operand rejection.
- include/forge/ops/*.h: dtype contracts.
- tests/fp16_test.cpp: operator/graph reference tests with quantized inputs and half-rounded storage boundaries.
- tests/graph_test.cpp, tests/operator_validation_test.cpp, tests/activation_validation_test.cpp: host dtype and unavailable-backend checks.
- benchmarks/matmul_compare.cpp, CMakeLists.txt: FP16 GEMM/fusion benchmark targets and GPU test.
- README.md, docs/status.md, docs/architecture.md, experiment note: implementation and validation limits.

## Decisions

Float16 inputs/output storage with Float32 arithmetic. No implicit dtype conversion. Softmax recomputes exponentials instead of rounding intermediate exponentials to half. Fusion converts the final activated accumulator only once. Tensor Cores remain a future explicit implementation.

## Local validation

Mac Release build with SDK 26.5 passed; seven host tests passed. CUDA-dependent kernels, reference tests and benchmark variants are not compiled or run locally. `git diff --check` passed. NVIDIA testing remains deferred by request.

```sh
cmake -S . -B build/host-sdk26 -DFORGE_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
cmake --build build/host-sdk26 -j
ctest --test-dir build/host-sdk26 --output-on-failure --timeout 20
```

## Deferred GPU validation

Full suite now registers twelve tests including fp16_cuda. Its GEMM checks cover scalar, edge-tile and long-reduction shapes; both kernels, fusion choices and stream paths. Softmax includes extreme finite logits, widths through 4097 and a 65,536-row case. Graph tests cover fusion, reuse and repeated execution. GEMM/graph tolerance is 0.002 absolute plus 0.002 relative; Softmax comparison uses 0.00002 absolute plus 0.002 relative against half-rounded reference probabilities, with row-sum tolerance 0.002.

```bash
%%bash
set -e
cd /content/forge
cmake -S . -B build/cuda -DFORGE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/cuda -j
ctest --test-dir build/cuda --output-on-failure --timeout 60
./build/cuda/forge_fp16_compare
./build/cuda/forge_fp16_fusion_compare
```

GPU correctness: NOT YET RUN. FP16 performance: NOT YET MEASURED. Milestones 4–9 still require NVIDIA validation.

## Commit message

feat: add fp16 operators with float32 accumulation
