# Milestone 2: storage ownership and transfers

## Objective

Separate owned memory from Tensor metadata, providing a small foundation for future runtime allocation and pooling without changing the MatMul API.

## Files

- include/forge/memory.h: move-only Buffer, bounded borrowed views, synchronous tensor copy API.
- src/memory.cpp: aligned CPU allocation, CUDA allocation, transfers and cleanup.
- src/cuda_support.h: reusable CUDA error handling and scoped device selection.
- tests/memory_test.cpp: CPU ownership, bounds, alignment, moves, overlap, invalid copies, host-only rejection.
- tests/memory_cuda_test.cpp: dtype byte roundtrips, device copies, overlap rejection, moved storage, optional multi-GPU restoration and cross-GPU rejection.
- tests/matmul_test.cpp: use owned buffers and transfer API; reject non-finite output.
- CMakeLists.txt: storage implementation and CPU/GPU test registration.
- README.md and docs/architecture.md: ownership contract, usage, limitations.

## Design decisions

Tensor remains a borrowed view and allocation-free. Buffer is explicitly owned and movable. Its allocation can later be held by a runtime pool. Copy operations are blocking setup/readback operations. No performance claim or asynchronous behavior is implied. CUDA destruction logs failed cleanup rather than throwing during stack unwinding.

## Local validation

AppleClang 17, Release: configure and build passed. CTest passed 2/2 (tensor_metadata and memory_cpu). No simulated CUDA execution. GPU code has not been compiled or run in this milestone. The sanitizer timeout recorded in milestone 1 is not evidence of sanitizer validation for this milestone.

```sh
cmake -S . -B build/host-release -DFORGE_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/host-release -j
ctest --test-dir build/host-release --output-on-failure --timeout 20
```

## NVIDIA validation checkpoint

Run on the T4 before proceeding with GPU-dependent changes. Preserve and share any actual compiler/test failures.

```sh
cmake -S . -B build/cuda -DFORGE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/cuda -j
ctest --test-dir build/cuda --output-on-failure --timeout 60
```

Expected registered tests: tensor_metadata, memory_cpu, matmul_cuda, memory_cuda. Multi-GPU branches run only when at least two devices are present. Actual GPU result: NOT YET RUN. Performance: NOT YET MEASURED. Existing baseline measurements remain unchanged.

## Commit message

feat: add owned buffers and validated tensor transfers
