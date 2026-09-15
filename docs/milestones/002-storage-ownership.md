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

AppleClang 17, Release: configure and build passed. CTest passed 2/2 (tensor_metadata and memory_cpu). No simulated CUDA execution. Subsequent user-run NVIDIA validation is recorded below. The sanitizer timeout recorded in milestone 1 is not evidence of sanitizer validation for this milestone.

```sh
cmake -S . -B build/host-release -DFORGE_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/host-release -j
ctest --test-dir build/host-release --output-on-failure --timeout 20
```

## NVIDIA validation checkpoint

The user ran the following NVIDIA build/test workflow and supplied successful output on 2026-09-14.

```sh
cmake -S . -B build/cuda -DFORGE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/cuda -j
ctest --test-dir build/cuda --output-on-failure --timeout 60
```

User-supplied validation evidence:

- GNU C++ compiler 13.3.0; NVIDIA CUDA compiler/toolkit 12.8.93.
- Configure and all build targets succeeded in /content/forge/build/cuda.
- CTest passed 4/4: tensor_metadata, memory_cpu, matmul_cuda, memory_cuda.
- GPU-labelled tests passed 2/2. CTest total elapsed time was 0.54 seconds; this is test-suite duration, not inference latency.

GPU model, device count, and commit SHA are not included in the supplied output. Multi-GPU branches run only when at least two devices are present, so their execution is not confirmed by this log. GPU correctness checkpoint passed for this test suite. Performance: NOT YET MEASURED. Existing baseline measurements remain unchanged.

## Commit message

feat: add owned buffers and validated tensor transfers
