# Milestone 3: stream-aware operators and launch checks

## Objective

Make CUDA execution explicitly stream-aware while preserving the existing MatMul API. Separate validation from hardware availability so API errors can be tested on the Mac.

## Files added or changed

- include/forge/stream.h and src/stream.cpp: owned, move-only non-blocking CUDA stream, synchronization, device restoration, opaque native handle.
- include/forge/ops/matmul.h and src/ops/matmul.cpp: explicit stream overload, shared validation, alias rejection, device-scoped dispatch, host-only rejection.
- cuda/matmul.h and cuda/matmul.cu: retained legacy launcher plus stream launcher, wide indexing, checked grid dimensions and launch failures.
- cuda/vector_add.cu: checked launch parameters/status and overflow-safe indexing/ceiling division.
- src/cuda_support.h: shared checked-call macro with operation/file/line context.
- benchmarks/matmul_benchmark.cpp and benchmarks/vector_add_benchmark.cpp: replace duplicate error macros with shared utility.
- tests/operator_validation_test.cpp: rank, shape, dtype, device, alias, missing storage, dimension-limit and host-only errors.
- tests/matmul_test.cpp: scalar, rectangular, non-tile-aligned and long-reduction cases against a double-precision host reference; both stream paths; dependent MatMul chain; stream moves and optional multi-GPU tests.
- CMakeLists.txt: build validation in both host/CUDA modes, add third host test.
- README.md and docs/architecture.md: stream ownership, synchronization, validation and limitations.

## Decisions

The original operator and six-argument kernel launcher remain available. Explicit streams are non-blocking and no per-operator barrier is introduced. Existing synchronous copies remain setup/readback operations. Caller storage must remain alive until work completes. CUDA error checks are centralized; destructors log cleanup errors instead of throwing. The naive algorithm is retained, but index arithmetic changed for correctness; performance must be measured again before comparing against historical results.

## Validation

Mac AppleClang 17 Release build passed, with CTest 3/3: tensor_metadata, memory_cpu, operator_validation. `git diff --check` passed. Host tests validate descriptors without simulating CUDA or executing a GPU kernel.

```sh
cmake -S . -B build/host-release -DFORGE_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/host-release -j
ctest --test-dir build/host-release --output-on-failure --timeout 20
```

User-supplied NVIDIA build/test output confirms successful compilation of the changed CUDA kernels, stream implementation, operators, benchmarks, and tests. CTest passed 5/5: tensor_metadata, memory_cpu, operator_validation, matmul_cuda, memory_cuda. The two GPU-labelled tests passed. Total CTest duration was 0.55 seconds; this is test-suite elapsed time, not inference latency. Large-index arithmetic is fixed in source but enormous allocations were not exercised. Multi-GPU checks report a skip when fewer than two devices are available. Performance: NOT YET MEASURED. This result validates the updated milestone 3 test suite. The supplied log does not include GPU model, device count, commit SHA, or successful test stdout, so execution of the optional multi-GPU branches is not confirmed.

Run in the NVIDIA notebook after updating its checkout:

```bash
%%bash
set -e
cd /content/forge
cmake -S . -B build/cuda -DFORGE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/cuda -j
ctest --test-dir build/cuda --output-on-failure --timeout 60
```

Expected tests: tensor_metadata, memory_cpu, operator_validation, matmul_cuda, memory_cuda (five total).

## Commit message

feat: add stream-aware matmul execution and launch validation
