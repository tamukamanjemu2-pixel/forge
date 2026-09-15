# Forge

A C++20/CUDA GPU inference runtime under development. The current local implementation contains non-owning contiguous tensor metadata, move-only CPU/CUDA storage buffers, synchronous copies, owned CUDA streams, a Float32 CUDA MatMul operator, naive GEMM and vector-add kernels, GPU benchmarks, and correctness tests. Graph execution, memory pooling, fusion, FP16, and Tensor Core execution are future milestones.

## Build and test on macOS

This builds and tests host metadata, CPU storage/copies, and operator validation only. It does not simulate CUDA or provide CPU operator execution.

```sh
cmake -S . -B build/host-release -DFORGE_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/host-release -j
ctest --test-dir build/host-release --output-on-failure
```

In CLion, add `-DFORGE_ENABLE_CUDA=OFF` to the macOS CMake profile.

## Build and test on NVIDIA hardware

Requires CMake 3.24+, a C++20 compiler, and a compatible NVIDIA CUDA Toolkit. CUDA remains enabled by default. Architecture 75 targets the Tesla T4; override it for other hardware. Use a separate build directory from the host build.

```sh
cmake -S . -B build/cuda -DFORGE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/cuda -j
ctest --test-dir build/cuda --output-on-failure
./build/cuda/forge_vector_add
./build/cuda/forge_matmul_benchmark
```

`-DBUILD_TESTING=OFF` omits test executables. GPU benchmarks are available only with CUDA enabled.

## Measurement discipline

The user-run CUDA 12.8.93 / GCC 13.3.0 build passed all four tests for milestone 2. Milestone 3 CUDA validation is pending. No new GPU performance measurements have been collected for milestones 1–3. Performance is **NOT YET MEASURED** for subsequent optimizations. Existing kernels and their benchmarks are preserved. Future reports must distinguish kernel timing from transfers and end-to-end time, and record hardware, toolchain, dimensions, dtype, warmup, and iteration counts.

See `docs/architecture.md` and `docs/milestones/` for implementation and validation checkpoints.

## Baseline results

Measured on an NVIDIA Tesla T4 (compute capability 7.5). These results establish the unoptimized baseline for later kernel work.

### Vector addition

| Elements | Best observed time | Effective bandwidth |
|---:|---:|---:|
| 1,048,576 | ~0.051 ms | ~248 GB/s |
| 4,194,304 | ~0.193 ms | ~261 GB/s |

### Naive matrix multiplication

20 benchmark iterations per size.

| Matrix size | Kernel time | Throughput |
|---:|---:|---:|
| 256 × 256 | 0.1571 ms | 213.54 GFLOP/s |
| 512 × 512 | 0.6943 ms | 386.65 GFLOP/s |
| 1024 × 1024 | 5.4875 ms | 391.34 GFLOP/s |
| 2048 × 2048 | 34.7110 ms | 494.94 GFLOP/s |

> These are baseline kernel measurements, not comparisons against cuBLAS. Future results will report speedups only against clearly defined and reproducible baselines.

## Repository layout

```text
forge/
├── benchmarks/       # CUDA microbenchmarks
├── cuda/             # CUDA kernels and launch interfaces
├── docs/             # Architecture and experiment notes
├── include/forge/    # Public runtime and operator headers
├── src/              # Runtime and operator implementations
├── tests/            # Correctness tests
└── CMakeLists.txt
```

## Roadmap

- [x] Non-owning tensor metadata and caller-managed CPU/CUDA storage
- [x] Host-only build and validated metadata tests
- [x] Move-only storage ownership and checked views (CPU and CUDA tests passed)
- [x] Vector-add baseline and bandwidth benchmark
- [x] Naive GEMM baseline, operator, tests, and benchmark
- [ ] Shared-memory tiled GEMM
- [ ] Activation and softmax kernels
- [ ] FP16 execution and Tensor Core path
- [ ] GPU memory pool and tensor-lifetime reuse
- [ ] Computation graph and execution scheduler
- [ ] CUDA streams, batching, and asynchronous execution
- [ ] Operator fusion
- [ ] NVIDIA Nsight profiling and bottleneck reports
- [ ] Multi-GPU execution experiments

## Benchmark discipline

Each optimization should include:

1. A correctness test.
2. A reproducible benchmark configuration.
3. Hardware and software environment details.
4. Latency and throughput before and after the change.
5. A short explanation of the measured bottleneck and tradeoff.

## Storage API

```cpp
#include "forge/memory.h"

forge::Buffer storage(32 * sizeof(float), forge::Device::cpu());
auto tensor = storage.view({4, 8}, forge::DataType::Float32);
// storage owns the allocation; tensor borrows it.
// Keep storage alive until all uses of tensor have completed.
```

`Buffer` also accepts `Device::cuda(index)` in CUDA builds. `copy_tensor(source, destination)` requires matching shape and dtype and completes synchronously. CPU buffers are 64-byte aligned. Views validate capacity and element alignment. CUDA transfers are intended for setup/readback; stream-aware MatMul is available separately; graph scheduling, asynchronous transfer APIs, and pooling are not implemented yet.

## Explicit CUDA stream execution

```cpp
#include "forge/stream.h"
#include "forge/ops/matmul.h"

forge::Stream stream(a.device());
forge::matmul(a, b, output, stream);
stream.synchronize();
// It is now safe to read back output with copy_tensor or release its storage.
```

The existing `matmul(a, b, output)` overload still enqueues on the tensor device's default stream. Neither overload synchronizes after launch. An explicit Stream uses a non-blocking CUDA stream and must belong to the tensors' device. Keep storage alive until completion, and synchronize before reading results with the blocking copy API. Launch failures throw immediately; execution failures can surface at synchronization. Stream destruction releases the handle but is not a completion/error-checking substitute for `synchronize()`.
