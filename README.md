# Forge

A C++20/CUDA GPU inference runtime under development. The current local implementation contains non-owning contiguous tensor metadata, move-only CPU/CUDA storage buffers, synchronous copies, owned CUDA streams, Float32 CUDA MatMul, ReLU, and last-axis Softmax operators, naive GEMM and vector-add kernels, GPU benchmarks, and correctness tests. Graph construction and a compiled single-stream execution runtime are implemented; cross-executable memory pooling, FP16, and Tensor Core execution are future milestones.

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

The user-run CUDA 12.8.93 / GCC 13.3.0 build passed all four tests for milestone 2. The subsequent milestone 3 CUDA build also succeeded and passed all five tests, including operator_validation and the expanded matmul_cuda suite. No new GPU performance measurements have been collected for milestones 1–3. Performance is **NOT YET MEASURED** for subsequent optimizations. Existing kernels and their benchmarks are preserved. Future reports must distinguish kernel timing from transfers and end-to-end time, and record hardware, toolchain, dimensions, dtype, warmup, and iteration counts.

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
- [x] Selectable shared-memory tiled GEMM (GPU correctness/performance pending)
- [x] ReLU and stable last-axis Softmax (GPU validation pending)
- [ ] FP16 execution and Tensor Core path
- [x] Per-executable storage slots and tensor-lifetime reuse (GPU validation pending)
- [x] Computation graph and single-stream execution scheduler (GPU validation pending)
- [ ] CUDA streams, batching, and asynchronous execution
- [x] Opt-in MatMul–ReLU fusion (GPU correctness/performance pending)
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

`Buffer` also accepts `Device::cuda(index)` in CUDA builds. `copy_tensor(source, destination)` requires matching shape and dtype and completes synchronously. CPU buffers are 64-byte aligned. Views validate capacity and element alignment. CUDA transfers are intended for setup/readback; stream-aware MatMul is available separately; graph scheduling, asynchronous transfer APIs and cross-executable pooling are not implemented yet.

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

## Activation operators

```cpp
#include "forge/ops/relu.h"
#include "forge/ops/softmax.h"

forge::relu(input, hidden, stream);
forge::softmax(hidden, probabilities, stream);
stream.synchronize();
```

Both operators require matching contiguous Float32 CUDA tensors with non-overlapping storage. The three-argument form enqueues on the supplied stream; omitting the stream uses the tensor device's default stream. ReLU accepts scalars and arbitrary ranks and propagates NaN. Softmax requires rank >= 1 and normalizes each row along the last dimension. Its supported input contract is finite logits; it subtracts the row maximum before exponentiation. Non-finite logits have no defined probability semantics and are not checked by a host-side scan.

Milestone 4 host validation passed; GPU compilation and correctness are pending. No activation performance results have been measured.

## Graph inference

The `forge_inference` example uploads inputs/weights and executes the two-layer model through the graph/runtime:

```cpp
forge::Graph graph;
auto input = graph.input(x);
auto weights1 = graph.input(w1);
auto weights2 = graph.input(w2);
auto hidden = graph.relu(graph.matmul(input, weights1));
auto output = graph.softmax(graph.matmul(hidden, weights2));
graph.output(output);
forge::Runtime runtime;
auto executable = runtime.compile(graph);
runtime.execute(executable);
// Copy executable.output(output) to a matching CPU tensor to read results.
```

Inputs and weights are borrowed CUDA tensors and must stay alive. Compilation owns intermediate/output allocations; subsequent executions reuse them. Execution blocks once at the end of the graph. Marked outputs become available after a successful run. Graphs are single-device and Float32; matrix batch size is the first dimension. Dead-node pruning, asynchronous graph submission, and cross-executable pooling are not implemented yet.

After building on NVIDIA hardware, run `./build/cuda/forge_inference`. GPU validation for milestones 4–8 remains deferred. The current full suite registers eleven tests.

If the Mac linker rejects `arm64e.x1` in the selected macOS 27 SDK, the locally verified workaround is a separate build using the installed 26.5 SDK:

```sh
cmake -S . -B build/host-sdk26 -DFORGE_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
cmake --build build/host-sdk26 -j
ctest --test-dir build/host-sdk26 --output-on-failure --timeout 20
```

## Memory planning

Graph compilation now assigns computed tensors to reusable storage slots. Inputs remain borrowed, outputs stay live through completion, and a slot is reused only after its prior value's final consumer. Slot capacities are rounded to 256 bytes. This plan relies on the runtime's single-stream ordering.

```cpp
auto executable = runtime.compile(graph); // lifetime reuse enabled
const auto& stats = executable.memory_statistics();
// stats.tensor_bytes, reserved_bytes, peak_live_bytes, allocation_count, reuse_count
// Disable reuse for an independent execution/memory baseline:
auto baseline = runtime.compile(graph, {.reuse_memory = false});
```

`allocated_bytes()` now reports reserved slot capacities including padding. Statistics describe the plan and requested allocations, not total GPU/driver memory or measured performance. Small tensors may reserve more bytes than their payload due to padding. Slots are owned by one executable; there is no global pool or per-operation allocation/free.

## GEMM implementations and comparison

Naive GEMM remains the default. Opt into the 16×16 shared-memory tiled candidate explicitly:

```cpp
forge::matmul(a, b, output, stream, forge::MatMulKernel::Tiled);
auto executable = runtime.compile(graph, {.matmul_kernel = forge::MatMulKernel::Tiled});
```

The tiled path is Float32 only and uses neither FP16 nor Tensor Cores. It is implemented but has not yet been compiled/tested on NVIDIA hardware or benchmarked. Kernel selection does not change the public tensor contract or stream policy.

After the GPU test suite passes, compare on the same GPU/build:

```sh
./build/cuda/forge_matmul_compare
# Optional: M N K warmup_iterations measured_iterations
./build/cuda/forge_matmul_compare 512 512 512 20 100
```

The benchmark prints GPU/toolchain metadata and CSV timing rows. Full CPU-reference checks, allocations and transfers are outside timing. CUDA-event intervals bracket operator submission and may include GPU idle time due to host dispatch; host timings additionally include event calls and waiting. Neither is an end-to-end model measurement. The printed ratio is computed from the current run's event medians, not historical baseline numbers. Defaults are ten warmups and fifty samples per implementation for each shape. Large custom dimensions also incur a full CPU GEMM reference calculation.

## MatMul–ReLU fusion

```cpp
auto executable = runtime.compile(graph, {
    .matmul_kernel = forge::MatMulKernel::Tiled,
    .fuse_matmul_relu = true
});
```

Fusion is off by default. Only a MatMul whose sole consumer is ReLU and whose raw result is not a graph output is eligible. The fused CUDA epilogue applies ReLU before storing the final GEMM result. Both naive and tiled implementations support it. Compilation replans tensor lifetimes after transformation; no unactivated intermediate buffer is allocated for an eligible pair.

`fused_pairs()` and `planned_kernel_launches()` report compile-time counts, not profiler telemetry. For later measurements, run `./build/cuda/forge_fusion_compare` after correctness tests pass. It compares separate and fused MatMul–ReLU for each kernel, using the same data/reference and event/host timing methodology as the GEMM comparison. Logical intermediate bytes are an analytical read/write payload estimate, not measured DRAM traffic. GPU correctness and performance are still pending.

See `docs/status.md` for completed work, validation gaps, and remaining project stages.
