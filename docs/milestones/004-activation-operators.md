# Milestone 4: ReLU and stable Softmax

## Objective

Add the remaining standalone operators needed by the planned two-layer inference graph using the established stream-aware dispatch and storage contracts.

## Files

- include/forge/ops/relu.h, include/forge/ops/softmax.h: public default/explicit stream APIs.
- src/ops/activations.cpp: shared shape, dtype, device, layout, storage, alias and stream validation plus backend dispatch.
- cuda/activations.h, cuda/relu.cu, cuda/softmax.cu: launch interfaces, ReLU and stable last-axis Softmax kernels.
- tests/activation_validation_test.cpp: host-executable invalid input checks and host-only dispatch rejection.
- tests/activation_test.cpp: GPU reference comparisons, numeric boundaries, same-stream chain, multiple ranks, grid-stride rows, and optional multiple-device checks.
- CMakeLists.txt: new operator/backend sources and tests.
- README.md and docs/architecture.md: API contracts and implementation status.

## Decisions

Float32 and contiguous tensors only. ReLU accepts scalars; Softmax requires at least one dimension. Last-axis Softmax accommodates vectors, matrices, and leading batch dimensions. Finite logits are the supported Softmax input domain. Input/output aliasing is rejected. Both operators enqueue without synchronizing and perform no allocations.

Softmax uses a maximum-subtracted formulation with shared-memory reductions. This is an initial correctness implementation, not a measured optimization. ReLU uses a grid-stride loop. FP16 and fusion are deferred.

## Local validation

Mac Release configure/build and CTest passed 4/4: tensor_metadata, memory_cpu, operator_validation, activation_validation. CUDA is not simulated. `git diff --check` passed.

```sh
cmake -S . -B build/host-release -DFORGE_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/host-release -j
ctest --test-dir build/host-release --output-on-failure --timeout 20
```

## Deferred GPU validation

User requested testing later. CUDA compilation and activation correctness: NOT YET RUN. Performance: NOT YET MEASURED. This milestone does not inherit CUDA validation from earlier milestones.

When ready, update the GPU checkout and run:

```bash
%%bash
set -e
cd /content/forge
cmake -S . -B build/cuda -DFORGE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/cuda -j
ctest --test-dir build/cuda --output-on-failure --timeout 60
```

Expected tests: tensor_metadata, memory_cpu, operator_validation, activation_validation, matmul_cuda, activations_cuda, memory_cuda (seven total). Tests cover Softmax widths 1, 3, 31, 32, 255, 256, 257, 1025, 4097; 65,536 rows; extreme/equal logits; probability row sums; scalar and multi-rank ReLU; NaN/infinity ReLU behavior; both stream paths and a ReLU-to-Softmax chain. Multi-GPU checks explicitly skip on single-device systems.

## Commit message

feat: add stream-aware relu and stable softmax operators
