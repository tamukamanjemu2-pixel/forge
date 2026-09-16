# Milestone 5: graph compilation and execution

## Objective

Execute Input → MatMul → ReLU → MatMul → Softmax through a graph and runtime instead of manually scheduling operators.

## Files

- include/forge/graph.h and src/graph.cpp: graph-owned value identities, shape inference, output marking and dependency ordering.
- include/forge/runtime.h and src/runtime.cpp: compiled graph snapshot, owned intermediate storage, stream dispatch and output access.
- tests/graph_test.cpp: host validation for graph ordering, branches, duplicate dependencies, foreign handles, shape/device/type errors, moves and missing bindings.
- tests/runtime_test.cpp: GPU inference reference comparisons, repeated runs, changed inputs, builder destruction, executable moves and input-only graphs.
- examples/inference.cpp: complete graph-based two-layer example.
- CMakeLists.txt: graph/core and runtime targets, example and test registrations.
- README.md and docs/architecture.md: usage, ownership and limitations.

The initial graph/runtime sources and CMake entries were already in the local checkout when work resumed. This completion adds the referenced missing tests/example and validates the assembled host build; it does not rewrite user commits.

## Decisions

Graphs are single-device Float32 DAGs. Input pointers are borrowed and captured during compilation. Explicit outputs permit multiple results. Computed tensors are allocated once per executable, with one buffer each and no pooling yet. Execution reuses those buffers and blocks at the end of the whole graph. All nodes run, including unused branches. The builder is move-only; compiled execution does not depend on its lifetime.

## Validation

The original host build compiled C++ but failed to link because the active linker did not recognize arm64e.x1 in the selected macOS 27 SDK. A separate Release build with the installed macOS 26.5 SDK succeeded. CTest passed 5/5: tensor_metadata, memory_cpu, operator_validation, activation_validation, graph_validation. The GPU runtime test and example also passed C++20 syntax checks with -Wall -Wextra -Wpedantic. This is not GPU compilation or execution evidence. `git diff --check` passed.

```sh
cmake -S . -B build/host-sdk26 -DFORGE_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
cmake --build build/host-sdk26 -j
ctest --test-dir build/host-sdk26 --output-on-failure --timeout 20
```

## Deferred NVIDIA validation

As requested, GPU testing remains deferred. Milestones 4 and 5: NOT YET GPU VALIDATED. Performance: NOT YET MEASURED.

```bash
%%bash
set -e
cd /content/forge
cmake -S . -B build/cuda -DFORGE_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build/cuda -j
ctest --test-dir build/cuda --output-on-failure --timeout 60
./build/cuda/forge_inference
```

Expected: nine tests, including new graph_validation and runtime_cuda. The runtime test compares batch sizes 1, 7, and 32 to an independent double-precision host reference, re-executes with changed inputs, and checks output allocation stability. No reference execution substitutes for GPU execution.

## Commit message

feat: complete graph runtime tests and inference example
