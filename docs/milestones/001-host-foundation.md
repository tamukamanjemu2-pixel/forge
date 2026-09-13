# Milestone 1: host-buildable, validated tensor foundation

## Objective

Make the existing core independently buildable on Apple Silicon and establish safe metadata invariants before adding runtime-managed storage or graph planning. Preserve existing CUDA kernels and the MatMul API.

## Changes

- CMakeLists.txt: explicit CUDA build option, CTest registration, preserved GPU targets, architecture selected before CUDA language initialization.
- include/forge/tensor.h and src/tensor.cpp: explicit borrowed-storage contract, cached validated sizes, device/dtype validation, checked stride and byte arithmetic.
- tests/tensor_test.cpp: Release-active checks, dtype/scalar/stride/storage tests, invalid device/shape/type cases, overflow regression cases.
- README.md and docs/architecture.md: accurate local implementation status, build boundaries, ownership and future architecture.

## Decisions

CUDA stays enabled by default to preserve the existing GPU workflow. Mac builds explicitly disable it. Empty shapes remain scalars, while zero dimensions remain invalid for compatibility. No allocating storage abstraction is introduced prematurely.

## Validation

AppleClang 17 on Apple Silicon: Release configure/build succeeded; CTest tensor_metadata passed (1/1). Tests use runtime checks rather than assertions removed by NDEBUG.

Additional AddressSanitizer/UndefinedBehaviorSanitizer build succeeded, but its test timed out after 15 seconds under this session sandbox (which also reported `/bin/ps: Operation not permitted`). Sanitizer runtime validation is inconclusive; this is not recorded as a pass. `git diff --check` passed.

GPU compilation, correctness, and benchmark execution: NOT YET RUN. No GPU performance claim is made. Run the NVIDIA commands in README.md on the T4 before relying on CUDA build validation.

## Commit message

build: enable host-only validation and harden tensor metadata
