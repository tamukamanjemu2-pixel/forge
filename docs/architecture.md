# Architecture

## Current foundation

`forge_core` contains CUDA-independent tensor metadata. `Tensor` is a non-owning view: it never allocates, copies, or frees the supplied storage. The caller must keep that storage alive and ensure that its capacity, alignment, and actual device match the metadata. `set_data` rebinds a borrowed pointer; copying a Tensor copies the view, not its data.

All currently representable layouts are contiguous. Scalars have an empty shape and one element. Non-scalar dimensions must be positive; zero-sized tensors are not supported. Construction validates dtype, device, element count, byte size, and representable strides. Metadata queries are allocation-free and non-throwing after successful construction. A CUDA device descriptor does not probe hardware, allowing host-only metadata validation.

`forge_ops` validates MatMul inputs and calls `forge_cuda`. The existing backend implements naive Float32 GEMM and vector addition. Allocation and transfers currently belong to callers/tests, not Tensor. CUDA-enabled builds retain these targets and their existing public entry points.

## Intended evolution

Graph validation and dependency ordering will precede lifetime analysis and memory planning. Runtime-owned storage will back borrowed tensor views; kernel selection and stream-aware execution will sit behind the operator API. This avoids coupling Tensor construction to cudaMalloc/cudaFree.

The current backend still uses the default stream and requires callers to check execution errors and synchronize. These are known limitations for a later runtime/backend milestone, not completed features.

## Build boundary

`FORGE_ENABLE_CUDA=OFF` builds only the host core and its tests. CUDA stays enabled by default and requires the real toolkit. CTest registers metadata tests in both configurations and GPU MatMul tests only in CUDA configurations. There is no simulated CUDA backend.
