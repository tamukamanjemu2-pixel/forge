# Architecture

## Current foundation

`forge_core` contains CUDA-independent tensor metadata. `Tensor` is a non-owning view: it never allocates, copies, or frees the supplied storage. The caller must keep that storage alive and ensure that its capacity, alignment, and actual device match the metadata. `set_data` rebinds a borrowed pointer; copying a Tensor copies the view, not its data.

All currently representable layouts are contiguous. Scalars have an empty shape and one element. Non-scalar dimensions must be positive; zero-sized tensors are not supported. Construction validates dtype, device, element count, byte size, and representable strides. Metadata queries are allocation-free and non-throwing after successful construction. A CUDA device descriptor does not probe hardware, allowing host-only metadata validation.

`forge_ops` validates MatMul inputs and calls `forge_cuda`. The existing backend implements naive Float32 GEMM and vector addition. Allocation and synchronous transfers now use `Buffer` and `copy_tensor`, not Tensor. Callers own the Buffer lifetime. CUDA-enabled builds retain these targets and their existing public entry points.

## Intended evolution

Graph validation and dependency ordering will precede lifetime analysis and memory planning. Runtime-owned storage will back borrowed tensor views; kernel selection and stream-aware execution will sit behind the operator API. This avoids coupling Tensor construction to cudaMalloc/cudaFree.

The current backend still uses the default stream and requires callers to check execution errors and synchronize. These are known limitations for a later runtime/backend milestone, not completed features.

## Build boundary

`FORGE_ENABLE_CUDA=OFF` builds only the host core and its tests. CUDA stays enabled by default and requires the real toolkit. CTest registers metadata tests in both configurations and GPU MatMul tests only in CUDA configurations. There is no simulated CUDA backend.

## Storage ownership (milestone 2)

`Buffer` is a move-only RAII allocation. CPU allocations use 64-byte aligned C++ allocation; CUDA allocations use cudaMalloc on the requested device. Moving ownership preserves existing views into the transferred allocation. Move assignment destroys the destination's previous allocation and invalidates its old views. A moved-from Buffer has null storage and zero capacity and cannot create views. Temporary buffers cannot create views. Storage is uninitialized.

A Buffer may provide several checked, element-aligned Tensor views using byte offsets. The Buffer must outlive all views and outstanding GPU work. Direct Tensor construction remains available for borrowed external storage, whose capacity and actual device cannot be checked from metadata alone.

`copy_tensor` validates shape, dtype, layout, and storage presence. CPU copies use memmove. CUDA copies select and restore the requested device, reject overlap and cross-GPU transfers, and synchronize the selected device before returning. The barrier is deliberately confined to the synchronous setup/readback API; it is not a future operator scheduling policy. Copies from work running on unrelated streams require callers to establish producer completion before invoking this API.

CUDA errors on normal calls throw contextual runtime errors. Destructors report cleanup errors to stderr without throwing; a failed cleanup can leave a device allocation unreleased. The shared internal CUDA helper is in src/cuda_support.h. Existing microbenchmarks still retain their own error handling pending the backend milestone.

This is an allocation foundation, not a pool: allocations remain explicit and do not yet have reuse statistics or graph lifetime planning. Host-only builds reject CUDA allocation/copy requests instead of emulating them.
