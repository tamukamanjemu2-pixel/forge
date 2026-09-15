# Architecture

## Current foundation

`forge_core` contains CUDA-independent tensor metadata. `Tensor` is a non-owning view: it never allocates, copies, or frees the supplied storage. The caller must keep that storage alive and ensure that its capacity, alignment, and actual device match the metadata. `set_data` rebinds a borrowed pointer; copying a Tensor copies the view, not its data.

All currently representable layouts are contiguous. Scalars have an empty shape and one element. Non-scalar dimensions must be positive; zero-sized tensors are not supported. Construction validates dtype, device, element count, byte size, and representable strides. Metadata queries are allocation-free and non-throwing after successful construction. A CUDA device descriptor does not probe hardware, allowing host-only metadata validation.

`forge_ops` validates MatMul inputs and calls `forge_cuda`. The existing backend implements naive Float32 GEMM and vector addition. Allocation and synchronous transfers now use `Buffer` and `copy_tensor`, not Tensor. Callers own the Buffer lifetime. CUDA-enabled builds retain these targets and their existing public entry points.

## Intended evolution

Graph validation and dependency ordering will precede lifetime analysis and memory planning. Runtime-owned storage will back borrowed tensor views; kernel selection and stream-aware execution will sit behind the operator API. This avoids coupling Tensor construction to cudaMalloc/cudaFree.

MatMul supports both default-stream and explicit-stream dispatch. It checks launch errors without synchronizing; callers establish completion and detect asynchronous errors through Stream::synchronize() or the blocking copy API. The vector-add microbenchmark still uses its default-stream entry point.

## Build boundary

`FORGE_ENABLE_CUDA=OFF` builds the host core, operator validation, and host tests. Valid CUDA operator calls fail explicitly in this configuration. CUDA stays enabled by default and requires the real toolkit. CTest registers metadata tests in both configurations and GPU MatMul tests only in CUDA configurations. There is no simulated CUDA backend.

## Storage ownership (milestone 2)

`Buffer` is a move-only RAII allocation. CPU allocations use 64-byte aligned C++ allocation; CUDA allocations use cudaMalloc on the requested device. Moving ownership preserves existing views into the transferred allocation. Move assignment destroys the destination's previous allocation and invalidates its old views. A moved-from Buffer has null storage and zero capacity and cannot create views. Temporary buffers cannot create views. Storage is uninitialized.

A Buffer may provide several checked, element-aligned Tensor views using byte offsets. The Buffer must outlive all views and outstanding GPU work. Direct Tensor construction remains available for borrowed external storage, whose capacity and actual device cannot be checked from metadata alone.

`copy_tensor` validates shape, dtype, layout, and storage presence. CPU copies use memmove. CUDA copies select and restore the requested device, reject overlap and cross-GPU transfers, and synchronize the selected device before returning. The barrier is deliberately confined to the synchronous setup/readback API; it is not a future operator scheduling policy. Copies from work running on unrelated streams require callers to establish producer completion before invoking this API.

CUDA errors on normal calls throw contextual runtime errors. Destructors report cleanup errors to stderr without throwing; a failed cleanup can leave a device allocation unreleased. The shared internal CUDA helper is in src/cuda_support.h. Both microbenchmarks and kernel launchers now use this shared error handling; benchmark timing methodology is otherwise retained.

This is an allocation foundation, not a pool: allocations remain explicit and do not yet have reuse statistics or graph lifetime planning. Host-only builds reject CUDA allocation/copy requests instead of emulating them.

## Stream-aware operators (milestone 3)

`Stream` owns a non-blocking CUDA stream through a private implementation, keeping CUDA headers out of the public core headers. It is movable and not copyable. Moved-from objects reject use. Its device is immutable; synchronize, creation, and destruction select/restore that device. The native handle is borrowed interoperability access and must not be destroyed externally. Destruction does not promise work completion; storage lifetime remains the caller's responsibility.

MatMul validates rank, Float32 dtype, CUDA device agreement, contiguous layout, inner/output dimensions, storage, integer dimension limits, and output/input overlap before dispatch. Read-only input/input overlap is allowed. Explicit streams must match the tensor device. The old API delegates to the same validation and uses the default stream on that device.

The naive CUDA kernel retains one output per thread. Index multiplication now uses size_t and launch dimensions use overflow-safe ceiling division. The launcher checks device grid limits before enqueueing and checks the CUDA launch status immediately afterward. Grid attribute queries currently occur per launch; no host overhead or kernel speedup is claimed. Rebenchmark before comparing performance with the historical baseline.

No implicit per-operator synchronization was added. Event-based dependencies, asynchronous copy APIs, pooling, kernel selection, and graph scheduling remain future milestones.

## Activations (milestone 4)

ReLU and Softmax share runtime-side validation and dispatch in src/ops/activations.cpp. Public headers expose separate operators while backend launchers live in cuda/activations.h. Both support the same explicit/default stream policy as MatMul, restore the caller's device, check launch failures, and avoid synchronization inside the operator.

ReLU uses a grid-stride elementwise kernel, allowing arbitrary contiguous shape, including scalars. Negative values become zero; NaNs propagate. Softmax flattens leading dimensions into rows and normalizes the final axis. One 256-thread block reduces each row's maximum and exponential sum with shared memory, including barriers before shared storage reuse. Blocks iterate over additional rows when row count exceeds the grid cap. Column loops support non-power-of-two widths and widths larger than a block. No scratch allocation is performed; the output temporarily holds unnormalized exponentials.

Softmax's numeric contract covers finite Float32 logits. Subtracting the maximum prevents positive exponential overflow. NaN/infinite inputs are outside its specified probability semantics. Input and output overlap, including exact aliasing, is rejected for both operators. Float16, configurable axes, in-place execution, fusion and measured tuning remain future work.
