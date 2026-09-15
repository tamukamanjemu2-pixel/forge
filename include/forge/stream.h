#pragma once

#include "forge/tensor.h"
#include <memory>

namespace forge {
// Owns a non-blocking CUDA stream. Enqueue operations do not synchronize.
// Keep all borrowed tensors/storage alive until synchronize() completes.
class Stream {
public:
    explicit Stream(Device device = Device::cuda());
    ~Stream();
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    Stream(Stream&&) noexcept;
    Stream& operator=(Stream&&) noexcept;

    Device device() const;
    void synchronize() const;
    // CUDA interoperability only: borrowed cudaStream_t, represented opaquely to
    // keep CUDA headers out of the public core API. Do not destroy this handle.
    void* native_handle() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace forge
