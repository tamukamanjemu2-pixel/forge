#pragma once

#include "forge/tensor.h"

namespace forge {

// Owns storage independently of tensor metadata. No allocation occurs in Tensor.
// Views borrow this storage and must not outlive it. Allocations are uninitialized.
class Buffer {
public:
    explicit Buffer(std::size_t bytes, Device device = Device::cpu());
    ~Buffer() noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    std::size_t nbytes() const noexcept { return bytes_; }
    Device device() const noexcept { return device_; }

    // Checks capacity and element alignment. Offsets are measured in bytes.
    Tensor view(std::vector<std::int64_t> shape, DataType dtype,
                std::size_t offset = 0) &;
    Tensor view(std::vector<std::int64_t>, DataType, std::size_t = 0) && = delete;

private:
    void release() noexcept;
    void* data_ = nullptr;
    std::size_t bytes_ = 0;
    Device device_ = Device::cpu();
};

// Blocking, exact-shape/dtype copy. CPU overlap is supported; CUDA overlap and
// cross-GPU copies are rejected. Borrowed pointers must match their metadata.
// Callers must complete producers on unrelated streams before copying.
// This API is for setup/readback, not stream-aware runtime scheduling.
void copy_tensor(const Tensor& source, Tensor& destination);

} // namespace forge
