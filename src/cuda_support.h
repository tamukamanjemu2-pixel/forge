#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace forge::detail {
inline void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

// Cleanup cannot throw during stack unwinding. Report every failed CUDA call.
inline void cuda_cleanup_check(cudaError_t result, const char* operation) noexcept {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "Forge CUDA cleanup error: %s: %s\n", operation,
                     cudaGetErrorString(result));
    }
}

class DeviceScope {
public:
    explicit DeviceScope(int target) {
        cuda_check(cudaGetDevice(&previous_), "cudaGetDevice");
        changed_ = previous_ != target;
        if (changed_) cuda_check(cudaSetDevice(target), "cudaSetDevice");
    }
    ~DeviceScope() noexcept {
        if (changed_) cuda_cleanup_check(cudaSetDevice(previous_), "restore CUDA device");
    }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
private:
    int previous_ = 0;
    bool changed_ = false;
};
} // namespace forge::detail
