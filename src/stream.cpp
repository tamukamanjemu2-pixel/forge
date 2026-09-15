#include "forge/stream.h"
#include <stdexcept>
#ifdef FORGE_HAS_CUDA
#include "cuda_support.h"
#endif

namespace forge {
struct Stream::Impl {
    Device device;
#ifdef FORGE_HAS_CUDA
    cudaStream_t handle = nullptr;
    explicit Impl(Device target) : device(target) {
        detail::DeviceScope scope(target.index);
        detail::cuda_check(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }
    ~Impl() noexcept {
        try {
            detail::DeviceScope scope(device.index);
            detail::cuda_cleanup_check(cudaStreamDestroy(handle), "cudaStreamDestroy");
        } catch (const std::exception& error) {
            std::fprintf(stderr, "Forge CUDA stream cleanup error: %s\n", error.what());
        }
    }
#else
    explicit Impl(Device target) : device(target) {
        throw std::runtime_error("CUDA streams are unavailable: build with FORGE_ENABLE_CUDA=ON");
    }
#endif
};

Stream::Stream(Device device) {
    if (device.type != DeviceType::CUDA || device.index < 0)
        throw std::invalid_argument("Stream requires a valid CUDA device");
    impl_ = std::make_unique<Impl>(device);
}
Stream::~Stream() = default;
Stream::Stream(Stream&&) noexcept = default;
Stream& Stream::operator=(Stream&&) noexcept = default;
Device Stream::device() const {
    if (!impl_) throw std::logic_error("Stream was moved from");
    return impl_->device;
}
void* Stream::native_handle() const {
    (void)device();
#ifdef FORGE_HAS_CUDA
    return impl_->handle;
#else
    throw std::runtime_error("CUDA streams are unavailable");
#endif
}
void Stream::synchronize() const {
    const auto target = device();
#ifdef FORGE_HAS_CUDA
    detail::DeviceScope scope(target.index);
    detail::cuda_check(cudaStreamSynchronize(impl_->handle), "cudaStreamSynchronize");
#else
    (void)target;
    throw std::runtime_error("CUDA streams are unavailable");
#endif
}
} // namespace forge
