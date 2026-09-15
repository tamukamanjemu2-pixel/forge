#include "forge/memory.h"

#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>

#ifdef FORGE_HAS_CUDA
#include "cuda_support.h"
#endif

namespace forge {
namespace {
constexpr auto cpu_alignment = std::align_val_t{64};

void validate_device(Device device) {
    if ((device.type != DeviceType::CPU && device.type != DeviceType::CUDA) ||
        device.index < 0 || (device.type == DeviceType::CPU && device.index != 0)) {
        throw std::invalid_argument("Invalid buffer device");
    }
}
}

Buffer::Buffer(std::size_t bytes, Device device) : bytes_(bytes), device_(device) {
    validate_device(device);
    if (bytes == 0) throw std::invalid_argument("Buffer size must be positive");
    if (device.type == DeviceType::CPU) {
        data_ = ::operator new(bytes, cpu_alignment);
    } else {
#ifdef FORGE_HAS_CUDA
        detail::DeviceScope scope(device.index);
        detail::cuda_check(cudaMalloc(&data_, bytes), "cudaMalloc");
#else
        throw std::runtime_error("CUDA storage is unavailable: build with FORGE_ENABLE_CUDA=ON");
#endif
    }
}

Buffer::~Buffer() noexcept { release(); }

Buffer::Buffer(Buffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      bytes_(std::exchange(other.bytes_, 0)), device_(other.device_) {}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        release();
        data_ = std::exchange(other.data_, nullptr);
        bytes_ = std::exchange(other.bytes_, 0);
        device_ = other.device_;
    }
    return *this;
}

void Buffer::release() noexcept {
    if (!data_) return;
    if (device_.type == DeviceType::CPU) {
        ::operator delete(data_, cpu_alignment);
    } else {
#ifdef FORGE_HAS_CUDA
        // Retain the original device even if the caller changes the current device.
        int previous = 0;
        const auto queried = cudaGetDevice(&previous);
        detail::cuda_cleanup_check(queried, "cudaGetDevice before free");
        const auto selected = cudaSetDevice(device_.index);
        detail::cuda_cleanup_check(selected, "cudaSetDevice before free");
        if (selected == cudaSuccess) detail::cuda_cleanup_check(cudaFree(data_), "cudaFree");
        if (queried == cudaSuccess && previous != device_.index)
            detail::cuda_cleanup_check(cudaSetDevice(previous), "restore CUDA device after free");
#endif
    }
    data_ = nullptr;
    bytes_ = 0;
}

Tensor Buffer::view(std::vector<std::int64_t> shape, DataType dtype, std::size_t offset) & {
    Tensor tensor(std::move(shape), dtype, device_);
    if (!data_) throw std::logic_error("Cannot create a view from a moved-from buffer");
    if (offset % Tensor::element_size(dtype) != 0)
        throw std::invalid_argument("Tensor view offset is not element-aligned");
    if (offset > bytes_ || tensor.nbytes() > bytes_ - offset)
        throw std::out_of_range("Tensor view exceeds buffer capacity");
    tensor.set_data(static_cast<std::byte*>(data_) + offset);
    return tensor;
}

void copy_tensor(const Tensor& source, Tensor& destination) {
    if (source.shape() != destination.shape() || source.dtype() != destination.dtype())
        throw std::invalid_argument("Tensor copy requires matching shapes and dtypes");
    if (!source.data() || !destination.data())
        throw std::invalid_argument("Tensor copy requires storage");
    if (!source.is_contiguous() || !destination.is_contiguous())
        throw std::invalid_argument("Tensor copy requires contiguous tensors");
    const auto src = source.device();
    const auto dst = destination.device();
    if (src.type == DeviceType::CPU && dst.type == DeviceType::CPU) {
        std::memmove(destination.data(), source.data(), source.nbytes());
        return;
    }
#ifdef FORGE_HAS_CUDA
    if (src.type == DeviceType::CUDA && dst.type == DeviceType::CUDA && src != dst)
        throw std::invalid_argument("Cross-GPU tensor copies are not supported");
    if (src == dst) {
        const auto a = reinterpret_cast<std::uintptr_t>(source.data());
        const auto b = reinterpret_cast<std::uintptr_t>(destination.data());
        const auto distance = a > b ? a - b : b - a;
        if (distance < source.nbytes())
            throw std::invalid_argument("Overlapping CUDA tensor copies are not supported");
    }
    detail::DeviceScope scope(src.type == DeviceType::CUDA ? src.index : dst.index);
    const auto kind = src.type == DeviceType::CPU ? cudaMemcpyHostToDevice :
                      dst.type == DeviceType::CPU ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
    detail::cuda_check(cudaMemcpy(destination.data(), source.data(), source.nbytes(), kind), "cudaMemcpy");
    // cudaMemcpy can return before device-to-device work completes. This setup API
    // deliberately guarantees completion; a future stream API will avoid this barrier.
    detail::cuda_check(cudaDeviceSynchronize(), "synchronize tensor copy");
#else
    throw std::runtime_error("CUDA copies are unavailable: build with FORGE_ENABLE_CUDA=ON");
#endif
}
} // namespace forge
