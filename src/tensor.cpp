#include "forge/tensor.h"

#include <limits>
#include <stdexcept>
#include <utility>

namespace forge {

Device Device::cpu() {
    return {
        .type = DeviceType::CPU,
        .index = 0
    };
}

Device Device::cuda(int index) {
    if (index < 0) {
        throw std::invalid_argument(
            "CUDA device index cannot be negative"
        );
    }

    return {
        .type = DeviceType::CUDA,
        .index = index
    };
}

Tensor::Tensor(
    std::vector<std::int64_t> shape,
    DataType dtype,
    Device device,
    void* data
)
    : shape_(std::move(shape)),
      strides_(compute_contiguous_strides(shape_)),
      dtype_(dtype),
      device_(device),
      data_(data) {

    // Force validation during construction.
    (void)compute_numel(shape_);
}

const std::vector<std::int64_t>&
Tensor::shape() const noexcept {
    return shape_;
}

const std::vector<std::int64_t>&
Tensor::strides() const noexcept {
    return strides_;
}

std::size_t Tensor::ndim() const noexcept {
    return shape_.size();
}

std::size_t Tensor::numel() const noexcept {
    return compute_numel(shape_);
}

std::size_t Tensor::nbytes() const noexcept {
    return numel() * element_size(dtype_);
}

DataType Tensor::dtype() const noexcept {
    return dtype_;
}

Device Tensor::device() const noexcept {
    return device_;
}

void* Tensor::data() noexcept {
    return data_;
}

const void* Tensor::data() const noexcept {
    return data_;
}

void Tensor::set_data(void* data) noexcept {
    data_ = data;
}

bool Tensor::is_contiguous() const noexcept {
    return strides_ == compute_contiguous_strides(shape_);
}

std::size_t Tensor::element_size(DataType dtype) {
    switch (dtype) {
        case DataType::Float32:
            return 4;

        case DataType::Float16:
            return 2;

        case DataType::Int32:
            return 4;
    }

    throw std::invalid_argument("Unsupported Forge data type");
}

std::vector<std::int64_t>
Tensor::compute_contiguous_strides(
    const std::vector<std::int64_t>& shape
) {
    std::vector<std::int64_t> strides(
        shape.size(),
        1
    );

    if (shape.empty()) {
        return strides;
    }

    for (std::size_t i = shape.size(); i-- > 1;) {
        strides[i - 1] =
            strides[i] * shape[i];
    }

    return strides;
}

std::size_t Tensor::compute_numel(
    const std::vector<std::int64_t>& shape
) {
    if (shape.empty()) {
        // Scalar tensor.
        return 1;
    }

    std::size_t elements = 1;

    for (const auto dimension : shape) {
        if (dimension <= 0) {
            throw std::invalid_argument(
                "Tensor dimensions must be positive"
            );
        }

        const auto dim =
            static_cast<std::size_t>(dimension);

        if (
            elements >
            std::numeric_limits<std::size_t>::max() / dim
        ) {
            throw std::overflow_error(
                "Tensor element count overflow"
            );
        }

        elements *= dim;
    }

    return elements;
}

} // namespace forge