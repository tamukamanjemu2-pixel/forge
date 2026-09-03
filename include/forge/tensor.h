#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace forge {

    enum class DataType {
        Float32,
        Float16,
        Int32
    };

    enum class DeviceType {
        CPU,
        CUDA
    };

    struct Device {
        DeviceType type;
        int index;

        static Device cpu();
        static Device cuda(int index = 0);

        bool operator==(const Device& other) const = default;
    };

    class Tensor {
    public:
        Tensor(
            std::vector<std::int64_t> shape,
            DataType dtype,
            Device device,
            void* data = nullptr
        );

        const std::vector<std::int64_t>& shape() const noexcept;
        const std::vector<std::int64_t>& strides() const noexcept;

        std::size_t ndim() const noexcept;
        std::size_t numel() const noexcept;
        std::size_t nbytes() const noexcept;

        DataType dtype() const noexcept;
        Device device() const noexcept;

        void* data() noexcept;
        const void* data() const noexcept;

        void set_data(void* data) noexcept;

        bool is_contiguous() const noexcept;

        static std::size_t element_size(DataType dtype);

    private:
        static std::vector<std::int64_t> compute_contiguous_strides(
            const std::vector<std::int64_t>& shape
        );

        static std::size_t compute_numel(
            const std::vector<std::int64_t>& shape
        );

        std::vector<std::int64_t> shape_;
        std::vector<std::int64_t> strides_;

        DataType dtype_;
        Device device_;

        void* data_;
    };

} // namespace forge