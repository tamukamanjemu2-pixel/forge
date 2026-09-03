#include "forge/tensor.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    using forge::DataType;
    using forge::Device;
    using forge::DeviceType;
    using forge::Tensor;

    {
        Tensor tensor(
            {32, 1024},
            DataType::Float32,
            Device::cuda(0)
        );

        assert(tensor.ndim() == 2);
        assert(tensor.numel() == 32 * 1024);
        assert(
            tensor.nbytes() ==
            32 * 1024 * sizeof(float)
        );

        assert(
            tensor.shape() ==
            std::vector<std::int64_t>(
                {32, 1024}
            )
        );

        assert(
            tensor.strides() ==
            std::vector<std::int64_t>(
                {1024, 1}
            )
        );

        assert(
            tensor.device().type ==
            DeviceType::CUDA
        );

        assert(tensor.device().index == 0);
        assert(tensor.data() == nullptr);
        assert(tensor.is_contiguous());
    }

    {
        float storage[16]{};

        Tensor tensor(
            {4, 4},
            DataType::Float32,
            Device::cpu(),
            storage
        );

        assert(tensor.data() == storage);
        assert(tensor.numel() == 16);
        assert(tensor.nbytes() == 64);
    }

    {
        Tensor scalar(
            {},
            DataType::Float32,
            Device::cpu()
        );

        assert(scalar.ndim() == 0);
        assert(scalar.numel() == 1);
        assert(scalar.nbytes() == 4);
    }

    {
        bool threw = false;

        try {
            Tensor invalid(
                {32, 0},
                DataType::Float32,
                Device::cpu()
            );
        } catch (const std::invalid_argument&) {
            threw = true;
        }

        assert(threw);
    }

    std::cout
        << "Tensor tests passed\n";

    return 0;
}