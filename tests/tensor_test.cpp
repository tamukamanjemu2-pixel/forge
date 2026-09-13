#include "forge/tensor.h"

#include <limits>
#include <string>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(bool condition) {
    if (!condition) throw std::runtime_error("Tensor test check failed");
}

template<class Error, class Function>
void expect_error(Function function, const char* name) {
    try { function(); }
    catch (const Error&) { return; }
    throw std::runtime_error(std::string("Expected error: ") + name);
}
}

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

        check(tensor.ndim() == 2);
        check(tensor.numel() == 32 * 1024);
        check(
            tensor.nbytes() ==
            32 * 1024 * sizeof(float)
        );

        check(
            tensor.shape() ==
            std::vector<std::int64_t>(
                {32, 1024}
            )
        );

        check(
            tensor.strides() ==
            std::vector<std::int64_t>(
                {1024, 1}
            )
        );

        check(
            tensor.device().type ==
            DeviceType::CUDA
        );

        check(tensor.device().index == 0);
        check(tensor.data() == nullptr);
        check(tensor.is_contiguous());
    }

    {
        float storage[16]{};

        Tensor tensor(
            {4, 4},
            DataType::Float32,
            Device::cpu(),
            storage
        );

        check(tensor.data() == storage);
        check(tensor.numel() == 16);
        check(tensor.nbytes() == 64);
    }

    {
        Tensor scalar(
            {},
            DataType::Float32,
            Device::cpu()
        );

        check(scalar.ndim() == 0);
        check(scalar.numel() == 1);
        check(scalar.nbytes() == 4);
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

        check(threw);
    }

    for (const auto dtype : {DataType::Float32, DataType::Float16, DataType::Int32}) {
        Tensor tensor({2, 3, 4}, dtype, Device::cpu());
        check(tensor.strides() == std::vector<std::int64_t>({12, 4, 1}));
        check(tensor.nbytes() == 24 * Tensor::element_size(dtype));
        int storage = 0;
        tensor.set_data(&storage);
        check(tensor.data() == &storage);
        const Tensor copy = tensor;
        check(copy.data() == tensor.data());
        check(copy.numel() == 24);
    }
    expect_error<std::invalid_argument>([] {
        Tensor t({2, -3}, DataType::Float32, Device::cpu());
    }, "negative dimension");
    expect_error<std::invalid_argument>([] {
        Tensor t({}, static_cast<DataType>(99), Device::cpu());
    }, "invalid dtype at construction");
    expect_error<std::invalid_argument>([] { Device::cuda(-1); }, "negative CUDA index");
    for (const auto device : {Device{DeviceType::CPU, 1}, Device{DeviceType::CUDA, -1},
                              Device{static_cast<DeviceType>(99), 0}}) {
        expect_error<std::invalid_argument>([&] {
            Tensor t({1}, DataType::Float32, device);
        }, "invalid aggregate device");
    }
    constexpr auto largest = std::numeric_limits<std::int64_t>::max();
    expect_error<std::overflow_error>([&] {
        Tensor t({largest, 3}, DataType::Float32, Device::cpu());
    }, "element count overflow");
    expect_error<std::overflow_error>([&] {
        Tensor t({largest}, DataType::Float32, Device::cpu());
    }, "byte size overflow");
    expect_error<std::overflow_error>([&] {
        Tensor t({1, largest, largest}, DataType::Float16, Device::cpu());
    }, "oversized trailing dimensions");

    std::cout
        << "Tensor tests passed\n";

    return 0;
}
