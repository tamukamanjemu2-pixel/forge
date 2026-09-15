#include "forge/ops/matmul.h"
#include "forge/tensor.h"
#include "forge/memory.h"
#include "../src/cuda_support.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
    using forge::DataType;
    using forge::Device;
    using forge::Tensor;

    //
    // A = 2 x 3
    //
    // [1 2 3]
    // [4 5 6]
    //
    const std::vector<float> h_a = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };

    //
    // B = 3 x 2
    //
    // [ 7  8]
    // [ 9 10]
    // [11 12]
    //
    const std::vector<float> h_b = {
         7.0f,  8.0f,
         9.0f, 10.0f,
        11.0f, 12.0f
    };

    //
    // Expected:
    //
    // [ 58  64]
    // [139 154]
    //
    const std::vector<float> expected = {
        58.0f, 64.0f,
        139.0f, 154.0f
    };

    std::vector<float> h_output(4);

    forge::Buffer a_storage(h_a.size() * sizeof(float), Device::cuda());
    forge::Buffer b_storage(h_b.size() * sizeof(float), Device::cuda());
    forge::Buffer output_storage(h_output.size() * sizeof(float), Device::cuda());
    auto a = a_storage.view({2, 3}, DataType::Float32);
    auto b = b_storage.view({3, 2}, DataType::Float32);
    auto output = output_storage.view({2, 2}, DataType::Float32);
    // Borrowed host inputs remain alive throughout upload.
    auto host_a_values = h_a;
    auto host_b_values = h_b;
    Tensor host_a({2, 3}, DataType::Float32, Device::cpu(), host_a_values.data());
    Tensor host_b({3, 2}, DataType::Float32, Device::cpu(), host_b_values.data());
    Tensor host_output({2, 2}, DataType::Float32, Device::cpu(), h_output.data());
    forge::copy_tensor(host_a, a);
    forge::copy_tensor(host_b, b);

    forge::matmul(
        a,
        b,
        output
    );

    forge::detail::cuda_check(cudaGetLastError(), "MatMul launch");
    forge::copy_tensor(output, host_output);

    constexpr float tolerance = 1e-5f;

    for (
        std::size_t i = 0;
        i < expected.size();
        ++i
    ) {
        if (
            !std::isfinite(h_output[i]) || std::abs(
                h_output[i] - expected[i]
            ) > tolerance
        ) {
            std::cerr
                << "MatMul mismatch at index "
                << i
                << ": expected "
                << expected[i]
                << ", got "
                << h_output[i]
                << '\n';

            return EXIT_FAILURE;
        }
    }

    std::cout
        << "MatMul tests passed\n";

    return EXIT_SUCCESS;
}