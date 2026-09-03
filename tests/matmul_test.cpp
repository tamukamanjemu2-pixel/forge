#include "forge/ops/matmul.h"
#include "forge/tensor.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        const cudaError_t error = (call);                     \
        if (error != cudaSuccess) {                           \
            std::cerr                                         \
                << "CUDA error: "                             \
                << cudaGetErrorString(error)                  \
                << " at "                                     \
                << __FILE__                                   \
                << ":"                                        \
                << __LINE__                                   \
                << '\n';                                      \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)

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

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_output = nullptr;

    CUDA_CHECK(
        cudaMalloc(
            &d_a,
            h_a.size() * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_b,
            h_b.size() * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMalloc(
            &d_output,
            h_output.size() * sizeof(float)
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_a,
            h_a.data(),
            h_a.size() * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    CUDA_CHECK(
        cudaMemcpy(
            d_b,
            h_b.data(),
            h_b.size() * sizeof(float),
            cudaMemcpyHostToDevice
        )
    );

    Tensor a(
        {2, 3},
        DataType::Float32,
        Device::cuda(),
        d_a
    );

    Tensor b(
        {3, 2},
        DataType::Float32,
        Device::cuda(),
        d_b
    );

    Tensor output(
        {2, 2},
        DataType::Float32,
        Device::cuda(),
        d_output
    );

    forge::matmul(
        a,
        b,
        output
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(
        cudaMemcpy(
            h_output.data(),
            d_output,
            h_output.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        )
    );

    constexpr float tolerance = 1e-5f;

    for (
        std::size_t i = 0;
        i < expected.size();
        ++i
    ) {
        if (
            std::abs(
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

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_output));

    std::cout
        << "MatMul tests passed\n";

    return EXIT_SUCCESS;
}