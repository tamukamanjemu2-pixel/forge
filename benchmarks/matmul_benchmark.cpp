#include "../src/cuda_support.h"
#include "forge/ops/matmul.h"
#include "forge/tensor.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

struct BenchmarkResult {
    int size;
    float kernel_ms;
    double gflops;
};

BenchmarkResult run_benchmark(
    int size,
    int iterations
) {
    using forge::DataType;
    using forge::Device;
    using forge::Tensor;

    const std::size_t elements =
        static_cast<std::size_t>(size) * size;

    const std::size_t bytes =
        elements * sizeof(float);

    std::vector<float> h_a(elements, 1.0f);
    std::vector<float> h_b(elements, 1.0f);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    FORGE_CUDA_CHECK(cudaMalloc(&d_a, bytes));
    FORGE_CUDA_CHECK(cudaMalloc(&d_b, bytes));
    FORGE_CUDA_CHECK(cudaMalloc(&d_c, bytes));

    FORGE_CUDA_CHECK(
        cudaMemcpy(
            d_a,
            h_a.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );

    FORGE_CUDA_CHECK(
        cudaMemcpy(
            d_b,
            h_b.data(),
            bytes,
            cudaMemcpyHostToDevice
        )
    );

    Tensor a(
        {size, size},
        DataType::Float32,
        Device::cuda(),
        d_a
    );

    Tensor b(
        {size, size},
        DataType::Float32,
        Device::cuda(),
        d_b
    );

    Tensor c(
        {size, size},
        DataType::Float32,
        Device::cuda(),
        d_c
    );

    // Warmup
    for (int i = 0; i < 5; ++i) {
        forge::matmul(a, b, c);
    }

    FORGE_CUDA_CHECK(cudaGetLastError());
    FORGE_CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;

    FORGE_CUDA_CHECK(cudaEventCreate(&start));
    FORGE_CUDA_CHECK(cudaEventCreate(&stop));

    FORGE_CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        forge::matmul(a, b, c);
    }

    FORGE_CUDA_CHECK(cudaEventRecord(stop));
    FORGE_CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    FORGE_CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );

    const float kernel_ms =
        total_ms / static_cast<float>(iterations);

    //
    // Matrix multiplication:
    //
    // C[M,N] = A[M,K] * B[K,N]
    //
    // Approximate floating-point operations:
    //
    // 2 * M * N * K
    //
    const double operations =
        2.0 *
        static_cast<double>(size) *
        static_cast<double>(size) *
        static_cast<double>(size);

    const double gflops =
        operations /
        (kernel_ms / 1000.0) /
        1e9;

    FORGE_CUDA_CHECK(cudaEventDestroy(start));
    FORGE_CUDA_CHECK(cudaEventDestroy(stop));

    FORGE_CUDA_CHECK(cudaFree(d_a));
    FORGE_CUDA_CHECK(cudaFree(d_b));
    FORGE_CUDA_CHECK(cudaFree(d_c));

    return {
        size,
        kernel_ms,
        gflops
    };
}

int main() {
    const std::vector<int> sizes = {
        256,
        512,
        1024,
        2048
    };

    constexpr int iterations = 20;

    cudaDeviceProp properties{};

    FORGE_CUDA_CHECK(
        cudaGetDeviceProperties(
            &properties,
            0
        )
    );

    std::cout
        << "Forge Naive MatMul Benchmark\n\n";

    std::cout
        << "GPU: "
        << properties.name
        << '\n';

    std::cout
        << "Compute capability: "
        << properties.major
        << "."
        << properties.minor
        << '\n';

    std::cout
        << "Iterations: "
        << iterations
        << "\n\n";

    std::cout
        << std::left
        << std::setw(12) << "Size"
        << std::setw(18) << "Kernel (ms)"
        << std::setw(18) << "GFLOP/s"
        << '\n';

    std::cout
        << std::string(48, '-')
        << '\n';

    for (const int size : sizes) {
        const auto result =
            run_benchmark(
                size,
                iterations
            );

        std::cout
            << std::left
            << std::setw(12)
            << result.size
            << std::setw(18)
            << std::fixed
            << std::setprecision(4)
            << result.kernel_ms
            << std::setw(18)
            << std::fixed
            << std::setprecision(2)
            << result.gflops
            << '\n';
    }

    return 0;
}