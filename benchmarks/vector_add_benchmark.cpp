#include "vector_add.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = call;                                          \
        if (error != cudaSuccess) {                                        \
            std::cerr << "CUDA error: "                                    \
                      << cudaGetErrorString(error)                          \
                      << " at " << __FILE__                                 \
                      << ":" << __LINE__ << std::endl;                      \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

struct BenchmarkResult {
    int n;
    int threads_per_block;
    float kernel_ms;
    double bandwidth_gbps;
    float max_error;
};

BenchmarkResult run_benchmark(
    int n,
    int threads_per_block,
    int iterations
) {
    const std::size_t bytes =
        static_cast<std::size_t>(n) * sizeof(float);

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);

    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i) * 0.5f;
        h_b[i] = static_cast<float>(i) * 0.25f;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_a,
        h_a.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_b,
        h_b.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    // Warmup
    for (int i = 0; i < 10; ++i) {
        launch_vector_add(
            d_a,
            d_b,
            d_c,
            n,
            threads_per_block
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        launch_vector_add(
            d_a,
            d_b,
            d_c,
            n,
            threads_per_block
        );
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;

    CUDA_CHECK(
        cudaEventElapsedTime(
            &total_ms,
            start,
            stop
        )
    );

    const float kernel_ms =
        total_ms / static_cast<float>(iterations);

    CUDA_CHECK(cudaMemcpy(
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost
    ));

    float max_error = 0.0f;

    for (int i = 0; i < n; ++i) {
        const float expected =
            h_a[i] + h_b[i];

        max_error =
            std::max(
                max_error,
                std::abs(h_c[i] - expected)
            );
    }

    //
    // Vector add:
    //
    // read A  -> bytes
    // read B  -> bytes
    // write C -> bytes
    //
    // total traffic = 3 * bytes
    //
    const double transferred_bytes =
        3.0 * static_cast<double>(bytes);

    const double bandwidth_gbps =
        transferred_bytes /
        (kernel_ms / 1000.0) /
        1e9;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return {
        n,
        threads_per_block,
        kernel_ms,
        bandwidth_gbps,
        max_error
    };
}

int main() {
    const std::vector<int> problem_sizes = {
        1 << 20,
        1 << 22,
        1 << 24,
        1 << 26
    };

    const std::vector<int> block_sizes = {
        64,
        128,
        256,
        512,
        1024
    };

    constexpr int iterations = 100;

    int device = 0;

    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp properties{};

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &properties,
            device
        )
    );

    std::cout
        << "Forge Vector Add Benchmark\n\n";

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
        << std::setw(14) << "N"
        << std::setw(12) << "Threads"
        << std::setw(16) << "Kernel (ms)"
        << std::setw(20) << "Bandwidth (GB/s)"
        << std::setw(12) << "Max Error"
        << '\n';

    std::cout
        << std::string(74, '-')
        << '\n';

    for (const int n : problem_sizes) {
        for (const int threads : block_sizes) {

            if (threads >
                properties.maxThreadsPerBlock) {
                continue;
            }

            const BenchmarkResult result =
                run_benchmark(
                    n,
                    threads,
                    iterations
                );

            std::cout
                << std::left
                << std::setw(14)
                << result.n
                << std::setw(12)
                << result.threads_per_block
                << std::setw(16)
                << std::fixed
                << std::setprecision(4)
                << result.kernel_ms
                << std::setw(20)
                << std::fixed
                << std::setprecision(2)
                << result.bandwidth_gbps
                << std::setw(12)
                << result.max_error
                << '\n';
        }

        std::cout << '\n';
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}