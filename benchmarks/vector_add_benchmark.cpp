#include "vector_add.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call)                                             \
    do {                                                             \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)   \
                      << " at " << __FILE__ << ":" << __LINE__      \
                      << '\n';                                      \
            std::exit(EXIT_FAILURE);                                \
        }                                                            \
    } while (0)

int main() {
    constexpr int N = 1 << 24;

    const size_t bytes = static_cast<size_t>(N) * sizeof(float);

    std::vector<float> h_a(N);
    std::vector<float> h_b(N);
    std::vector<float> h_c(N);

    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // Warm-up
    launch_vector_add(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr int iterations = 100;

    cudaEvent_t start;
    cudaEvent_t stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        launch_vector_add(d_a, d_b, d_c, N);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &elapsed_ms,
        start,
        stop));

    const double kernel_ms =
        elapsed_ms / iterations;

    CUDA_CHECK(cudaMemcpy(
        h_c.data(),
        d_c,
        bytes,
        cudaMemcpyDeviceToHost));

    // Correctness check
    float max_error = 0.0f;

    for (int i = 0; i < N; ++i) {
        const float expected = h_a[i] + h_b[i];
        max_error = std::max(
            max_error,
            std::abs(h_c[i] - expected));
    }

    // Vector add:
    // 2 reads + 1 write = 3 * sizeof(float) bytes/element
    const double bytes_moved =
        3.0 * static_cast<double>(bytes);

    const double bandwidth_gbps =
        (bytes_moved / (kernel_ms / 1000.0)) /
        1e9;

    std::cout << "N: " << N << '\n';
    std::cout << "Kernel time: "
              << kernel_ms << " ms\n";
    std::cout << "Effective bandwidth: "
              << bandwidth_gbps << " GB/s\n";
    std::cout << "Max error: "
              << max_error << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return max_error == 0.0f
        ? EXIT_SUCCESS
        : EXIT_FAILURE;
}