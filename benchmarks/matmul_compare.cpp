#include "forge/memory.h"
#include "forge/ops/matmul.h"
#include "forge/stream.h"
#include "../src/cuda_support.h"
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
class Event {
public:
    Event() { FORGE_CUDA_CHECK(cudaEventCreate(&handle)); }
    ~Event() { forge::detail::cuda_cleanup_check(cudaEventDestroy(handle), "benchmark event destruction"); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    cudaEvent_t handle = nullptr;
};
int number(const char* text, int minimum, int maximum) {
    int value = 0;
    const std::string_view input(text);
    const auto parsed = std::from_chars(input.data(), input.data() + input.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != input.data() + input.size() || value < minimum || value > maximum)
        throw std::invalid_argument("Invalid benchmark argument or out-of-range value");
    return value;
}
double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>(fraction * (values.size() - 1))];
}
struct Shape { int m, n, k; };
void run(Shape shape, int warmup, int iterations) {
    using namespace forge;
    const auto [m, n, k] = shape;
    std::vector<float> av(static_cast<std::size_t>(m) * k), bv(static_cast<std::size_t>(k) * n), actual(static_cast<std::size_t>(m) * n);
    for (std::size_t i = 0; i < av.size(); ++i) av[i] = (static_cast<int>(i % 31) - 15) / 16.0f;
    for (std::size_t i = 0; i < bv.size(); ++i) bv[i] = (static_cast<int>(i % 29) - 14) / 16.0f;
    // Full double-precision reference; performed outside all measured intervals.
    std::vector<double> expected(actual.size(), 0);
    for (int row = 0; row < m; ++row)
        for (int inner = 0; inner < k; ++inner)
            for (int col = 0; col < n; ++col)
                expected[static_cast<std::size_t>(row) * n + col] +=
                    static_cast<double>(av[static_cast<std::size_t>(row) * k + inner]) * bv[static_cast<std::size_t>(inner) * n + col];
    Buffer as(av.size() * 4, Device::cuda()), bs(bv.size() * 4, Device::cuda()), cs(actual.size() * 4, Device::cuda());
    auto a = as.view({m, k}, DataType::Float32), b = bs.view({k, n}, DataType::Float32), c = cs.view({m, n}, DataType::Float32);
    Tensor ha({m, k}, DataType::Float32, Device::cpu(), av.data());
    Tensor hb({k, n}, DataType::Float32, Device::cpu(), bv.data());
    Tensor hc({m, n}, DataType::Float32, Device::cpu(), actual.data());
    copy_tensor(ha, a); copy_tensor(hb, b);
    Stream stream;
    const auto native = static_cast<cudaStream_t>(stream.native_handle());
    Event start, end;
    double naive_median = 0;
    for (auto kernel : {MatMulKernel::Naive, MatMulKernel::Tiled}) {
        matmul(a, b, c, stream, kernel);
        stream.synchronize();
        copy_tensor(c, hc);
        double max_error = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const auto error = std::abs(actual[i] - expected[i]);
            if (!std::isfinite(actual[i]) || error > 1e-4 + 1e-4 * std::abs(expected[i]))
                throw std::runtime_error("Benchmark correctness check failed before timing");
            max_error = std::max(max_error, error);
        }
        for (int i = 0; i < warmup; ++i) matmul(a, b, c, stream, kernel);
        stream.synchronize();
        std::vector<double> gpu_ms, host_ms;
        gpu_ms.reserve(iterations); host_ms.reserve(iterations);
        for (int i = 0; i < iterations; ++i) {
            const auto before = std::chrono::steady_clock::now();
            FORGE_CUDA_CHECK(cudaEventRecord(start.handle, native));
            matmul(a, b, c, stream, kernel);
            FORGE_CUDA_CHECK(cudaEventRecord(end.handle, native));
            FORGE_CUDA_CHECK(cudaEventSynchronize(end.handle));
            const auto after = std::chrono::steady_clock::now();
            float elapsed = 0;
            FORGE_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start.handle, end.handle));
            if (!(elapsed > 0) || !std::isfinite(elapsed)) throw std::runtime_error("Invalid CUDA event duration");
            gpu_ms.push_back(elapsed);
            host_ms.push_back(std::chrono::duration<double, std::milli>(after - before).count());
        }
        const double median = percentile(gpu_ms, 0.5);
        if (kernel == MatMulKernel::Naive) naive_median = median;
        std::cout << (kernel == MatMulKernel::Naive ? "naive" : "tiled") << ',' << m << ',' << n << ',' << k << ','
                  << warmup << ',' << iterations << ',' << percentile(gpu_ms, 0) << ',' << median << ','
                  << percentile(gpu_ms, 0.95) << ',' << percentile(host_ms, 0.5) << ','
                  << (2.0 * m * n * k / (median * 1e6)) << ',' << max_error << ',' << naive_median / median << '\n';
    }
}
}
int main(int argc, char** argv) {
    try {
        std::vector<Shape> shapes{{64, 64, 64}, {127, 131, 67}, {256, 256, 256}, {512, 512, 512}};
        int warmup = 10, iterations = 50;
        if (argc != 1) {
            if (argc != 4 && argc != 6)
                throw std::invalid_argument("Usage: forge_matmul_compare [M N K [warmup iterations]]; dimensions 1..4096");
            shapes = {{number(argv[1], 1, 4096), number(argv[2], 1, 4096), number(argv[3], 1, 4096)}};
            if (argc == 6) { warmup = number(argv[4], 0, 10000); iterations = number(argv[5], 1, 10000); }
        }
        FORGE_CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp properties{};
        int runtime = 0, driver = 0;
        FORGE_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        FORGE_CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        FORGE_CUDA_CHECK(cudaDriverGetVersion(&driver));
        std::cout << "# GPU=" << properties.name << ", compute_capability=" << properties.major << '.' << properties.minor
                  << ", CUDA_headers=" << CUDART_VERSION << ", CUDA_runtime=" << runtime << ", driver=" << driver
                  << ", host_compiler=" << __VERSION__ << '\n';
#ifdef NDEBUG
        std::cout << "# build=Release-like (NDEBUG), dtype=Float32\n";
#else
        std::cout << "# build=assertions-enabled, dtype=Float32\n";
#endif
        std::cout << "# Inputs: deterministic modulo-31/modulo-29 patterns; full CPU double reference.\n"
                  << "# CUDA-event intervals bracket operator submission and may include GPU idle time from host dispatch.\n"
                  << "# Host intervals include submission, event calls and completion wait. Transfers/allocations/reference excluded.\n"
                  << "# Fixed order: naive then tiled per shape. No clock locking; repeat runs before drawing conclusions.\n"
                  << "kernel,m,n,k,warmup,iterations,event_min_ms,event_median_ms,event_p95_ms,host_median_ms,event_gflops,max_abs_error,naive_over_current\n"
                  << std::fixed << std::setprecision(6);
        for (auto shape : shapes) run(shape, warmup, iterations);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
