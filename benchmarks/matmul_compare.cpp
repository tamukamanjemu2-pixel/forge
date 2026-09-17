#include "forge/memory.h"
#include "forge/ops/matmul.h"
#include "forge/stream.h"
#include "forge/ops/relu.h"
#ifndef FORGE_COMPARE_FUSION
#define FORGE_COMPARE_FUSION 0
#endif
#ifndef FORGE_COMPARE_FP16
#define FORGE_COMPARE_FP16 0
#endif
#if FORGE_COMPARE_FP16
#include <cuda_fp16.h>
#endif
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
#if FORGE_COMPARE_FP16
using Storage = __half;
constexpr auto dtype = forge::DataType::Float16;
Storage encode_value(float value) { return __float2half_rn(value); }
float decode_value(Storage value) { return __half2float(value); }
#else
using Storage = float;
constexpr auto dtype = forge::DataType::Float32;
Storage encode_value(float value) { return value; }
float decode_value(Storage value) { return value; }
#endif
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
    std::vector<Storage> av(static_cast<std::size_t>(m) * k), bv(static_cast<std::size_t>(k) * n), actual(static_cast<std::size_t>(m) * n);
    for (std::size_t i = 0; i < av.size(); ++i) av[i] = encode_value((static_cast<int>(i % 31) - 15) / 16.0f);
    for (std::size_t i = 0; i < bv.size(); ++i) bv[i] = encode_value((static_cast<int>(i % 29) - 14) / 16.0f);
    // Full double-precision reference; performed outside all measured intervals.
    std::vector<double> expected(actual.size(), 0);
    for (int row = 0; row < m; ++row)
        for (int inner = 0; inner < k; ++inner)
            for (int col = 0; col < n; ++col)
                expected[static_cast<std::size_t>(row) * n + col] +=
                    static_cast<double>(decode_value(av[static_cast<std::size_t>(row) * k + inner])) * decode_value(bv[static_cast<std::size_t>(inner) * n + col]);
    if (FORGE_COMPARE_FUSION) for (auto& value : expected) value = std::max(value, 0.0);
    if (FORGE_COMPARE_FP16) for (auto& value : expected) value = decode_value(encode_value(static_cast<float>(value)));
    Buffer scratch(actual.size() * sizeof(Storage), Device::cuda());
    auto intermediate = scratch.view({m, n}, dtype);
    Buffer as(av.size() * sizeof(Storage), Device::cuda()), bs(bv.size() * sizeof(Storage), Device::cuda()), cs(actual.size() * sizeof(Storage), Device::cuda());
    auto a = as.view({m, k}, dtype), b = bs.view({k, n}, dtype), c = cs.view({m, n}, dtype);
    Tensor ha({m, k}, dtype, Device::cpu(), av.data());
    Tensor hb({k, n}, dtype, Device::cpu(), bv.data());
    Tensor hc({m, n}, dtype, Device::cpu(), actual.data());
    copy_tensor(ha, a); copy_tensor(hb, b);
    Stream stream;
    const auto native = static_cast<cudaStream_t>(stream.native_handle());
    Event start, end;
    double naive_median = 0;
    for (auto kernel : {MatMulKernel::Naive, MatMulKernel::Tiled}) {
      double separate_median = 0;
      for (int variant = 0; variant < (FORGE_COMPARE_FUSION ? 2 : 1); ++variant) {
        const auto submit = [&] {
            if (FORGE_COMPARE_FUSION && variant == 1) matmul_relu(a, b, c, stream, kernel);
            else if (FORGE_COMPARE_FUSION) {
                matmul(a, b, intermediate, stream, kernel);
                relu(intermediate, c, stream);
            } else matmul(a, b, c, stream, kernel);
        };
        submit();
        stream.synchronize();
        copy_tensor(c, hc);
        double max_error = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const auto error = std::abs(decode_value(actual[i]) - expected[i]);
            if (!std::isfinite(decode_value(actual[i])) || error > (FORGE_COMPARE_FP16 ? 0.002 : 0.0001) * (1 + std::abs(expected[i])))
                throw std::runtime_error("Benchmark correctness check failed before timing");
            max_error = std::max(max_error, error);
        }
        for (int i = 0; i < warmup; ++i) submit();
        stream.synchronize();
        std::vector<double> gpu_ms, host_ms;
        gpu_ms.reserve(iterations); host_ms.reserve(iterations);
        for (int i = 0; i < iterations; ++i) {
            const auto before = std::chrono::steady_clock::now();
            FORGE_CUDA_CHECK(cudaEventRecord(start.handle, native));
            submit();
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
        if (variant == 0) separate_median = median;
        std::cout << (kernel == MatMulKernel::Naive ? "naive" : "tiled") << ',' << (FORGE_COMPARE_FUSION ? (variant ? "fused" : "separate") : "gemm") << ','
                  << (FORGE_COMPARE_FUSION && variant == 0 ? 2 : 1) << ','
                  << (FORGE_COMPARE_FUSION && variant == 0 ? 2 * actual.size() * sizeof(Storage) : 0) << ',' << m << ',' << n << ',' << k << ','
                  << warmup << ',' << iterations << ',' << percentile(gpu_ms, 0) << ',' << median << ','
                  << percentile(gpu_ms, 0.95) << ',' << percentile(host_ms, 0.5) << ','
                  << (2.0 * m * n * k / (median * 1e6)) << ',' << max_error << ',' << (FORGE_COMPARE_FUSION ? separate_median : naive_median) / median << '\n';
      }
    }
}
}
int main(int argc, char** argv) {
    try {
        std::vector<Shape> shapes{{64, 64, 64}, {127, 131, 67}, {256, 256, 256}, {512, 512, 512}};
        int warmup = 10, iterations = 50;
        if (argc != 1) {
            if (argc != 4 && argc != 6)
                throw std::invalid_argument("Usage: comparison_binary [M N K [warmup iterations]]; dimensions 1..4096");
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
        std::cout << "# build=Release-like (NDEBUG)\n";
#else
        std::cout << "# build=assertions-enabled\n";
#endif
        std::cout << "# dtype=" << (FORGE_COMPARE_FP16 ? "Float16" : "Float32") << ", accumulation=Float32, Tensor_Cores=no\n";
        std::cout << "# Mode=" << (FORGE_COMPARE_FUSION ? "MatMul+ReLU; baseline=separate per kernel" : "GEMM; baseline=naive") << '\n';
        std::cout << "# Inputs: deterministic modulo-31/modulo-29 patterns; full CPU double reference.\n"
                  << "# CUDA-event intervals bracket operator submission and may include GPU idle time from host dispatch.\n"
                  << "# Host intervals include submission, event calls and completion wait. Transfers/allocations/reference excluded.\n"
                  << "# logical_intermediate_bytes is analytic write+read payload, not measured DRAM traffic.\n"
                  << "# Fixed order: naive then tiled per shape. No clock locking; repeat runs before drawing conclusions.\n"
                  << "kernel,variant,planned_launches,logical_intermediate_bytes,m,n,k,warmup,iterations,event_min_ms,event_median_ms,event_p95_ms,host_median_ms,event_gflops,max_abs_error,baseline_over_current\n"
                  << std::fixed << std::setprecision(6);
        for (auto shape : shapes) run(shape, warmup, iterations);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
