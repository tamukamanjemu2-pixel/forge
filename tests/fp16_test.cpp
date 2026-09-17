#include "forge/runtime.h"
#include "forge/memory.h"
#include "forge/stream.h"
#include "forge/ops/relu.h"
#include "forge/ops/softmax.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace forge;
float rounded(double x) { return __half2float(__float2half_rn(static_cast<float>(x))); }
std::vector<__half> encode(const std::vector<float>& values) {
    std::vector<__half> result;
    for (auto v : values) result.push_back(__float2half_rn(v));
    return result;
}
std::vector<float> decode(const std::vector<__half>& values) {
    std::vector<float> result;
    for (auto v : values) result.push_back(__half2float(v));
    return result;
}
void close(float actual, float expected, float absolute = 0.002f, float relative = 0.002f) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > absolute + relative * std::abs(expected)) {
        std::cerr << "FP16 mismatch: expected " << expected << ", got " << actual << '\n';
        throw std::runtime_error("FP16 tolerance exceeded");
    }
}
std::vector<float> reference_gemm(const std::vector<float>& a, const std::vector<float>& b, int m, int n, int k, bool activate = false) {
    std::vector<float> result(static_cast<std::size_t>(m) * n);
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col) {
            double sum = 0;
            for (int i = 0; i < k; ++i) sum += static_cast<double>(a[row * k + i]) * b[i * n + col];
            result[row * n + col] = rounded(activate ? std::max(sum, 0.0) : sum);
        }
    return result;
}
std::vector<float> reference_softmax(const std::vector<float>& values, std::size_t columns) {
    std::vector<float> result(values.size());
    for (std::size_t base = 0; base < values.size(); base += columns) {
        double maximum = -std::numeric_limits<double>::infinity(), sum = 0;
        for (std::size_t i = 0; i < columns; ++i) maximum = std::max(maximum, static_cast<double>(values[base + i]));
        for (std::size_t i = 0; i < columns; ++i) sum += std::exp(static_cast<double>(values[base + i]) - maximum);
        for (std::size_t i = 0; i < columns; ++i) result[base + i] = rounded(std::exp(static_cast<double>(values[base + i]) - maximum) / sum);
    }
    return result;
}
void gemm_case(int m, int n, int k, MatMulKernel kernel, bool fused, bool explicit_stream) {
    std::vector<float> af(m * k), bf(k * n);
    for (std::size_t i = 0; i < af.size(); ++i) af[i] = (static_cast<int>(i % 17) - 8) / 13.0f;
    for (std::size_t i = 0; i < bf.size(); ++i) bf[i] = (static_cast<int>(i % 11) - 5) / 9.0f;
    auto av = encode(af), bv = encode(bf);
    std::vector<__half> cv(m * n);
    Buffer as(av.size() * 2, Device::cuda()), bs(bv.size() * 2, Device::cuda()), cs(cv.size() * 2, Device::cuda());
    auto a = as.view({m, k}, DataType::Float16), b = bs.view({k, n}, DataType::Float16), c = cs.view({m, n}, DataType::Float16);
    Tensor ha({m, k}, DataType::Float16, Device::cpu(), av.data()), hb({k, n}, DataType::Float16, Device::cpu(), bv.data());
    Tensor hc({m, n}, DataType::Float16, Device::cpu(), cv.data());
    copy_tensor(ha, a); copy_tensor(hb, b);
    if (explicit_stream) {
        Stream stream;
        if (fused) matmul_relu(a, b, c, stream, kernel); else matmul(a, b, c, stream, kernel);
        stream.synchronize();
    } else {
        if (fused) matmul_relu(a, b, c, kernel); else matmul(a, b, c, kernel);
    }
    copy_tensor(c, hc);
    auto expected = reference_gemm(decode(av), decode(bv), m, n, k, fused);
    for (std::size_t i = 0; i < cv.size(); ++i) close(__half2float(cv[i]), expected[i]);
}
void activation_case(std::vector<float> values, std::vector<std::int64_t> shape, bool normalize, bool explicit_stream) {
    auto av = encode(values);
    std::vector<__half> cv(values.size());
    Buffer as(av.size() * 2, Device::cuda()), cs(cv.size() * 2, Device::cuda());
    auto a = as.view(shape, DataType::Float16), c = cs.view(shape, DataType::Float16);
    Tensor ha(shape, DataType::Float16, Device::cpu(), av.data()), hc(shape, DataType::Float16, Device::cpu(), cv.data());
    copy_tensor(ha, a);
    if (explicit_stream) {
        Stream stream;
        if (normalize) softmax(a, c, stream); else relu(a, c, stream);
        stream.synchronize();
    } else {
        if (normalize) softmax(a, c); else relu(a, c);
    }
    copy_tensor(c, hc);
    const auto quantized = decode(av);
    if (normalize) {
        auto expected = reference_softmax(quantized, shape.back());
        double sum = 0;
        for (std::size_t i = 0; i < cv.size(); ++i) {
            const auto actual = __half2float(cv[i]);
            close(actual, expected[i], 0.00002f, 0.002f);
            if (actual < 0 || actual > 1) throw std::runtime_error("FP16 invalid probability");
            sum += actual;
            if ((i + 1) % static_cast<std::size_t>(shape.back()) == 0) {
                if (std::abs(sum - 1) > 0.002) throw std::runtime_error("FP16 row sum tolerance");
                sum = 0;
            }
        }
    } else {
        for (std::size_t i = 0; i < cv.size(); ++i) {
            const auto actual = __half2float(cv[i]), input = quantized[i];
            if (std::isnan(input)) {
                if (!std::isnan(actual)) throw std::runtime_error("FP16 ReLU lost NaN");
            } else if (actual != (input < 0 ? 0 : input)) throw std::runtime_error("FP16 ReLU mismatch");
        }
    }
}
void graph_case(int batch, MatMulKernel kernel, bool fused, bool reuse) {
    constexpr int width = 3, hidden = 5, classes = 2;
    std::vector<float> xf(batch * width), w1f(width * hidden), w2f(hidden * classes);
    for (std::size_t i = 0; i < xf.size(); ++i) xf[i] = (static_cast<int>(i % 9) - 4) / 7.0f;
    for (std::size_t i = 0; i < w1f.size(); ++i) w1f[i] = (static_cast<int>(i % 7) - 3) / 5.0f;
    for (std::size_t i = 0; i < w2f.size(); ++i) w2f[i] = (static_cast<int>(i % 5) - 2) / 3.0f;
    auto xv = encode(xf), w1v = encode(w1f), w2v = encode(w2f);
    std::vector<__half> result(batch * classes);
    Buffer xs(xv.size() * 2, Device::cuda()), s1(w1v.size() * 2, Device::cuda()), s2(w2v.size() * 2, Device::cuda());
    auto x = xs.view({batch, width}, DataType::Float16), w1 = s1.view({width, hidden}, DataType::Float16), w2 = s2.view({hidden, classes}, DataType::Float16);
    Tensor hx({batch, width}, DataType::Float16, Device::cpu(), xv.data()), h1({width, hidden}, DataType::Float16, Device::cpu(), w1v.data()), h2({hidden, classes}, DataType::Float16, Device::cpu(), w2v.data());
    Tensor hr({batch, classes}, DataType::Float16, Device::cpu(), result.data());
    copy_tensor(hx, x); copy_tensor(h1, w1); copy_tensor(h2, w2);
    Graph graph;
    auto input = graph.input(x), weights1 = graph.input(w1), weights2 = graph.input(w2);
    auto output = graph.softmax(graph.matmul(graph.relu(graph.matmul(input, weights1)), weights2));
    graph.output(output);
    Runtime runtime;
    auto executable = runtime.compile(graph, {.reuse_memory = reuse, .matmul_kernel = kernel, .fuse_matmul_relu = fused});
    auto activations = reference_gemm(decode(xv), decode(w1v), batch, hidden, width, fused);
    for (auto& v : activations) v = std::max(v, 0.0f);
    auto logits = reference_gemm(activations, decode(w2v), batch, classes, hidden);
    auto expected = reference_softmax(logits, classes);
    for (int repeat = 0; repeat < 2; ++repeat) {
        runtime.execute(executable);
        copy_tensor(executable.output(output), hr);
        for (std::size_t i = 0; i < result.size(); ++i) close(__half2float(result[i]), expected[i], 0.002f, 0.002f);
    }
}
}
int main() {
    static_assert(sizeof(__half) == 2);
    for (auto kernel : {forge::MatMulKernel::Naive, forge::MatMulKernel::Tiled})
        for (bool fused : {false, true}) {
            for (bool stream : {false, true}) {
                gemm_case(1, 1, 1, kernel, fused, stream);
                gemm_case(15, 17, 33, kernel, fused, stream);
                gemm_case(3, 7, 257, kernel, fused, stream);
            }
            for (bool reuse : {false, true}) {
                graph_case(1, kernel, fused, reuse);
                graph_case(7, kernel, fused, reuse);
            }
        }
    for (bool stream : {false, true}) {
        activation_case({-2}, {}, false, stream);
        activation_case({-65504, -0.0f, 0, 0.000000059604645f, 2,
                         std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}, {7}, false, stream);
        activation_case({65504, -65504, 0}, {3}, true, stream);
        for (int width : {1, 3, 255, 257, 1025, 4097}) {
            std::vector<float> values(2 * width);
            for (std::size_t i = 0; i < values.size(); ++i) values[i] = 1000 + (static_cast<int>(i % 13) - 6) * 0.5f;
            activation_case(values, {2, width}, true, stream);
        }
        activation_case(std::vector<float>(65536, 1), {65536, 1}, true, stream);
    }
    std::cout << "FP16 operator and runtime tests passed (Float32 accumulation; no Tensor Cores)\n";
}
