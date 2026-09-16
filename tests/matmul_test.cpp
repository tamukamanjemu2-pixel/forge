#include "forge/ops/matmul.h"
#include "forge/memory.h"
#include "forge/stream.h"
#include "../src/cuda_support.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using namespace forge;

void run_case(int m, int n, int k, bool explicit_stream, MatMulKernel kernel) {
    std::vector<float> av(m * k), bv(k * n), actual(m * n);
    for (std::size_t i = 0; i < av.size(); ++i) av[i] = (static_cast<int>(i % 17) - 8) / 7.0f;
    for (std::size_t i = 0; i < bv.size(); ++i) bv[i] = (static_cast<int>(i % 13) - 6) / 5.0f;
    Buffer a_storage(av.size() * sizeof(float), Device::cuda());
    Buffer b_storage(bv.size() * sizeof(float), Device::cuda());
    Buffer c_storage(actual.size() * sizeof(float), Device::cuda());
    auto a = a_storage.view({m, k}, DataType::Float32);
    auto b = b_storage.view({k, n}, DataType::Float32);
    auto c = c_storage.view({m, n}, DataType::Float32);
    Tensor host_a({m, k}, DataType::Float32, Device::cpu(), av.data());
    Tensor host_b({k, n}, DataType::Float32, Device::cpu(), bv.data());
    Tensor host_c({m, n}, DataType::Float32, Device::cpu(), actual.data());
    copy_tensor(host_a, a);
    copy_tensor(host_b, b);
    if (explicit_stream) {
        Stream original;
        unsigned flags = 0;
        FORGE_CUDA_CHECK(cudaStreamGetFlags(static_cast<cudaStream_t>(original.native_handle()), &flags));
        if (!(flags & cudaStreamNonBlocking)) throw std::runtime_error("Expected non-blocking stream");
        Stream stream(std::move(original));
        bool rejected = false;
        try { original.synchronize(); } catch (const std::logic_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Moved-from stream was accepted");
        // A dependent MatMul on the same stream checks producer/consumer ordering.
        std::vector<float> identity(n * n, 0.0f);
        for (int i = 0; i < n; ++i) identity[i * n + i] = 1.0f;
        Buffer eye_storage(identity.size() * sizeof(float), Device::cuda());
        Buffer result_storage(actual.size() * sizeof(float), Device::cuda());
        auto eye = eye_storage.view({n, n}, DataType::Float32);
        auto result = result_storage.view({m, n}, DataType::Float32);
        Tensor host_eye({n, n}, DataType::Float32, Device::cpu(), identity.data());
        // Prepare the identity before the chain: synchronous upload would otherwise
        // mask ordering errors by completing the first operator.
        copy_tensor(host_eye, eye);
        if (kernel == MatMulKernel::Naive) matmul(a, b, c, stream);
        else matmul(a, b, c, stream, kernel);
        matmul(c, eye, result, stream, kernel);
        stream.synchronize();
        copy_tensor(result, host_c);
    } else {
        if (kernel == MatMulKernel::Naive) matmul(a, b, c);
        else matmul(a, b, c, kernel);
        copy_tensor(c, host_c);
    }
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            double expected = 0;
            for (int i = 0; i < k; ++i)
                expected += static_cast<double>(av[row * k + i]) * bv[i * n + col];
            const auto got = actual[row * n + col];
            if (!std::isfinite(got) || std::abs(got - expected) > 1e-4 + 1e-4 * std::abs(expected)) {
                std::cerr << "MatMul " << m << 'x' << n << 'x' << k << " at " << row << ',' << col
                          << ": expected " << expected << ", got " << got << '\n';
                throw std::runtime_error("MatMul numerical mismatch");
            }
        }
    }
}

void check_device_restoration() {
    int count = 0;
    FORGE_CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count < 2) {
        std::cout << "Multi-GPU stream checks: SKIPPED (fewer than two devices)\n";
        return;
    }
    Buffer storage(12, Device::cuda(0));
    auto a = storage.view({1, 1}, DataType::Float32);
    auto b = storage.view({1, 1}, DataType::Float32, 4);
    auto c = storage.view({1, 1}, DataType::Float32, 8);
    float value = 2;
    Tensor host({1, 1}, DataType::Float32, Device::cpu(), &value);
    copy_tensor(host, a);
    copy_tensor(host, b);
    {
        Stream stream(Device::cuda(0));
        FORGE_CUDA_CHECK(cudaSetDevice(1));
        matmul(a, b, c, stream);
        stream.synchronize();
        int current = -1;
        FORGE_CUDA_CHECK(cudaGetDevice(&current));
        if (current != 1) throw std::runtime_error("MatMul/sync changed current device");
        Stream wrong(Device::cuda(1));
        bool rejected = false;
        try { matmul(a, b, c, wrong); } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("Mismatched stream device was accepted");
        copy_tensor(c, host);
        if (value != 4) throw std::runtime_error("Wrong-device MatMul result");
    }
    int current = -1;
    FORGE_CUDA_CHECK(cudaGetDevice(&current));
    if (current != 1) throw std::runtime_error("Stream destruction changed current device");
    FORGE_CUDA_CHECK(cudaSetDevice(0));
}
}

int main() {
    FORGE_CUDA_CHECK(cudaSetDevice(0));
    for (auto kernel : {forge::MatMulKernel::Naive, forge::MatMulKernel::Tiled}) {
    for (bool explicit_stream : {false, true}) {
        run_case(1, 1, 1, explicit_stream, kernel);
        run_case(2, 2, 3, explicit_stream, kernel);
        run_case(15, 16, 17, explicit_stream, kernel);
        run_case(16, 17, 15, explicit_stream, kernel);
        run_case(17, 19, 23, explicit_stream, kernel);
        run_case(32, 48, 16, explicit_stream, kernel);
        run_case(3, 7, 257, explicit_stream, kernel);
    }
    }
    check_device_restoration();
    std::cout << "MatMul default/explicit stream tests passed\n";
}
