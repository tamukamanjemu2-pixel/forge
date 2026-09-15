#include "forge/memory.h"
#include "forge/stream.h"
#include "forge/ops/relu.h"
#include "forge/ops/softmax.h"
#include "../src/cuda_support.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using namespace forge;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void run(std::vector<std::int64_t> shape, std::vector<float> values, bool normalize, bool explicit_stream, bool chain = false) {
    std::vector<float> result(values.size());
    Buffer input_storage(values.size() * sizeof(float), Device::cuda());
    Buffer output_storage(values.size() * sizeof(float), Device::cuda());
    Buffer intermediate_storage(values.size() * sizeof(float), Device::cuda());
    auto input = input_storage.view(shape, DataType::Float32);
    auto output = output_storage.view(shape, DataType::Float32);
    auto intermediate = intermediate_storage.view(shape, DataType::Float32);
    Tensor host_input(shape, DataType::Float32, Device::cpu(), values.data());
    Tensor host_output(shape, DataType::Float32, Device::cpu(), result.data());
    copy_tensor(host_input, input);
    if (explicit_stream) {
        Stream stream;
        if (chain) {
            relu(input, intermediate, stream);
            softmax(intermediate, output, stream);
        } else if (normalize) softmax(input, output, stream);
        else relu(input, output, stream);
        stream.synchronize();
    } else {
        if (chain) {
            relu(input, intermediate);
            softmax(intermediate, output);
        } else if (normalize) softmax(input, output);
        else relu(input, output);
    }
    copy_tensor(output, host_output);
    if (chain) for (auto& value : values) value = std::max(value, 0.0f);
    if (!normalize) {
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (std::isnan(values[i])) check(std::isnan(result[i]), "ReLU must propagate NaN");
            else check(result[i] == (values[i] < 0 ? 0 : values[i]), "ReLU mismatch");
        }
    } else {
        const auto columns = static_cast<std::size_t>(shape.back());
        for (std::size_t base = 0; base < values.size(); base += columns) {
            double maximum = -std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < columns; ++i) maximum = std::max(maximum, static_cast<double>(values[base + i]));
            double denominator = 0;
            for (std::size_t i = 0; i < columns; ++i) denominator += std::exp(static_cast<double>(values[base + i]) - maximum);
            double row_sum = 0;
            for (std::size_t i = 0; i < columns; ++i) {
                const double expected = std::exp(static_cast<double>(values[base + i]) - maximum) / denominator;
                const auto actual = result[base + i];
                check(std::isfinite(actual) && actual >= 0 && actual <= 1, "Invalid softmax probability");
                if (std::abs(actual - expected) > 2e-6 + 2e-5 * expected) {
                    std::cerr << "Softmax width " << columns << " index " << base + i << ": expected "
                              << expected << ", got " << actual << '\n';
                    throw std::runtime_error("Softmax mismatch");
                }
                row_sum += actual;
            }
            check(std::abs(row_sum - 1.0) < 2e-5, "Softmax row does not sum to one");
        }
    }
}

void multi_device() {
    int count = 0;
    FORGE_CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count < 2) {
        std::cout << "Activation multi-GPU checks: SKIPPED\n";
        return;
    }
    Buffer input_storage(4, Device::cuda(0)), output_storage(4, Device::cuda(0));
    auto input = input_storage.view({1}, DataType::Float32);
    auto output = output_storage.view({1}, DataType::Float32);
    float value = -2;
    Tensor host({1}, DataType::Float32, Device::cpu(), &value);
    copy_tensor(host, input);
    Stream correct(Device::cuda(0)), wrong(Device::cuda(1));
    FORGE_CUDA_CHECK(cudaSetDevice(1));
    for (bool normalize : {false, true}) {
        bool rejected = false;
        try {
            if (normalize) softmax(input, output, wrong);
            else relu(input, output, wrong);
        } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "Activation accepted wrong-device stream");
        if (normalize) softmax(input, output, correct);
        else relu(input, output, correct);
        correct.synchronize();
        int current = -1;
        FORGE_CUDA_CHECK(cudaGetDevice(&current));
        check(current == 1, "Activation changed current device");
        copy_tensor(output, host);
        check(value == (normalize ? 1.0f : 0.0f), "Activation wrong-device result");
    }
    FORGE_CUDA_CHECK(cudaSetDevice(0));
}
}

int main() {
    FORGE_CUDA_CHECK(cudaSetDevice(0));
    for (bool stream : {false, true}) {
        run({}, {-3}, false, stream);
        run({7}, {-3, -0.0f, 0, 2, std::numeric_limits<float>::infinity(),
                  -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}, false, stream);
        std::vector<float> relu_values(1025);
        for (std::size_t i = 0; i < relu_values.size(); ++i) relu_values[i] = static_cast<int>(i % 19) - 9;
        run({5, 205}, relu_values, false, stream);
        for (std::int64_t width : {1, 3, 31, 32, 255, 256, 257, 1025, 4097}) {
            std::vector<float> values(static_cast<std::size_t>(6 * width));
            for (std::size_t i = 0; i < values.size(); ++i)
                values[i] = 10000.0f + (static_cast<int>(i % 23) - 11) * 0.25f;
            run({2, 3, width}, values, true, stream);
        }
        run({4}, {10000, 10001, -10000, 9999}, true, stream);
        run({2, 3}, {-10000, -10001, -10002, 4, 4, 4}, true, stream);
        run({3}, {-std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), 0}, true, stream);
        run({2, 3}, {-5, 1, 2, -8, -1, 0}, true, stream, true);
        // Exercises the block-stride row loop beyond the capped grid dimension.
        run({65536, 1}, std::vector<float>(65536, 42), true, stream);
    }
    multi_device();
    std::cout << "ReLU and Softmax tests passed\n";
}
