#include "forge/runtime.h"
#include "forge/memory.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>
namespace {
void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class E, class F> void expect(F action) {
    try { action(); } catch (const E&) { return; }
    throw std::runtime_error("Expected runtime rejection");
}
void run(int batch, bool reuse) {
    using namespace forge;
    constexpr int inputs = 3, hidden = 5, classes = 2;
    std::vector<float> x(batch * inputs), w1(inputs * hidden), w2(hidden * classes), result(batch * classes);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<int>(i % 7) - 3;
    for (std::size_t i = 0; i < w1.size(); ++i) w1[i] = (static_cast<int>(i % 5) - 2) * 0.25f;
    for (std::size_t i = 0; i < w2.size(); ++i) w2[i] = (static_cast<int>(i % 3) - 1) * 0.5f;
    Buffer xs(x.size() * 4, Device::cuda()), ws1(w1.size() * 4, Device::cuda()), ws2(w2.size() * 4, Device::cuda());
    auto tx = xs.view({batch, inputs}, DataType::Float32);
    auto tw1 = ws1.view({inputs, hidden}, DataType::Float32);
    auto tw2 = ws2.view({hidden, classes}, DataType::Float32);
    Tensor hx({batch, inputs}, DataType::Float32, Device::cpu(), x.data());
    Tensor hw1({inputs, hidden}, DataType::Float32, Device::cpu(), w1.data());
    Tensor hw2({hidden, classes}, DataType::Float32, Device::cpu(), w2.data());
    Tensor hr({batch, classes}, DataType::Float32, Device::cpu(), result.data());
    copy_tensor(hw1, tw1); copy_tensor(hw2, tw2);
    Runtime runtime;
    Graph::Value output, intermediate;
    auto executable = [&] {
        Graph graph;
        auto a = graph.input(tx), b = graph.input(tw1), c = graph.input(tw2);
        intermediate = graph.matmul(a, b);
        auto activation = graph.relu(intermediate);
        output = graph.softmax(graph.matmul(activation, c));
        graph.output(output);
        return runtime.compile(graph, {.reuse_memory = reuse});
    }(); // Graph builder is destroyed; compiled snapshot remains valid.
    expect<std::logic_error>([&] { executable.output(output); });
    expect<std::invalid_argument>([&] { executable.output(intermediate); });
    check(executable.memory_statistics().tensor_bytes == static_cast<std::size_t>(batch * (2 * hidden + 2 * classes) * 4), "Payload accounting");
    check(executable.allocated_bytes() == executable.memory_statistics().reserved_bytes, "Reserved accounting");
    check(executable.memory_statistics().allocation_count == (reuse ? 2u : 4u), "Allocation count");
    check(executable.memory_statistics().reuse_count == (reuse ? 2u : 0u), "Reuse count");
    const void* first_pointer = nullptr;
    for (int iteration = 0; iteration < 2; ++iteration) {
        for (auto& value : x) value += iteration * 0.75f;
        copy_tensor(hx, tx);
        runtime.execute(executable);
        const auto& device_output = executable.output(output);
        if (first_pointer) check(first_pointer == device_output.data(), "Output storage changed");
        first_pointer = device_output.data();
        copy_tensor(device_output, hr);
        for (int row = 0; row < batch; ++row) {
            double activation[hidden]{}, logits[classes]{};
            for (int j = 0; j < hidden; ++j) {
                for (int k = 0; k < inputs; ++k) activation[j] += static_cast<double>(x[row * inputs + k]) * w1[k * hidden + j];
                activation[j] = std::max(activation[j], 0.0);
            }
            for (int j = 0; j < classes; ++j)
                for (int k = 0; k < hidden; ++k) logits[j] += activation[k] * w2[k * classes + j];
            const double maximum = *std::max_element(logits, logits + classes);
            double sum = 0;
            for (auto& value : logits) { value = std::exp(value - maximum); sum += value; }
            for (int j = 0; j < classes; ++j) {
                const auto actual = result[row * classes + j];
                check(std::isfinite(actual) && std::abs(actual - logits[j] / sum) < 2e-5, "Inference mismatch");
            }
        }
    }
    auto moved = std::move(executable);
    expect<std::logic_error>([&] { runtime.execute(executable); });
    runtime.execute(moved);
    check(moved.output(output).data() == first_pointer, "Move changed output storage");
    Graph pass;
    auto input = pass.input(tx);
    pass.output(input);
    auto identity = runtime.compile(pass);
    check(identity.allocated_bytes() == 0, "Input-only graph allocated tensors");
    runtime.execute(identity);
    check(identity.output(input).data() == tx.data(), "Input-only graph changed pointer");
    expect<std::invalid_argument>([&] { identity.output(output); });
}
void retained_output() {
    using namespace forge;
    std::vector<float> values{-1, 2, -3, 4}, result(4);
    Buffer storage(16, Device::cuda());
    auto input = storage.view({4}, DataType::Float32);
    Tensor host({4}, DataType::Float32, Device::cpu(), values.data());
    Tensor readback({4}, DataType::Float32, Device::cpu(), result.data());
    copy_tensor(host, input);
    Graph graph;
    auto x = graph.input(input);
    auto early = graph.relu(x);
    auto probabilities = graph.softmax(early);
    auto later = graph.softmax(graph.relu(probabilities));
    graph.output(early); graph.output(later);
    Runtime runtime;
    auto executable = runtime.compile(graph);
    for (int iteration = 0; iteration < 2; ++iteration) {
        runtime.execute(executable);
        copy_tensor(executable.output(early), readback);
        check(result == std::vector<float>({0, 2, 0, 4}), "Early output overwritten by reuse");
        copy_tensor(executable.output(later), readback);
        double sum = 0;
        for (float probability : result) {
            check(std::isfinite(probability) && probability >= 0 && probability <= 1, "Invalid retained probability");
            sum += probability;
        }
        check(std::abs(sum - 1) < 2e-5, "Retained probability sum");
    }
}

}
int main() {
    for (bool reuse : {false, true}) { run(1, reuse); run(7, reuse); run(32, reuse); }
    retained_output();
    std::cout << "Graph runtime inference tests passed\n";
}
