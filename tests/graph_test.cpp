#include "forge/runtime.h"
#include <iostream>
#include <stdexcept>
#include <utility>
namespace {
void check(bool value) { if (!value) throw std::runtime_error("Graph check failed"); }
template<class E, class F> void expect(F action) {
    try { action(); } catch (const E&) { return; }
    throw std::runtime_error("Expected graph error");
}
}
int main() {
    using namespace forge;
    Runtime runtime;
    Graph graph;
    expect<std::invalid_argument>([&] { graph.execution_order(); });
    auto x = graph.input(Tensor({2, 3}, DataType::Float32, Device::cuda()));
    auto w = graph.input(Tensor({3, 4}, DataType::Float32, Device::cuda()));
    auto h = graph.matmul(x, w);
    auto p = graph.softmax(graph.relu(h));
    graph.output(p); graph.output(h); graph.output(p);
    check(graph.outputs().size() == 2);
    check(graph.descriptor(p).shape() == std::vector<std::int64_t>({2, 4}));
    const auto order = graph.execution_order();
    check(order.size() == graph.nodes().size());
    std::vector<bool> visited(order.size());
    for (auto i : order) {
        check(!visited[i]);
        for (auto dep : graph.nodes()[i].inputs) check(visited[dep]);
        visited[i] = true;
    }
    expect<std::invalid_argument>([&] { runtime.compile(graph, {.matmul_kernel = static_cast<MatMulKernel>(99)}); });
    expect<std::invalid_argument>([&] { runtime.compile(graph); });
    expect<std::invalid_argument>([&] { graph.matmul(x, x); });
    expect<std::invalid_argument>([&] { graph.relu({}); });
    Graph foreign;
    auto f = foreign.input(Tensor({2, 3}, DataType::Float32, Device::cuda()));
    expect<std::invalid_argument>([&] { graph.relu(f); });
    expect<std::invalid_argument>([&] { graph.output(f); });
    expect<std::invalid_argument>([&] { foreign.execution_order(); });
    expect<std::invalid_argument>([&] { graph.input(Tensor({1}, DataType::Float32, Device::cpu())); });
    expect<std::invalid_argument>([&] { graph.input(Tensor({1}, DataType::Int32, Device::cuda())); });
    expect<std::invalid_argument>([&] { graph.input(Tensor({1}, DataType::Float32, Device::cuda(1))); });
    auto scalar = foreign.input(Tensor({}, DataType::Float32, Device::cuda()));
    expect<std::invalid_argument>([&] { foreign.softmax(scalar); });
    Graph moved(std::move(graph));
    check(moved.descriptor(p).numel() == 8);
    expect<std::invalid_argument>([&] { graph.descriptor(p); });
    Graph half_graph;
    auto half = half_graph.input(Tensor({2, 2}, DataType::Float16, Device::cuda()));
    auto single = half_graph.input(Tensor({2, 2}, DataType::Float32, Device::cuda()));
    expect<std::invalid_argument>([&] { half_graph.matmul(half, single); });
    auto half_result = half_graph.softmax(half_graph.relu(half_graph.matmul(half, half)));
    half_graph.output(half_result);
    check(half_graph.descriptor(half_result).dtype() == DataType::Float16);
    check(half_graph.descriptor(half_result).nbytes() == 8);
    Graph repeated;
    auto a = repeated.input(Tensor({2, 2}, DataType::Float32, Device::cuda()));
    auto b = repeated.matmul(a, a);
    auto c = repeated.relu(a);
    repeated.output(repeated.matmul(b, c));
    check(repeated.execution_order().size() == 4);
#if !FORGE_TEST_CUDA
    float storage = 0;
    Graph bound;
    auto v = bound.input(Tensor({1}, DataType::Float32, Device::cuda(), &storage));
    bound.output(v);
    expect<std::runtime_error>([&] { runtime.compile(bound); });
#endif
    std::cout << "Graph validation tests passed\n";
}
