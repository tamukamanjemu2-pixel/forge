#include "forge/memory_plan.h"
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
using namespace forge;
void check(bool condition) { if (!condition) throw std::runtime_error("Memory planner invariant failed"); }
template<class E, class F> void expect(F action) {
    try { action(); } catch (const E&) { return; }
    throw std::runtime_error("Expected planner error");
}
// Simulate slot contents independently of the planner's lifetime computation.
void verify(const Graph& graph, const MemoryPlan& plan) {
    const auto& nodes = graph.nodes();
    std::vector<std::size_t> content(plan.capacities.size(), MemoryPlan::no_slot);
    std::size_t payload = 0, reserved = 0;
    for (auto capacity : plan.capacities) { check(capacity % 256 == 0); reserved += capacity; }
    for (auto i : plan.order) {
        const auto slot = plan.slots[i];
        if (nodes[i].operation == Graph::Operation::Input) {
            check(slot == MemoryPlan::no_slot); continue;
        }
        check(slot < content.size());
        check(plan.capacities[slot] >= nodes[i].metadata.nbytes());
        for (auto dependency : nodes[i].inputs) {
            if (nodes[dependency].operation == Graph::Operation::Input) continue;
            check(content[plan.slots[dependency]] == dependency);
            check(slot != plan.slots[dependency]);
        }
        content[slot] = i;
        payload += nodes[i].metadata.nbytes();
    }
    for (auto output : graph.outputs())
        if (nodes[output].operation != Graph::Operation::Input)
            check(content[plan.slots[output]] == output);
    check(plan.statistics.tensor_bytes == payload);
    check(plan.statistics.reserved_bytes == reserved);
    check(plan.statistics.peak_live_bytes <= reserved);
    check(plan.statistics.allocation_count == content.size());
}
}
int main() {
    using namespace forge;
    Graph chain;
    auto x = chain.input(Tensor({8}, DataType::Float32, Device::cuda()));
    auto a = chain.relu(x), b = chain.relu(a), c = chain.relu(b), d = chain.relu(c);
    chain.output(d);
    auto plan = plan_memory(chain);
    verify(chain, plan);
    check(plan.statistics.allocation_count == 2);
    check(plan.statistics.reuse_count == 2);
    check(plan.statistics.tensor_bytes == 128);
    check(plan.statistics.peak_live_bytes == 64);
    check(plan.statistics.reserved_bytes == 512);
    check(plan.slots[1] == plan.slots[3] && plan.slots[2] == plan.slots[4]);
    auto baseline = plan_memory(chain, false);
    verify(chain, baseline);
    check(baseline.statistics.allocation_count == 4 && baseline.statistics.reuse_count == 0);
    check(baseline.statistics.reserved_bytes == 1024);
    chain.output(a); // Early output must survive all later operators.
    verify(chain, plan_memory(chain));
    check(plan_memory(chain).statistics.allocation_count == 3);
    expect<std::invalid_argument>([&] { plan_memory(chain, true, 0); });
    expect<std::invalid_argument>([&] { plan_memory(chain, true, 3); });
    Graph identity;
    auto input = identity.input(Tensor({1}, DataType::Float32, Device::cuda()));
    identity.output(input);
    check(plan_memory(identity).statistics.reserved_bytes == 0);
    Graph oversized;
    auto giant = oversized.input(Tensor({std::numeric_limits<std::int64_t>::max() / 2}, DataType::Float32, Device::cuda()));
    oversized.output(oversized.relu(giant));
    expect<std::overflow_error>([&] { plan_memory(oversized); });

    Graph varying;
    auto large = varying.input(Tensor({17, 17}, DataType::Float32, Device::cuda()));
    auto weights = varying.input(Tensor({17, 2}, DataType::Float32, Device::cuda()));
    auto left = varying.relu(large);
    auto small = varying.matmul(left, weights);
    auto output = varying.softmax(varying.relu(small));
    varying.output(output);
    verify(varying, plan_memory(varying));

    for (unsigned seed = 0; seed < 32; ++seed) {
        std::mt19937 random(seed);
        Graph graph;
        std::vector<Graph::Value> values;
        values.push_back(graph.input(Tensor({2, 2}, DataType::Float32, Device::cuda())));
        for (int i = 0; i < 80; ++i) {
            const auto source = values[random() % values.size()];
            switch (random() % 3) {
                case 0: values.push_back(graph.relu(source)); break;
                case 1: values.push_back(graph.softmax(source)); break;
                default: values.push_back(graph.matmul(source, values[random() % values.size()])); break;
            }
            if (i % 13 == 0) graph.output(values.back());
        }
        graph.output(values.back());
        const auto reused = plan_memory(graph);
        const auto dedicated = plan_memory(graph, false);
        verify(graph, reused); verify(graph, dedicated);
        check(reused.statistics.reserved_bytes <= dedicated.statistics.reserved_bytes);
        check(reused.slots == plan_memory(graph).slots); // deterministic planning
    }
    std::cout << "Memory planning tests passed\n";
}
