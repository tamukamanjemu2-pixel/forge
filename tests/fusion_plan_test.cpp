#include "forge/graph_plan.h"
#include "forge/memory_plan.h"
#include <iostream>
#include <stdexcept>
namespace {
using namespace forge;
void check(bool value) { if (!value) throw std::runtime_error("Fusion planning check failed"); }
void verify(const Graph& graph) {
    auto program = plan_graph(graph, true);
    auto memory = plan_memory(graph, true, 256, true);
    std::vector<std::size_t> contents(memory.capacities.size(), MemoryPlan::no_slot);
    for (auto i : program.order) {
        const auto& node = program.nodes[i];
        if (node.operation == Graph::Operation::Input || node.operation == Graph::Operation::Elided) {
            check(memory.slots[i] == MemoryPlan::no_slot); continue;
        }
        for (auto input : node.inputs) {
            check(node.operation != Graph::Operation::Elided);
            if (program.nodes[input].operation != Graph::Operation::Input) {
                check(contents[memory.slots[input]] == input);
                check(memory.slots[input] != memory.slots[i]);
            }
        }
        contents[memory.slots[i]] = i;
    }
    for (auto output : program.outputs)
        if (program.nodes[output].operation != Graph::Operation::Input)
            check(contents[memory.slots[output]] == output);
}
}
int main() {
    using namespace forge;
    Graph graph;
    auto x = graph.input(Tensor({2, 2}, DataType::Float32, Device::cuda()));
    auto w = graph.input(Tensor({2, 2}, DataType::Float32, Device::cuda()));
    auto raw = graph.matmul(x, w);
    auto activation = graph.relu(raw);
    auto output = graph.softmax(activation);
    graph.output(output);
    check(plan_graph(graph).fused_pairs == 0);
    auto fused = plan_graph(graph, true);
    check(fused.fused_pairs == 1 && fused.kernel_launches == 2);
    check(fused.nodes[2].operation == Graph::Operation::Elided);
    check(fused.nodes[3].operation == Graph::Operation::MatMulReLU);
    check(fused.nodes[3].inputs == std::vector<std::size_t>({0, 1}));
    check(graph.nodes()[2].operation == Graph::Operation::MatMul); // Original unchanged.
    check(plan_memory(graph, false, 256, true).statistics.tensor_bytes == 32);
    verify(graph);
    graph.output(raw); // Observable pre-activation blocks fusion.
    check(plan_graph(graph, true).fused_pairs == 0);
    verify(graph);

    Graph branch;
    auto a = branch.input(Tensor({3, 3}, DataType::Float32, Device::cuda()));
    auto mm = branch.matmul(a, a);
    branch.output(branch.relu(mm));
    branch.output(branch.softmax(mm));
    check(plan_graph(branch, true).fused_pairs == 0);
    verify(branch);

    Graph chain;
    auto input = chain.input(Tensor({4, 4}, DataType::Float32, Device::cuda()));
    auto h1 = chain.relu(chain.matmul(input, input));
    auto h2 = chain.relu(chain.matmul(h1, h1));
    chain.output(h1); chain.output(h2);
    check(plan_graph(chain, true).fused_pairs == 2);
    verify(chain);
    // Inputs to a delayed fused operation must stay live until the ReLU's position.
    Graph delayed;
    auto d = delayed.input(Tensor({2, 2}, DataType::Float32, Device::cuda()));
    auto first = delayed.relu(d);
    auto second = delayed.matmul(first, first);
    auto parallel = delayed.softmax(first);
    delayed.output(delayed.relu(second)); delayed.output(parallel);
    verify(delayed);
    std::cout << "Fusion planning tests passed\n";
}
