#include "forge/graph_plan.h"
#include <algorithm>

namespace forge {
GraphPlan plan_graph(const Graph& graph, bool fuse_matmul_relu) {
    GraphPlan plan{graph.nodes(), graph.execution_order(), graph.outputs()};
    if (fuse_matmul_relu) {
        std::vector<std::size_t> consumers(plan.nodes.size(), 0);
        for (const auto& node : plan.nodes)
            for (auto input : node.inputs) ++consumers[input];
        for (auto i : plan.order) {
            auto& relu = plan.nodes[i];
            if (relu.operation != Graph::Operation::ReLU) continue;
            const auto producer = relu.inputs[0];
            auto& matmul = plan.nodes[producer];
            if (matmul.operation != Graph::Operation::MatMul || consumers[producer] != 1 ||
                std::find(plan.outputs.begin(), plan.outputs.end(), producer) != plan.outputs.end()) continue;
            relu.operation = Graph::Operation::MatMulReLU;
            relu.inputs = matmul.inputs;
            matmul.operation = Graph::Operation::Elided;
            matmul.inputs.clear();
            ++plan.fused_pairs;
        }
    }
    for (const auto& node : plan.nodes)
        if (node.operation != Graph::Operation::Input && node.operation != Graph::Operation::Elided)
            ++plan.kernel_launches;
    return plan;
}
}
