#pragma once
#include "forge/graph.h"
namespace forge {
struct GraphPlan {
    std::vector<Graph::Node> nodes;
    std::vector<std::size_t> order;
    std::vector<std::size_t> outputs;
    std::size_t fused_pairs = 0;
    std::size_t kernel_launches = 0;
};
// Metadata-only snapshot. Fuse only a MatMul with exactly one ReLU consumer,
// provided the unactivated MatMul is not a marked graph output.
GraphPlan plan_graph(const Graph& graph, bool fuse_matmul_relu = false);
}
