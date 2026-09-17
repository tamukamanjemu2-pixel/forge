#include "forge/memory_plan.h"
#include "forge/graph_plan.h"
#include <algorithm>
#include <stdexcept>

namespace forge {
namespace {
std::size_t add(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("Graph memory plan size overflow");
    return a + b;
}
}
MemoryPlan plan_memory(const Graph& graph, bool reuse, std::size_t alignment, bool fuse_matmul_relu) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        throw std::invalid_argument("Memory plan alignment must be a power of two");
    MemoryPlan plan;
    const auto program = plan_graph(graph, fuse_matmul_relu);
    plan.order = program.order;
    const auto& nodes = program.nodes;
    const auto count = nodes.size();
    plan.slots.assign(count, MemoryPlan::no_slot);
    plan.first_use.resize(count);
    plan.last_use.resize(count);
    for (std::size_t step = 0; step < count; ++step) {
        const auto node = plan.order[step];
        plan.first_use[node] = plan.last_use[node] = step;
    }
    for (auto node : plan.order)
        for (auto dependency : nodes[node].inputs)
            plan.last_use[dependency] = std::max(plan.last_use[dependency], plan.first_use[node]);
    for (auto output : graph.outputs()) plan.last_use[output] = count;

    std::vector<std::size_t> slot_end;
    std::vector<std::size_t> release(count + 1, 0);
    std::size_t live = 0;
    for (std::size_t step = 0; step < count; ++step) {
        const auto node = plan.order[step];
        if (nodes[node].operation != Graph::Operation::Input && nodes[node].operation != Graph::Operation::Elided) {
            const auto bytes = nodes[node].metadata.nbytes();
            const auto padded = add(bytes, alignment - 1) & ~(alignment - 1);
            plan.statistics.tensor_bytes = add(plan.statistics.tensor_bytes, bytes);
            live = add(live, bytes);
            plan.statistics.peak_live_bytes = std::max(plan.statistics.peak_live_bytes, live);
            release[plan.last_use[node]] = add(release[plan.last_use[node]], bytes);
            auto chosen = MemoryPlan::no_slot;
            if (reuse) {
                for (std::size_t slot = 0; slot < slot_end.size(); ++slot) {
                    // Strict inequality: an operator's output cannot overwrite an input
                    // that is still being consumed by that same operator.
                    if (slot_end[slot] < step && plan.capacities[slot] >= padded &&
                        (chosen == MemoryPlan::no_slot || plan.capacities[slot] < plan.capacities[chosen]))
                        chosen = slot;
                }
            }
            if (chosen == MemoryPlan::no_slot) {
                chosen = plan.capacities.size();
                plan.capacities.push_back(padded);
                slot_end.push_back(plan.last_use[node]);
                plan.statistics.reserved_bytes = add(plan.statistics.reserved_bytes, padded);
            } else {
                slot_end[chosen] = plan.last_use[node];
                ++plan.statistics.reuse_count;
            }
            plan.slots[node] = chosen;
        }
        live -= release[step];
    }
    plan.statistics.allocation_count = plan.capacities.size();
    return plan;
}
} // namespace forge
