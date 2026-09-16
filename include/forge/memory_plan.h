#pragma once
#include "forge/graph.h"
#include <limits>

namespace forge {
struct MemoryStatistics {
    std::size_t tensor_bytes = 0;       // Sum of computed tensor sizes, without reuse/padding.
    std::size_t reserved_bytes = 0;     // Sum of allocated slot capacities, including padding.
    std::size_t peak_live_bytes = 0;    // Peak live computed payload, excluding padding.
    std::size_t allocation_count = 0;
    std::size_t reuse_count = 0;
};
struct MemoryPlan {
    static constexpr std::size_t no_slot = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> order;
    std::vector<std::size_t> slots;       // Per-node slot, or no_slot for borrowed inputs.
    std::vector<std::size_t> capacities;  // Bytes per owned slot.
    std::vector<std::size_t> first_use;   // Inclusive schedule position of production.
    std::vector<std::size_t> last_use;    // Inclusive; outputs extend past the last operation.
    MemoryStatistics statistics;
};
// Pure metadata planning; performs no CUDA calls or allocations of device storage.
// Slot reuse is safe only for the supplied graph's serial execution order.
MemoryPlan plan_memory(const Graph& graph, bool reuse = true, std::size_t alignment = 256);
} // namespace forge
