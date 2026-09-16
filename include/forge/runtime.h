#pragma once
#include "forge/graph.h"
#include "forge/memory_plan.h"
#include <memory>

namespace forge {
class Executable {
public:
    ~Executable();
    Executable(Executable&&) noexcept;
    Executable& operator=(Executable&&) noexcept;
    Executable(const Executable&) = delete;
    Executable& operator=(const Executable&) = delete;
    // Available after successful execution; only marked outputs can be retrieved.
    // Returned tensor borrows executable-owned storage (or a bound graph input).
    const Tensor& output(Graph::Value value) const;
    std::size_t allocated_bytes() const;
    const MemoryStatistics& memory_statistics() const;
private:
    friend class Runtime;
    struct Impl;
    explicit Executable(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

struct CompileOptions {
    bool reuse_memory = true;
};

class Runtime {
public:
    // Snapshots graph metadata; input storage is borrowed. Intermediate/output
    // storage slots are allocated here and reused by non-overlapping lifetimes.
    Executable compile(const Graph& graph) const;
    Executable compile(const Graph& graph, CompileOptions options) const;
    // Blocking whole-graph execution. One owned stream, one final synchronization.
    // Not thread-safe for concurrent calls using the same executable or its inputs.
    void execute(Executable& executable) const;
};
} // namespace forge
