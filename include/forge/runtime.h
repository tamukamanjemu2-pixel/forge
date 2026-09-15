#pragma once
#include "forge/graph.h"
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
private:
    friend class Runtime;
    struct Impl;
    explicit Executable(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

class Runtime {
public:
    // Snapshots graph metadata; input storage is borrowed. Intermediate/output
    // storage is allocated once here and reused across executions, without pooling.
    Executable compile(const Graph& graph) const;
    // Blocking whole-graph execution. One owned stream, one final synchronization.
    // Not thread-safe for concurrent calls using the same executable or its inputs.
    void execute(Executable& executable) const;
};
} // namespace forge
