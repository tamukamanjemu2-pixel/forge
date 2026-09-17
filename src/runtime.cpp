#include "forge/runtime.h"
#include "forge/graph_plan.h"
#include "forge/memory.h"
#include "forge/stream.h"
#include "forge/ops/matmul.h"
#include "forge/ops/relu.h"
#include "forge/ops/softmax.h"
#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace forge {
struct Executable::Impl {
    explicit Impl(Device device) : stream(device) {}
    Stream stream;
    std::shared_ptr<const int> owner;
    std::vector<Graph::Node> nodes;
    std::vector<std::size_t> order, outputs;
    std::vector<Buffer> storage;
    std::vector<Tensor> tensors;
    MemoryStatistics statistics;
    bool completed = false;
    std::size_t fused_pairs = 0, kernel_launches = 0;
    MatMulKernel kernel = MatMulKernel::Naive;
};
Executable::Executable(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Executable::~Executable() = default;
Executable::Executable(Executable&&) noexcept = default;
Executable& Executable::operator=(Executable&&) noexcept = default;
const Tensor& Executable::output(Graph::Value value) const {
    if (!impl_) throw std::logic_error("Executable was moved from");
    if (value.owner_ != impl_->owner ||
        std::find(impl_->outputs.begin(), impl_->outputs.end(), value.index_) == impl_->outputs.end())
        throw std::invalid_argument("Value is not an output of this executable");
    if (!impl_->completed) throw std::logic_error("Executable has no completed result");
    return impl_->tensors[value.index_];
}
std::size_t Executable::allocated_bytes() const {
    if (!impl_) throw std::logic_error("Executable was moved from");
    return impl_->statistics.reserved_bytes;
}
const MemoryStatistics& Executable::memory_statistics() const {
    if (!impl_) throw std::logic_error("Executable was moved from");
    return impl_->statistics;
}
std::size_t Executable::fused_pairs() const {
    if (!impl_) throw std::logic_error("Executable was moved from");
    return impl_->fused_pairs;
}
std::size_t Executable::planned_kernel_launches() const {
    if (!impl_) throw std::logic_error("Executable was moved from");
    return impl_->kernel_launches;
}
Executable Runtime::compile(const Graph& graph) const { return compile(graph, {}); }
Executable Runtime::compile(const Graph& graph, CompileOptions options) const {
    if (options.matmul_kernel != MatMulKernel::Naive && options.matmul_kernel != MatMulKernel::Tiled)
        throw std::invalid_argument("Unknown graph MatMul kernel");
    auto program = plan_graph(graph, options.fuse_matmul_relu);
    auto memory = plan_memory(graph, options.reuse_memory, 256, options.fuse_matmul_relu);
    for (const auto& node : graph.nodes_) {
        if (node.operation == Graph::Operation::Input) {
            if (!node.metadata.data()) throw std::invalid_argument("Graph input has no bound storage");
        }
        if (node.operation == Graph::Operation::MatMul) {
            const auto& a = graph.nodes_[node.inputs[0]].metadata;
            const auto& b = graph.nodes_[node.inputs[1]].metadata;
            constexpr auto limit = std::numeric_limits<int>::max();
            if (a.shape()[0] > limit || a.shape()[1] > limit || b.shape()[1] > limit)
                throw std::overflow_error("Graph MatMul exceeds kernel dimension limits");
        }
    }
    auto impl = std::make_unique<Executable::Impl>(graph.nodes_.front().metadata.device());
    impl->owner = graph.owner_;
    impl->kernel = options.matmul_kernel;
    impl->nodes = std::move(program.nodes);
    impl->fused_pairs = program.fused_pairs;
    impl->kernel_launches = program.kernel_launches;
    impl->order = std::move(memory.order);
    impl->outputs = graph.outputs_;
    impl->statistics = memory.statistics;
    impl->tensors.reserve(graph.nodes_.size());
    impl->storage.reserve(memory.capacities.size());
    for (auto capacity : memory.capacities)
        impl->storage.emplace_back(capacity, graph.nodes_.front().metadata.device());
    for (std::size_t i = 0; i < graph.nodes_.size(); ++i) {
        const auto& node = impl->nodes[i];
        if (node.operation == Graph::Operation::Input || node.operation == Graph::Operation::Elided) impl->tensors.push_back(node.metadata);
        else {
            impl->tensors.push_back(impl->storage[memory.slots[i]].view(node.metadata.shape(), node.metadata.dtype()));
        }
    }
    return Executable(std::move(impl));
}
void Runtime::execute(Executable& executable) const {
    if (!executable.impl_) throw std::logic_error("Executable was moved from");
    auto& plan = *executable.impl_;
    plan.completed = false;
    try {
        for (auto i : plan.order) {
            const auto& node = plan.nodes[i];
            switch (node.operation) {
                case Graph::Operation::Input:
                case Graph::Operation::Elided: break;
                case Graph::Operation::MatMul:
                    matmul(plan.tensors[node.inputs[0]], plan.tensors[node.inputs[1]], plan.tensors[i], plan.stream, plan.kernel);
                    break;
                case Graph::Operation::MatMulReLU:
                    matmul_relu(plan.tensors[node.inputs[0]], plan.tensors[node.inputs[1]], plan.tensors[i], plan.stream, plan.kernel);
                    break;
                case Graph::Operation::ReLU:
                    relu(plan.tensors[node.inputs[0]], plan.tensors[i], plan.stream); break;
                case Graph::Operation::Softmax:
                    softmax(plan.tensors[node.inputs[0]], plan.tensors[i], plan.stream); break;
            }
        }
        plan.stream.synchronize();
        plan.completed = true;
    } catch (...) {
        // Drain any earlier launches before callers release borrowed input storage.
        const auto original = std::current_exception();
        plan.stream.synchronize(); // If draining fails, propagate that CUDA failure.
        std::rethrow_exception(original);
    }
}
} // namespace forge
