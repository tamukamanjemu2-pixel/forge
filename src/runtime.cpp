#include "forge/runtime.h"
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
    std::size_t bytes = 0;
    bool completed = false;
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
    return impl_->bytes;
}
Executable Runtime::compile(const Graph& graph) const {
    auto order = graph.execution_order();
    std::size_t bytes = 0;
    for (const auto& node : graph.nodes_) {
        if (node.operation == Graph::Operation::Input) {
            if (!node.metadata.data()) throw std::invalid_argument("Graph input has no bound storage");
        } else {
            if (node.metadata.nbytes() > std::numeric_limits<std::size_t>::max() - bytes)
                throw std::overflow_error("Graph allocation size overflow");
            bytes += node.metadata.nbytes();
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
    impl->nodes = graph.nodes_;
    impl->order = std::move(order);
    impl->outputs = graph.outputs_;
    impl->bytes = bytes;
    impl->tensors.reserve(graph.nodes_.size());
    impl->storage.reserve(graph.nodes_.size());
    for (const auto& node : graph.nodes_) {
        if (node.operation == Graph::Operation::Input) impl->tensors.push_back(node.metadata);
        else {
            impl->storage.emplace_back(node.metadata.nbytes(), node.metadata.device());
            impl->tensors.push_back(impl->storage.back().view(node.metadata.shape(), node.metadata.dtype()));
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
                case Graph::Operation::Input: break;
                case Graph::Operation::MatMul:
                    matmul(plan.tensors[node.inputs[0]], plan.tensors[node.inputs[1]], plan.tensors[i], plan.stream);
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
