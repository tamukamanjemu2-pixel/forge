#include "forge/graph.h"
#include <algorithm>
#include <queue>
#include <stdexcept>
#include <utility>

namespace forge {
Graph::Graph() : owner_(std::make_shared<const int>(0)) {}
std::size_t Graph::index(Value value) const {
    if (!owner_ || value.owner_ != owner_ || value.index_ >= nodes_.size())
        throw std::invalid_argument("Value does not belong to this graph");
    return value.index_;
}
Graph::Value Graph::append(Operation operation, std::vector<std::size_t> inputs, Tensor metadata) {
    if (!owner_) throw std::logic_error("Graph was moved from");
    nodes_.push_back({operation, std::move(inputs), std::move(metadata)});
    Value value;
    value.index_ = nodes_.size() - 1;
    value.owner_ = owner_;
    return value;
}
Graph::Value Graph::input(const Tensor& tensor) {
    if ((tensor.dtype() != DataType::Float32 && tensor.dtype() != DataType::Float16) || tensor.device().type != DeviceType::CUDA)
        throw std::invalid_argument("Graph inputs require Float32 or Float16 CUDA descriptors");
    if (!nodes_.empty() && tensor.device() != nodes_.front().metadata.device())
        throw std::invalid_argument("All graph values must use one CUDA device");
    return append(Operation::Input, {}, tensor);
}
Graph::Value Graph::matmul(Value a, Value b) {
    const auto ai = index(a), bi = index(b);
    const auto& left = nodes_[ai].metadata;
    const auto& right = nodes_[bi].metadata;
    if (left.dtype() != right.dtype()) throw std::invalid_argument("Graph MatMul dtype mismatch");
    if (left.ndim() != 2 || right.ndim() != 2 || left.shape()[1] != right.shape()[0])
        throw std::invalid_argument("Graph MatMul requires compatible matrices");
    Tensor result({left.shape()[0], right.shape()[1]}, left.dtype(), left.device());
    return append(Operation::MatMul, {ai, bi}, std::move(result));
}
Graph::Value Graph::relu(Value input_value) {
    const auto i = index(input_value);
    const auto& source = nodes_[i].metadata;
    return append(Operation::ReLU, {i}, Tensor(source.shape(), source.dtype(), source.device()));
}
Graph::Value Graph::softmax(Value input_value) {
    const auto i = index(input_value);
    const auto& source = nodes_[i].metadata;
    if (source.ndim() == 0) throw std::invalid_argument("Graph Softmax requires rank >= 1");
    return append(Operation::Softmax, {i}, Tensor(source.shape(), source.dtype(), source.device()));
}
void Graph::output(Value value) {
    const auto i = index(value);
    if (std::find(outputs_.begin(), outputs_.end(), i) == outputs_.end()) outputs_.push_back(i);
}
const Tensor& Graph::descriptor(Value value) const { return nodes_[index(value)].metadata; }
std::vector<std::size_t> Graph::execution_order() const {
    if (!owner_ || nodes_.empty() || outputs_.empty())
        throw std::invalid_argument("Graph requires inputs and at least one marked output");
    std::vector<std::size_t> indegree(nodes_.size(), 0);
    std::vector<std::vector<std::size_t>> consumers(nodes_.size());
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        for (auto dependency : nodes_[i].inputs) {
            if (dependency >= nodes_.size()) throw std::invalid_argument("Invalid graph dependency");
            ++indegree[i];
            consumers[dependency].push_back(i);
        }
    }
    std::queue<std::size_t> ready;
    for (std::size_t i = 0; i < nodes_.size(); ++i) if (!indegree[i]) ready.push(i);
    std::vector<std::size_t> order;
    while (!ready.empty()) {
        const auto i = ready.front(); ready.pop(); order.push_back(i);
        for (auto consumer : consumers[i]) if (--indegree[consumer] == 0) ready.push(consumer);
    }
    if (order.size() != nodes_.size()) throw std::invalid_argument("Graph contains a cycle");
    return order;
}
} // namespace forge
