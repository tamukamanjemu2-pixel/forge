#pragma once
#include "forge/tensor.h"
#include <memory>

namespace forge {
class Runtime;
class Executable;
class Graph {
public:
    class Value {
    public:
        Value() = default;
    private:
        friend class Graph;
        friend class Executable;
        std::size_t index_ = 0;
        std::shared_ptr<const int> owner_;
    };
    enum class Operation { Input, MatMul, ReLU, Softmax, MatMulReLU, Elided };
    struct Node {
        Operation operation;
        std::vector<std::size_t> inputs;
        Tensor metadata;
    };
    Graph();
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) noexcept = default;
    Graph& operator=(Graph&&) noexcept = default;

    // Inputs borrow storage. Null storage permits metadata-only graph design but
    // cannot be compiled for execution. Keep bound storage alive during execution.
    Value input(const Tensor& tensor);
    Value matmul(Value a, Value b);
    Value relu(Value input);
    Value softmax(Value input);
    void output(Value value);
    const Tensor& descriptor(Value value) const;
    const std::vector<Node>& nodes() const noexcept { return nodes_; }
    const std::vector<std::size_t>& outputs() const noexcept { return outputs_; }
    // Validates the whole graph; deterministic topological order includes inputs.
    std::vector<std::size_t> execution_order() const;
private:
    friend class Runtime;
    std::size_t index(Value value) const;
    Value append(Operation operation, std::vector<std::size_t> inputs, Tensor metadata);
    std::shared_ptr<const int> owner_;
    std::vector<Node> nodes_;
    std::vector<std::size_t> outputs_;
};
} // namespace forge
