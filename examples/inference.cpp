#include "forge/runtime.h"
#include "forge/memory.h"
#include <iostream>
#include <vector>
int main() {
    using namespace forge;
    std::vector<float> xv{1, -2, 3, 0, 1, -1};
    std::vector<float> wv1{1, -1, 0, 2, 0, 1, 1, -1, 1, 0, -1, 1};
    std::vector<float> wv2{1, -1, 0, 1, 1, 0, -1, 1};
    Buffer xs(xv.size() * sizeof(float), Device::cuda());
    Buffer ws1(wv1.size() * sizeof(float), Device::cuda());
    Buffer ws2(wv2.size() * sizeof(float), Device::cuda());
    auto x = xs.view({2, 3}, DataType::Float32);
    auto w1 = ws1.view({3, 4}, DataType::Float32);
    auto w2 = ws2.view({4, 2}, DataType::Float32);
    Tensor hx({2, 3}, DataType::Float32, Device::cpu(), xv.data());
    Tensor hw1({3, 4}, DataType::Float32, Device::cpu(), wv1.data());
    Tensor hw2({4, 2}, DataType::Float32, Device::cpu(), wv2.data());
    copy_tensor(hx, x); copy_tensor(hw1, w1); copy_tensor(hw2, w2);
    Graph graph;
    auto input = graph.input(x);
    auto weights1 = graph.input(w1);
    auto weights2 = graph.input(w2);
    auto hidden = graph.relu(graph.matmul(input, weights1));
    auto output = graph.softmax(graph.matmul(hidden, weights2));
    graph.output(output);
    Runtime runtime;
    auto executable = runtime.compile(graph);
    runtime.execute(executable);
    std::vector<float> probabilities(4);
    Tensor host_output({2, 2}, DataType::Float32, Device::cpu(), probabilities.data());
    copy_tensor(executable.output(output), host_output);
    for (int row = 0; row < 2; ++row)
        std::cout << "Row " << row << ": " << probabilities[row * 2] << ' ' << probabilities[row * 2 + 1] << '\n';
    std::cout << "Reused tensor assignments: " << executable.memory_statistics().reuse_count << '\n';
    std::cout << "Reserved tensor storage bytes: " << executable.allocated_bytes() << '\n';
}
