#include "forge/ops/relu.h"
#include "forge/ops/softmax.h"
#include <iostream>
#include <stdexcept>

namespace {
template<class Error, class F> void expect(F action) {
    try { action(); } catch (const Error&) { return; }
    throw std::runtime_error("Expected activation rejection");
}
}
int main() {
    using namespace forge;
    using Operation = void (*)(const Tensor&, Tensor&);
    float av[16]{}, bv[16]{};
    for (Operation op : {static_cast<Operation>(relu), static_cast<Operation>(softmax)}) {
        Tensor a({2, 3}, DataType::Float32, Device::cuda(), av);
        Tensor b({2, 3}, DataType::Float32, Device::cuda(), bv);
        Tensor shape({3, 2}, DataType::Float32, Device::cuda(), bv);
        expect<std::invalid_argument>([&] { op(a, shape); });
        Tensor dtype({2, 3}, DataType::Float16, Device::cuda(), av);
        expect<std::invalid_argument>([&] { op(dtype, b); });
        expect<std::invalid_argument>([&] { op(b, dtype); });
        Tensor cpu({2, 3}, DataType::Float32, Device::cpu(), av);
        expect<std::invalid_argument>([&] { op(cpu, b); });
        Tensor device({2, 3}, DataType::Float32, Device::cuda(1), av);
        expect<std::invalid_argument>([&] { op(device, b); });
        Tensor missing({2, 3}, DataType::Float32, Device::cuda());
        expect<std::invalid_argument>([&] { op(missing, b); });
        expect<std::invalid_argument>([&] { op(a, missing); });
        expect<std::invalid_argument>([&] { op(a, a); });
        Tensor overlap({2, 3}, DataType::Float32, Device::cuda(), av + 1);
        expect<std::invalid_argument>([&] { op(a, overlap); });
        expect<std::invalid_argument>([&] { op(overlap, a); });
        Tensor ints({2, 3}, DataType::Int32, Device::cuda(), av);
        Tensor int_out({2, 3}, DataType::Int32, Device::cuda(), bv);
        expect<std::invalid_argument>([&] { op(ints, int_out); });
#if !FORGE_TEST_CUDA
        Tensor half({2, 3}, DataType::Float16, Device::cuda(), av);
        Tensor half_out({2, 3}, DataType::Float16, Device::cuda(), bv);
        expect<std::runtime_error>([&] { op(half, half_out); });
        expect<std::runtime_error>([&] { op(a, b); });
#endif
    }
    Tensor scalar({}, DataType::Float32, Device::cuda(), av);
    Tensor result({}, DataType::Float32, Device::cuda(), bv);
    expect<std::invalid_argument>([&] { softmax(scalar, result); });
#if !FORGE_TEST_CUDA
    expect<std::runtime_error>([&] { relu(scalar, result); });
#endif
    std::cout << "Activation validation tests passed\n";
}
