#include "forge/ops/matmul.h"
#include "forge/stream.h"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
template<class Error, class F> void expect(F action, const char* name) {
    try { action(); } catch (const Error&) { return; }
    throw std::runtime_error(std::string("Expected rejection: ") + name);
}
}
int main() {
    using namespace forge;
    // Host storage only stands in for borrowed CUDA addresses during validation.
    // Every call below must reject before dispatch or touching CUDA.
    float av[16]{}, bv[16]{}, cv[16]{};
    Tensor a({2, 3}, DataType::Float32, Device::cuda(), av);
    Tensor b({3, 2}, DataType::Float32, Device::cuda(), bv);
    Tensor c({2, 2}, DataType::Float32, Device::cuda(), cv);
    expect<std::invalid_argument>([&] { matmul(a, b, c, static_cast<MatMulKernel>(99)); }, "unknown kernel");
    expect<std::invalid_argument>([&] { matmul_relu(a, b, c, static_cast<MatMulKernel>(99)); }, "unknown fused kernel");
    Tensor rank({6}, DataType::Float32, Device::cuda(), av);
    expect<std::invalid_argument>([&] { matmul(rank, b, c); }, "rank");
    Tensor dtype({2, 3}, DataType::Float16, Device::cuda(), av);
    expect<std::invalid_argument>([&] { matmul(dtype, b, c); }, "dtype");
    Tensor cpu({2, 3}, DataType::Float32, Device::cpu(), av);
    expect<std::invalid_argument>([&] { matmul(cpu, b, c); }, "CPU");
    Tensor device({2, 3}, DataType::Float32, Device::cuda(1), av);
    expect<std::invalid_argument>([&] { matmul(device, b, c); }, "device mismatch");
    Tensor inner({4, 2}, DataType::Float32, Device::cuda(), bv);
    expect<std::invalid_argument>([&] { matmul(a, inner, c); }, "inner dimension");
    Tensor output({2, 3}, DataType::Float32, Device::cuda(), cv);
    expect<std::invalid_argument>([&] { matmul(a, b, output); }, "output shape");
    Tensor missing({2, 3}, DataType::Float32, Device::cuda());
    expect<std::invalid_argument>([&] { matmul(missing, b, c); }, "missing storage");
    Tensor overlap({2, 2}, DataType::Float32, Device::cuda(), av + 1);
    expect<std::invalid_argument>([&] { matmul(a, b, overlap); }, "partial output overlap");
    overlap.set_data(bv);
    expect<std::invalid_argument>([&] { matmul(a, b, overlap); }, "output aliases B");
    const auto huge = static_cast<std::int64_t>(std::numeric_limits<int>::max()) + 1;
    Tensor huge_a({huge, 1}, DataType::Float32, Device::cuda(), av);
    Tensor one({1, 1}, DataType::Float32, Device::cuda(), bv);
    Tensor huge_c({huge, 1}, DataType::Float32, Device::cuda(), cv);
    expect<std::overflow_error>([&] { matmul(huge_a, one, huge_c); }, "kernel dimension limit");
    expect<std::invalid_argument>([] { Stream stream(Device::cpu()); }, "CPU stream");
    expect<std::invalid_argument>([] { Stream stream({DeviceType::CUDA, -1}); }, "invalid stream device");
    Tensor ia({2, 3}, DataType::Int32, Device::cuda(), av);
    Tensor ib({3, 2}, DataType::Int32, Device::cuda(), bv);
    Tensor ic({2, 2}, DataType::Int32, Device::cuda(), cv);
    expect<std::invalid_argument>([&] { matmul(ia, ib, ic); }, "Int32 dispatch");
#if !FORGE_TEST_CUDA
    Tensor ha({2, 3}, DataType::Float16, Device::cuda(), av);
    Tensor hb({3, 2}, DataType::Float16, Device::cuda(), bv);
    Tensor hc({2, 2}, DataType::Float16, Device::cuda(), cv);
    expect<std::runtime_error>([&] { matmul(ha, hb, hc); }, "host-only half");
    expect<std::runtime_error>([] { Stream stream; }, "host-only stream");
    expect<std::runtime_error>([&] { matmul_relu(a, b, c); }, "host-only fused dispatch");
    expect<std::runtime_error>([&] { matmul(a, b, c); }, "host-only dispatch");
    expect<std::runtime_error>([&] { matmul(a, b, c, MatMulKernel::Tiled); }, "host-only tiled dispatch");
#endif
    std::cout << "Operator validation tests passed\n";
}
