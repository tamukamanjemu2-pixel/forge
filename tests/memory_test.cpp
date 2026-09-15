#include "forge/memory.h"
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class Error, class F> void expect(F action) {
    try { action(); } catch (const Error&) { return; }
    throw std::runtime_error("Expected exception was not thrown");
}
}

int main() {
    using namespace forge;
    static_assert(!std::is_copy_constructible_v<Buffer>);
    static_assert(std::is_nothrow_move_constructible_v<Buffer>);
    Buffer storage(64);
    check(reinterpret_cast<std::uintptr_t>(storage.data()) % 64 == 0, "CPU alignment");
    auto a = storage.view({4}, DataType::Int32);
    auto b = storage.view({4}, DataType::Int32, 16);
    auto* values = static_cast<std::int32_t*>(a.data());
    for (int i = 0; i < 4; ++i) values[i] = i * 3 - 4;
    copy_tensor(a, b);
    for (int i = 0; i < 4; ++i)
        check(static_cast<std::int32_t*>(b.data())[i] == i * 3 - 4, "CPU copy");
    auto overlapping = storage.view({4}, DataType::Int32, 4);
    copy_tensor(a, overlapping);
    for (int i = 0; i < 4; ++i)
        check(static_cast<std::int32_t*>(overlapping.data())[i] == i * 3 - 4, "overlap copy");
    copy_tensor(overlapping, overlapping);
    const auto* pointer = storage.data();
    Buffer moved(std::move(storage));
    check(moved.data() == pointer && storage.data() == nullptr && storage.nbytes() == 0, "move construction");
    check(a.data() == moved.data(), "view survives ownership transfer");
    expect<std::logic_error>([&] { storage.view({1}, DataType::Float32); });
    Buffer assigned(8);
    assigned = std::move(moved);
    check(assigned.data() == pointer && moved.data() == nullptr, "move assignment");
    auto& alias = assigned;
    assigned = std::move(alias);
    check(assigned.data() == pointer, "self move");
    expect<std::invalid_argument>([] { Buffer empty(0); });
    expect<std::invalid_argument>([] { Buffer bad(4, {DeviceType::CPU, 1}); });
    expect<std::invalid_argument>([&] { assigned.view({1}, DataType::Float32, 1); });
    expect<std::out_of_range>([&] { assigned.view({17}, DataType::Float32); });
    expect<std::out_of_range>([&] { assigned.view({1}, DataType::Float16, 64); });
    expect<std::out_of_range>([&] {
        assigned.view({1}, DataType::Float16, std::numeric_limits<std::size_t>::max() - 1);
    });
    auto wrong_shape = assigned.view({2, 2}, DataType::Int32);
    expect<std::invalid_argument>([&] { copy_tensor(a, wrong_shape); });
    auto wrong_type = assigned.view({4}, DataType::Float32);
    expect<std::invalid_argument>([&] { copy_tensor(a, wrong_type); });
    Tensor missing({4}, DataType::Int32, Device::cpu());
    expect<std::invalid_argument>([&] { copy_tensor(missing, b); });
    for (auto dtype : {DataType::Float16, DataType::Float32, DataType::Int32}) {
        auto scalar = assigned.view({}, dtype);
        check(scalar.nbytes() == Tensor::element_size(dtype), "scalar view");
    }
#if !FORGE_TEST_CUDA
    expect<std::runtime_error>([] { Buffer gpu(4, Device::cuda()); });
    // Borrowed descriptor only: a host-only build must reject before dereference.
    Tensor gpu({4}, DataType::Int32, Device::cuda(), a.data());
    expect<std::runtime_error>([&] { copy_tensor(gpu, b); });
#endif
    std::cout << "Memory tests passed\n";
}
