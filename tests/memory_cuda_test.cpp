#include "forge/memory.h"
#include "../src/cuda_support.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>

int main() {
    using namespace forge;
    int count = 0;
    detail::cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count == 0) throw std::runtime_error("GPU test requires an NVIDIA device");
    detail::cuda_check(cudaSetDevice(0), "cudaSetDevice");
    for (auto dtype : {DataType::Float32, DataType::Float16, DataType::Int32}) {
        Buffer host(128), readback(128);
        {
            Buffer gpu(128, Device::cuda()), second(128, Device::cuda());
            auto input = host.view({32}, dtype);
            auto output = readback.view({32}, dtype);
            auto device_input = gpu.view({32}, dtype);
            auto device_output = second.view({32}, dtype);
            std::memset(input.data(), 0x35, input.nbytes());
            copy_tensor(input, device_input);
            copy_tensor(device_input, device_output);
            copy_tensor(device_output, output);
            if (std::memcmp(input.data(), output.data(), input.nbytes()) != 0)
                throw std::runtime_error("CUDA roundtrip changed tensor bytes");
            bool rejected = false;
            try { copy_tensor(device_input, device_input); }
            catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) throw std::runtime_error("CUDA overlap was not rejected");
            Buffer moved(std::move(gpu));
            copy_tensor(device_input, output);
            if (count > 1) {
                detail::cuda_check(cudaSetDevice(1), "cudaSetDevice(1)");
                Buffer allocated_on_zero(16, Device::cuda(0));
                int after_allocation = -1;
                detail::cuda_check(cudaGetDevice(&after_allocation), "cudaGetDevice after allocation");
                if (after_allocation != 1) throw std::runtime_error("Allocation failed to restore device");
                copy_tensor(device_input, output);
                int current = -1;
                detail::cuda_check(cudaGetDevice(&current), "cudaGetDevice");
                if (current != 1) throw std::runtime_error("Copy failed to restore current device");
                Buffer other(128, Device::cuda(1));
                auto other_view = other.view({32}, dtype);
                rejected = false;
                try { copy_tensor(device_input, other_view); }
                catch (const std::invalid_argument&) { rejected = true; }
                if (!rejected) throw std::runtime_error("Cross-GPU copy was not rejected");
            }
        }
        int current = -1;
        detail::cuda_check(cudaGetDevice(&current), "cudaGetDevice after destruction");
        if (current != (count > 1 ? 1 : 0))
            throw std::runtime_error("Destruction changed current device");
        detail::cuda_check(cudaSetDevice(0), "cudaSetDevice(0)");
    }
    std::cout << "CUDA memory tests passed\n";
}
