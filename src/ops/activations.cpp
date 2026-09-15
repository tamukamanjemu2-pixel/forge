#include "forge/ops/relu.h"
#include "forge/ops/softmax.h"
#include "forge/stream.h"
#include <stdexcept>
#ifdef FORGE_HAS_CUDA
#include "activations.h"
#include "../cuda_support.h"
#endif
namespace forge {
namespace {
void dispatch(const Tensor& input, Tensor& output, const Stream* stream, bool normalize) {
    if (input.dtype() != DataType::Float32 || output.dtype() != DataType::Float32)
        throw std::invalid_argument("Activation requires Float32 tensors");
    if (input.shape() != output.shape())
        throw std::invalid_argument("Activation input/output shapes must match");
    if (input.device().type != DeviceType::CUDA || input.device() != output.device())
        throw std::invalid_argument("Activation requires tensors on the same CUDA device");
    if (!input.is_contiguous() || !output.is_contiguous())
        throw std::invalid_argument("Activation requires contiguous tensors");
    if (normalize && input.ndim() == 0)
        throw std::invalid_argument("Softmax requires rank of at least one");
    if (!input.data() || !output.data())
        throw std::invalid_argument("Activation requires storage");
    const auto a = reinterpret_cast<std::uintptr_t>(input.data());
    const auto b = reinterpret_cast<std::uintptr_t>(output.data());
    if ((a > b ? a - b : b - a) < input.nbytes())
        throw std::invalid_argument("Activation input and output must not overlap");
    if (stream && stream->device() != input.device())
        throw std::invalid_argument("Activation stream device mismatch");
#ifdef FORGE_HAS_CUDA
    detail::DeviceScope scope(input.device().index);
    auto handle = stream ? static_cast<cudaStream_t>(stream->native_handle()) : nullptr;
    if (normalize) {
        const auto columns = static_cast<std::size_t>(input.shape().back());
        launch_softmax(static_cast<const float*>(input.data()), static_cast<float*>(output.data()),
                       input.numel() / columns, columns, handle);
    } else {
        launch_relu(static_cast<const float*>(input.data()), static_cast<float*>(output.data()), input.numel(), handle);
    }
#else
    throw std::runtime_error("CUDA activations are unavailable: build with FORGE_ENABLE_CUDA=ON");
#endif
}
}
void relu(const Tensor& input, Tensor& output) { dispatch(input, output, nullptr, false); }
void relu(const Tensor& input, Tensor& output, const Stream& stream) { dispatch(input, output, &stream, false); }
void softmax(const Tensor& input, Tensor& output) { dispatch(input, output, nullptr, true); }
void softmax(const Tensor& input, Tensor& output, const Stream& stream) { dispatch(input, output, &stream, true); }
} // namespace forge
