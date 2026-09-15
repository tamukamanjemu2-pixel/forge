#pragma once
#include "forge/tensor.h"
namespace forge {
class Stream;
// Float32 CUDA, matching contiguous input/output; overlapping storage is rejected.
// Any rank, including scalars. NaNs propagate.
void relu(const Tensor& input, Tensor& output);
void relu(const Tensor& input, Tensor& output, const Stream& stream);
} // namespace forge
