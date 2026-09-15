#pragma once
#include "forge/tensor.h"
namespace forge {
class Stream;
// Float32 CUDA, matching contiguous input/output; overlapping storage is rejected.
// Normalizes the last dimension; rank must be at least one. Finite inputs required.
void softmax(const Tensor& input, Tensor& output);
void softmax(const Tensor& input, Tensor& output, const Stream& stream);
} // namespace forge
