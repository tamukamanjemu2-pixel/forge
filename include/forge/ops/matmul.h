#pragma once

#include "forge/tensor.h"

namespace forge {

    void matmul(
        const Tensor& a,
        const Tensor& b,
        Tensor& output
    );

} // namespace forge