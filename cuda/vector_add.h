#pragma once

void launch_vector_add(
    const float* d_a,
    const float* d_b,
    float* d_c,
    int n,
    int threads_per_block
);