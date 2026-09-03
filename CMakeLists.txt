cmake_minimum_required(VERSION 3.24)

project(Forge
        VERSION 0.1.0
        LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# --------------------------------------------------
# Forge core runtime
# --------------------------------------------------

add_library(forge_core
        src/tensor.cpp
)

target_include_directories(
        forge_core
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# --------------------------------------------------
# CUDA backend
# --------------------------------------------------

add_library(forge_cuda
        cuda/vector_add.cu
        cuda/matmul.cu
)

target_include_directories(
        forge_cuda
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)

set_target_properties(
        forge_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# --------------------------------------------------
# Forge operators
# --------------------------------------------------

add_library(forge_ops
        src/ops/matmul.cpp
)

target_include_directories(
        forge_ops
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)

target_link_libraries(
        forge_ops
        PUBLIC
        forge_core
        PRIVATE
        forge_cuda
)

# --------------------------------------------------
# Benchmarks
# --------------------------------------------------

add_executable(
        forge_vector_add
        benchmarks/vector_add_benchmark.cpp
)

target_link_libraries(
        forge_vector_add
        PRIVATE
        forge_cuda
        CUDA::cudart
)

# --------------------------------------------------
# Tests
# --------------------------------------------------

add_executable(
        forge_tensor_test
        tests/tensor_test.cpp
)

target_link_libraries(
        forge_tensor_test
        PRIVATE
        forge_core
)

add_executable(
        forge_matmul_test
        tests/matmul_test.cpp
)

target_link_libraries(
        forge_matmul_test
        PRIVATE
        forge_ops
        CUDA::cudart
)