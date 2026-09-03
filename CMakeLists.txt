cmake_minimum_required(VERSION 3.24)

project(Forge
        VERSION 0.1.0
        LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default target for our current benchmark hardware:
# NVIDIA Tesla T4 = compute capability 7.5.
#
# Can be overridden later:
# cmake -DCMAKE_CUDA_ARCHITECTURES=...
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

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

target_link_libraries(
        forge_cuda
        PUBLIC
        CUDA::cudart
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
        forge_cuda
)

add_executable(
        forge_matmul_benchmark
        benchmarks/matmul_benchmark.cpp
)

target_link_libraries(
        forge_matmul_benchmark
        PRIVATE
        forge_ops
        forge_cuda
)