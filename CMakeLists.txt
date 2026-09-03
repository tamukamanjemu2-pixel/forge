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

add_library(forge_cuda
        cuda/vector_add.cu
)

target_include_directories(forge_cuda
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda
)

set_target_properties(
        forge_cuda
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

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