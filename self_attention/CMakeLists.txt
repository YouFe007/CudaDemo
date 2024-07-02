cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(my_cuda_project LANGUAGES CXX CUDA)

# 指定 CUDA 架构（如果需要）
set(CMAKE_CUDA_ARCHITECTURES 75) # 根据您的 GPU 调整这个值

# 添加包含目录
include_directories(${CMAKE_SOURCE_DIR})

# 指定项目的源文件
set(SOURCE_FILES
    scaled_dot_product_attention_kernel_pure_c.cu
    kernel.cu
)

set(CONDA_ENV_PATH "/home/felix/miniconda3/envs/local_mnr388")
set(PYBIND11_INCLUDE_DIR  "${CONDA_ENV_PATH}/lib/python3.8/site-packages/torch/include")
set(PYTHON_INCLUDE_DIR "${CONDA_ENV_PATH}/include/python3.8")
set(PYTHON_LIBRARY_DIR "${CONDA_ENV_PATH}/lib")

# 包含 pybind11 和 Python 头文件
include_directories(${PYBIND11_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR})

# 链接 Python 库
link_directories(${PYTHON_LIBRARY_DIR})

# 添加可执行文件
add_executable(use_demo ${SOURCE_FILES})

# 指定链接库（如果有）
# target_link_libraries(use_demo PRIVATE some_library)

# 设置 C++ 标准
set_property(TARGET use_demo PROPERTY CXX_STANDARD 14)

# 设置 CUDA 标准
set_property(TARGET use_demo PROPERTY CUDA_STANDARD 14)

# 如果需要，可以指定编译选项
target_compile_options(use_demo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-g -G>)

find_library(PYTHON_LIB python3.8 PATHS ${PYTHON_LIBRARY_DIR} NO_DEFAULT_PATH)
if(PYTHON_LIB)
    target_link_libraries(use_demo PRIVATE ${PYTHON_LIB})
else()
    message(FATAL_ERROR "Python library not found")
endif()