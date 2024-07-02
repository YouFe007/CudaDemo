#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

#include "scaled_dot_product_attention.h"

namespace py = pybind11;


int getCudaCoresPerSM(int major, int minor) {
    // 根据不同的架构，返回每个 SM 的 CUDA 核心数
    // 注意：这里的值可能需要根据最新的 GPU 和 CUDA 版本进行更新
    if (major == 1) { // Tesla
        return 8;
    } else if (major == 2) { // Fermi
        return 32;
    } else if (major == 3) { // Kepler
        return 192;
    } else if (major == 5) { // Maxwell
        return 128;
    } else if (major == 6) { // Pascal
        return (minor == 1 ? 128 : 64);
    } else if (major == 7) { // Volta & Turing
        return 64;
    } else if (major == 8) { // Ampere
        // 这里是一个假设的值，需要根据具体型号调整
        return 64;
    } else {
        // 未知架构
        return -1;
    }
}

void printDeviceProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
        std::cout << "  Total amount of global memory: " << deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "  Shared memory per multiprocessor: " << deviceProp.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;
        std::cout << "  L2 cache size: " << deviceProp.l2CacheSize / 1024.0 << " KB" << std::endl;
        std::cout << "  Constant memory size: " << deviceProp.totalConstMem / 1024.0 << " KB" << std::endl;
        std::cout << "  Registers per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "  Registers per multiprocessor: " << deviceProp.regsPerMultiprocessor << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "  Max threads per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "  Max dimensions of block: (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "  Max dimensions of grid: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "  Number of SMs: " << deviceProp.multiProcessorCount << std::endl;
        int cudaCoresPerSM = getCudaCoresPerSM(deviceProp.major, deviceProp.minor);
        if (cudaCoresPerSM > 0) {
            std::cout << "  CUDA Cores: " << cudaCoresPerSM * deviceProp.multiProcessorCount << std::endl;
        } else {
            std::cout << "  CUDA Cores: Unknown (compute capability " << deviceProp.major << "." << deviceProp.minor << ")" << std::endl;
        }
    }
}

void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

std::pair<dim3, dim3> calculateBestConfig(int seq_len, int batch_size) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int blockDimX = std::min(seq_len, deviceProp.maxThreadsDim[0]);
    int blockDimY = std::min(seq_len, deviceProp.maxThreadsDim[1]);

    // 确保块大小不超过设备的最大线程数
    while (blockDimX * blockDimY > maxThreadsPerBlock) {
        if (blockDimX > blockDimY) {
            blockDimX--;
        } else {
            blockDimY--;
        }
    }

    dim3 threads(blockDimX, blockDimY);
    dim3 blocks(batch_size);

    return std::make_pair(blocks, threads);
}

void scaled_dot_product_attention_pure_c(
    const float* h_queries,
    const float* h_keys,
    float* h_outputs,
    int batch_size, int seq_len, int d_k, float scale) {
    
    float *d_queries, *d_keys, *d_outputs;
    size_t size_queries = batch_size * seq_len * d_k * sizeof(float);
    size_t size_keys = size_queries; // Same size as queries
    size_t size_outputs = batch_size * seq_len * seq_len * sizeof(float);

    // 分配设备内存
    cudaError_t err;

    // 分配设备内存
    err = cudaMalloc(&d_queries, size_queries);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    checkCudaError(cudaMalloc(&d_keys, size_keys));
    checkCudaError(cudaMalloc(&d_outputs, size_outputs));

    // 将数据从主机复制到设备
    checkCudaError(cudaMemcpy(d_queries, h_queries, size_queries, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_keys, h_keys, size_keys, cudaMemcpyHostToDevice));

    // 计算 grid 和 block 的大小
    std::pair<dim3, dim3> config = calculateBestConfig(seq_len, batch_size);
    dim3 blocks = config.first;
    dim3 threads = config.second;
    printf("Blocks: x=%d, y=%d, z=%d\n", blocks.x, blocks.y, blocks.z);
    printf("Threads: x=%d, y=%d, z=%d\n", threads.x, threads.y, threads.z);

    // 调用 CUDA 核函数
    scaled_dot_product_attention<<<blocks, threads>>>(
        d_queries, d_keys, d_outputs, batch_size, seq_len, d_k, scale);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
    // 等待 CUDA 核函数完成
    checkCudaError(cudaDeviceSynchronize());

    // 将结果从设备复制回主机
    checkCudaError(cudaMemcpy(h_outputs, d_outputs, size_outputs, cudaMemcpyDeviceToHost));

    // 释放设备内存
    checkCudaError(cudaFree(d_queries));
    checkCudaError(cudaFree(d_keys));
    checkCudaError(cudaFree(d_outputs));
}

void scaled_dot_product_attention_pure_c_wrapper(
    py::array_t<float> queries,
    py::array_t<float> keys,
    py::array_t<float> outputs,
    int batch_size, int seq_len, int d_k, float scale) {
    // 获取 buffer_info 对象，它包含指向数据的指针和其他元信息
    py::buffer_info queries_buf = queries.request();
    py::buffer_info keys_buf = keys.request();
    py::buffer_info outputs_buf = outputs.request();

    // 获取指向实际数据的指针
    float* h_queries = static_cast<float*>(queries_buf.ptr);
    float* h_keys = static_cast<float*>(keys_buf.ptr);
    float* h_outputs = static_cast<float*>(outputs_buf.ptr);
    
    scaled_dot_product_attention_pure_c(h_queries, h_keys, h_outputs, batch_size, seq_len, d_k, scale);
}

int main() {
    printDeviceProperties();
    // 假设的参数值
    int batch_size = 32;
    int seq_len = 128;
    int d_k = 64;
    float scale = 0.1;

    // 分配主机内存
    float* h_queries = new float[batch_size * seq_len * d_k];
    float* h_keys = new float[batch_size * seq_len * d_k];
    float* h_outputs = new float[batch_size * seq_len * seq_len];

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // 初始化 h_queries 和 h_keys
    for (int i = 0; i < batch_size * seq_len * d_k; ++i) {
        h_queries[i] = dist(gen);
        h_keys[i] = dist(gen);
    }
    for (int i = 0; i < std::min(batch_size * seq_len * seq_len, 10); ++i) {
        if (h_queries[i] != 0)
            std::cout << "h_queries[" << i << "] = " << h_queries[i] << std::endl;
            std::cout << "h_keys[" << i << "] = " << h_keys[i] << std::endl;
    }

    // 调用包装器函数
    scaled_dot_product_attention_pure_c(h_queries, h_keys, h_outputs, batch_size, seq_len, d_k, scale);
    for (int i = 0; i < std::min(batch_size * seq_len * seq_len, 10); ++i) {
        std::cout << "h_outputs[" << i << "] = " << h_outputs[i] << std::endl;
    }
    // 使用 h_outputs
    // ...

    // 释放主机内存
    delete[] h_queries;
    delete[] h_keys;
    delete[] h_outputs;
    printf("end\n");

    return 0;
}
