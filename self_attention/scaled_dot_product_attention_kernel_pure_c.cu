#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <utility>

// #include "scaled_dot_product_attention.h"

extern "C" __global__ void scaled_dot_product_attention(
    const float* queries, const float* keys, float* outputs,
    int batch_size, int seq_len, int d_k, float scale) {
        // 确定当前线程负责的元素位置
        int batch = blockIdx.x;
        int i = threadIdx.x;
        int j = threadIdx.y;

        // 确保线程索引在合法范围内
        if (i < seq_len && j < seq_len) {
            float sum = 0.0;
            for (int k = 0; k < d_k; ++k) {
                sum += queries[batch * seq_len * d_k + i * d_k + k] *
                    keys[batch * seq_len * d_k + j * d_k + k];
            }
            if ((batch == 0) && (threadIdx.x == 0 && threadIdx.y == 0)) {
                printf("Block %d\n", blockIdx.x);
                printf("sum %f\n", sum);
            }
            outputs[batch * seq_len * seq_len + i * seq_len + j] = sum * scale;
    }
}

void printDeviceProperties(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d: %s\n", device, deviceProp.name);
        printf("  Total amount of global memory: %.2f GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Max dimensions of block: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max dimensions of grid: (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
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

void scaled_dot_product_attention_pure_c_wrapper(
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
    scaled_dot_product_attention_pure_c_wrapper(h_queries, h_keys, h_outputs, batch_size, seq_len, d_k, scale);
    for (int i = 0; i < std::min(batch_size * seq_len * seq_len, 10); ++i) {
        std::cout << "h_outputs[" << i << "] = " << h_outputs[i] << std::endl;
    }
    printf("asd\n");

    // 使用 h_outputs
    // ...

    // 释放主机内存
    delete[] h_queries;
    delete[] h_keys;
    delete[] h_outputs;

    return 0;
}
