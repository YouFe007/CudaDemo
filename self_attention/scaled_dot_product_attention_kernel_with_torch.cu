// scaled_dot_product_attention_kernel.cu
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "scaled_dot_product_attention.h"

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
            outputs[batch * seq_len * seq_len + i * seq_len + j] = sum * scale;
    }
}

void scaled_dot_product_attention_torch_wrapper(
    const torch::Tensor queries,
    const torch::Tensor keys,
    torch::Tensor outputs,
    float scale) {
    
    const int batch_size = queries.size(0);
    const int seq_len = queries.size(1);
    const int d_k = queries.size(2);

    // 计算 grid 和 block 的大小
    dim3 blocks(batch_size);
    dim3 threads(seq_len, seq_len);

    // 调用 CUDA 核函数
    scaled_dot_product_attention<<<blocks, threads>>>(
        queries.data_ptr<float>(),
        keys.data_ptr<float>(),
        outputs.data_ptr<float>(),
        batch_size, seq_len, d_k, scale);

    // 检查 CUDA 错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // 处理错误
    }
}

