#include <cuda_runtime.h>


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