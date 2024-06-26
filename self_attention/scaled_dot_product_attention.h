// scaled_dot_product_attention.h
#ifndef SCALED_DOT_PRODUCT_ATTENTION_H
#define SCALED_DOT_PRODUCT_ATTENTION_H

extern "C" {
__global__ void scaled_dot_product_attention(
    const float* queries, const float* keys, float* outputs,
    int batch_size, int seq_len, int d_k, float scale);
}

#endif // SCALED_DOT_PRODUCT_ATTENTION_H