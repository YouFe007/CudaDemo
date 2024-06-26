// scaled_dot_product_attention_module.cpp
#include <pybind11/pybind11.h>
#include <torch/extension.h>

// 声明来自其他源文件的函数
void scaled_dot_product_attention_torch_wrapper(
    const torch::Tensor queries,
    const torch::Tensor keys,
    torch::Tensor outputs,
    float scale);
void scaled_dot_product_attention_pure_c_wrapper(
    const float* h_queries,
    const float* h_keys,
    float* h_outputs,
    int batch_size, int seq_len, int d_k, float scale);

PYBIND11_MODULE(scaled_dot_product_attention_cuda, m) {
    m.def("scaled_dot_product_attention_torch", &scaled_dot_product_attention_torch_wrapper, "Torch verison scaled Dot Product Attention");
    m.def("scaled_dot_product_attention_pure_c", &scaled_dot_product_attention_pure_c_wrapper, "Pure c version scaled Dot Product Attention");
}