import torch
import numpy as np
import time

import scaled_dot_product_attention_cuda


def scaled_dot_product_attention_numpy(queries, keys, scale):
    batch_size, seq_len, d_k = queries.shape
    outputs = np.empty((batch_size, seq_len, seq_len))
    for batch in range(batch_size):
        q = queries[batch]
        k = keys[batch].T
        outputs[batch] = np.matmul(q, k) * scale
    return outputs

# 假设的参数值
batch_size = 32
seq_len = 128
d_k = 64
scale = 0.1

# 初始化输入和输出张量
queries = torch.randn(batch_size, seq_len, d_k).cuda()
keys = torch.randn(batch_size, seq_len, d_k).cuda()
outputs = torch.empty(batch_size, seq_len, seq_len).cuda()

# 创建CUDA事件来记录时间
start_cuda = torch.cuda.Event(enable_timing=True)
end_cuda = torch.cuda.Event(enable_timing=True)


# 记录cuda函数开始时间
start_time = time.time()
# 调用CUDA扩展函数
scaled_dot_product_attention_cuda.scaled_dot_product_attention_torch(queries, keys, outputs, scale)
elapsed_time_cuda = (time.time() - start_time) * 1000  # 转换为毫秒

# 等待事件完成
torch.cuda.synchronize()

# 计算CUDA函数运行时间
print(f"CUDA function elapsed time: {elapsed_time_cuda} ms")

# 将张量转换为NumPy数组
queries_np = queries.cpu().numpy()
keys_np = keys.cpu().numpy()

# 记录NumPy函数开始时间
start_time_numpy = time.time()

# 调用NumPy函数
outputs_np = scaled_dot_product_attention_numpy(queries_np, keys_np, scale)

# 记录NumPy函数结束时间并计算运行时间
elapsed_time_numpy = (time.time() - start_time_numpy) * 1000  # 转换为毫秒
print(f"NumPy function elapsed time: {elapsed_time_numpy} ms")


# 创建随机数据作为输入
h_queries = torch.randn(batch_size, seq_len, d_k, dtype=torch.float32)
h_keys = torch.randn(batch_size, seq_len, d_k, dtype=torch.float32)

# 分配输出张量
h_outputs = torch.empty(batch_size, seq_len, seq_len, dtype=torch.float32)

# 确保张量在CPU上，并且是连续的，以便与C API兼容
h_queries = h_queries.contiguous()
h_keys = h_keys.contiguous()
h_outputs = h_outputs.contiguous()

# 调用CUDA扩展函数
start_time_numpy = time.time()

scaled_dot_product_attention_cuda.scaled_dot_product_attention_pure_c(
    h_queries.data_ptr(), h_keys.data_ptr(), h_outputs.data_ptr(), batch_size, seq_len, d_k, scale
)
elapsed_time_numpy = (time.time() - start_time_numpy) * 1000  # 转换为毫秒
print(f"NumPy function elapsed time: {elapsed_time_numpy} ms")
