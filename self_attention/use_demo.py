import torch
import numpy as np
import time
import ctypes

import scaled_dot_product_attention_cuda


def scaled_dot_product_attention_numpy(queries, keys, scale):
    # 使用NumPy的广播和矩阵乘法来计算所有批次的输出
    # queries 的形状为 (batch_size, seq_len, d_k)
    # keys 的形状为 (batch_size, seq_len, d_k)
    # 我们需要对 keys 进行转置，以便进行矩阵乘法
    # 转置后 keys 的形状为 (batch_size, d_k, seq_len)
    # 然后执行批量矩阵乘法，得到 (batch_size, seq_len, seq_len) 形状的输出
    outputs = np.matmul(queries, keys.transpose(0, 2, 1)) * scale
    return outputs

# 假设的参数值
batch_size = 32
seq_len = 128
d_k = 64
scale = 0.1

# 初始化输入和输出张量
queries = torch.rand(batch_size, seq_len, d_k, device='cuda')
keys = torch.rand(batch_size, seq_len, d_k, device='cuda')
scale = 1 / torch.sqrt(torch.tensor(d_k, device='cuda'))
outputs = torch.empty(batch_size, seq_len, seq_len, device='cuda')


def scaled_dot_product_attention_torch(queries, keys, scale):
    outputs = torch.bmm(queries, keys.transpose(1, 2)) * scale
    return outputs

start_time_torch = time.time()
outputs_torch = scaled_dot_product_attention_torch(queries, keys, scale)
elapsed_time_torch = (time.time() - start_time_torch) * 1000
print(f"PyTorch function elapsed time: {elapsed_time_torch} ms")


start_cuda = torch.cuda.Event(enable_timing=True)
end_cuda = torch.cuda.Event(enable_timing=True)


start_time = time.time()
scale = 0.1
# 调用CUDA扩展函数
scaled_dot_product_attention_cuda.scaled_dot_product_attention_torch(queries, keys, outputs, scale)
elapsed_time_cuda = (time.time() - start_time) * 1000  # 转换为毫秒

# 等待事件完成
torch.cuda.synchronize()

# 计算CUDA函数运行时间
print(f"Torch enhanced CUDA function elapsed time: {elapsed_time_cuda} ms")

# 将张量转换为NumPy数组
queries_np = queries.cpu().numpy()
keys_np = keys.cpu().numpy()

# 记录NumPy函数开始时间
start_time_numpy = time.time()

# 调用NumPy函数
outputs_np = scaled_dot_product_attention_numpy(queries_np, keys_np, scale)

# 记录NumPy函数结束时间并计算运行时间
elapsed_time_numpy = (time.time() - start_time_numpy) * 1000
print(f"NumPy function elapsed time: {elapsed_time_numpy} ms")


queries = np.random.rand(batch_size, seq_len, d_k).astype(np.float32)
keys = np.random.rand(batch_size, seq_len, d_k).astype(np.float32)
outputs = np.empty((batch_size, seq_len, seq_len), dtype=np.float32)

# 调用CUDA扩展函数
start_time_numpy = time.time()

scaled_dot_product_attention_cuda.scaled_dot_product_attention_pure_c(
    queries,
    keys,
    outputs,
    batch_size, seq_len, d_k, scale
)
elapsed_time_numpy = (time.time() - start_time_numpy) * 1000 
print(f"pure c CUDA function elapsed time: {elapsed_time_numpy} ms")
