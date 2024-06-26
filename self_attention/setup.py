from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='scaled_dot_product_attention',
    ext_modules=[
        CUDAExtension('scaled_dot_product_attention_cuda', [
            'scaled_dot_product_attention_kernel_with_torch.cu',
            'scaled_dot_product_attention_kernel_pure_c.cu',
            'scaled_dot_product_attention_kernel.cpp',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O0', '-G', '-lineinfo']}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)