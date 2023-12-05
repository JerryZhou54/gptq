from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='lutgemm_cuda',
    ext_modules=[cpp_extension.CUDAExtension(
        'lutgemm_cuda', ['lutGemm_cuda.cpp', 'lutGemm_cuda_kernel.cu']
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)
