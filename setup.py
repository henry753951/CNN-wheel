from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='conv2d_cuda',
    ext_modules=[
        CUDAExtension(
            name='conv2d_cuda',
            sources=[
                'src/cuda/conv2d_cuda.cpp',
                'src/cuda/conv2d_kernel.cu'
            ],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)