from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

EXTRA_COMPILE_ARGS = {
    "cxx": ["-O2", "-std=c++17"],
    "nvcc": ["-O2", "--expt-relaxed-constexpr", "--use_fast_math"],
}

setup(
    name="conv2d_cuda",
    packages=["custom_cnn", "custom_cnn.cuda", "custom_cnn.cpu"],
    version="0.1",
    ext_modules=[
        CUDAExtension(
            name="custom_cnn.cuda._base",  # 對應 custom_cnn.cuda._base
            sources=["src/cnn/cuda/base/conv2d_base.cu"],
            extra_compile_args=EXTRA_COMPILE_ARGS,
        ),
        # CUDAExtension(
        #     name="custom_cnn.cuda._fft",  # 對應 custom_cnn.cuda._fft
        #     sources=["src/cnn/cuda/fft/conv2d_fft.cu"],
        #     extra_compile_args=EXTRA_COMPILE_ARGS,
        # ),
        # CUDAExtension(
        #     name="custom_cnn.cuda._img2col",  # 對應 custom_cnn.cuda._img2col
        #     sources=["src/cnn/cuda/img2col/conv2d_img2col.cu"],
        #     extra_compile_args=EXTRA_COMPILE_ARGS,
        # ),
        # CppExtension(
        #     name="custom_cnn.cpu._base",  # 對應 custom_cnn.cpu._base
        #     sources=["src/cnn/cpu/base/conv2d_base.cpp"],
        #     extra_compile_args=EXTRA_COMPILE_ARGS,
        # ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
