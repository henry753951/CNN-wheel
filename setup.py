import os
import subprocess
import sys

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


class BuildExtensionWithStubs(BuildExtension):
    def run(self):
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
        ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld_path = f"{torch_lib_path}{os.pathsep}{ld_library_path}"
        os.environ["LD_LIBRARY_PATH"] = new_ld_path
        print(f"Temporarily setting LD_LIBRARY_PATH to: {new_ld_path}")
        super().run()
        for ext in self.extensions:
            print(f"Generating stubs for {ext.name}...")
            build_dir = os.path.abspath(self.get_ext_fullpath(ext.name).replace(self.get_ext_filename(ext.name), ""))
            python_path = os.environ.get("PYTHONPATH", "")
            stub_env_path = f"{build_dir}{os.pathsep}{python_path}"
            stub_env = os.environ.copy()
            stub_env["PYTHONPATH"] = stub_env_path

            cmd = [
                sys.executable,
                "-m",
                "pybind11_stubgen",
                ext.name,
                "-o",
                ".",
            ]

            print(f"Running command: {' '.join(cmd)}")
            print(f"With PYTHONPATH: {stub_env_path}")

            try:
                subprocess.run(cmd, check=True, env=stub_env)
            except subprocess.CalledProcessError as e:
                print(f"Failed to generate stubs for {ext.name}: {e}", file=sys.stderr)
            except FileNotFoundError:
                print(
                    "Error: 'pybind11-stubgen' not found. Please install it with 'pip install pybind11-stubgen'",
                    file=sys.stderr,
                )


os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Enable CUDA DSA (Dynamic Parallelism) for better performance
EXTRA_COMPILE_ARGS = {
    "cxx": ["-O2", "-std=c++17"],
    "nvcc": ["-O2", "--expt-relaxed-constexpr", "--use_fast_math"],
}

setup(
    name="conv2d_cuda",
    version="0.1",
    ext_modules=[
        CUDAExtension(
            name="custom_cnn.cuda._base",  # 對應 custom_cnn.cuda._base
            sources=["src/cuda/base/module.cu"],
            extra_compile_args=EXTRA_COMPILE_ARGS,
        ),
        # CUDAExtension(
        #     name="custom_cnn.cuda._fft",  # 對應 custom_cnn.cuda._fft
        #     sources=["src/cuda/fft/conv2d_fft.cu"],
        #     extra_compile_args=EXTRA_COMPILE_ARGS,
        # ),
        # CUDAExtension(
        #     name="custom_cnn.cuda._img2col",  # 對應 custom_cnn.cuda._img2col
        #     sources=["src/cuda/img2col/conv2d_img2col.cu"],
        #     extra_compile_args=EXTRA_COMPILE_ARGS,
        # ),
        CppExtension(
            name="custom_cnn.cpu._base",  # 對應 custom_cnn.cpu._base
            sources=["src/cpu/base/conv2d_base.cpp"],
            extra_compile_args=EXTRA_COMPILE_ARGS,
        ),
    ],
    cmdclass={"build_ext": BuildExtensionWithStubs},
    package_data={
        "": ["*.pyi"],
    },
    install_requires=["torch"],
)
