#include <torch/extension.h>

#include "conv2d_base_half.cu"
#include "conv2d_base.cu"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("conv2d", &conv2d, "Custom 2D Convolution (CUDA)",
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias"),
            py::arg("stride") = 1,
            py::arg("padding") = 0);

      m.def("conv2d_half", &conv2d_half, "Custom 2D Convolution (CUDA, Float32 input, Float16 compute)",
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias"),
            py::arg("stride") = 1,
            py::arg("padding") = 0);
}
