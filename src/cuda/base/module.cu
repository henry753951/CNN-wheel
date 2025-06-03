#include <torch/extension.h>
#include "conv2d_base.cu"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("conv2d", &conv2d, "Custom 2D Convolution (CUDA)",
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias"),
            py::arg("stride") = 1,
            py::arg("padding") = 0);
}
