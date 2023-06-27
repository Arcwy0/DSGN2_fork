#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/ceil_div.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>
at::Tensor BuildGeometryVolume_forward_cuda(const at::Tensor &img,
                                        const at::Tensor &coord);

at::Tensor BuildGeometryVolume_backward_cuda(const at::Tensor &grad,
                                                                 const at::Tensor &coord,
                                                                 const int height,
                                                                 const int width);

// Interface for Python
at::Tensor BuildGeometryVolume_forward(const at::Tensor &img,
                                   const at::Tensor &coord)
{
  if (img.type().is_cuda())
  {
#ifdef WITH_CUDA
    return BuildGeometryVolume_forward_cuda(img, coord);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

at::Tensor BuildGeometryVolume_backward(const at::Tensor &grad,
                                                            const at::Tensor &coord,
                                                            const int height,
                                                            const int width)
{
  if (grad.type().is_cuda())
  {
#ifdef WITH_CUDA
    return BuildGeometryVolume_backward_cuda(grad, coord, height, width);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("Not implemented on the CPU");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("build_geometry_volume_forward", &BuildGeometryVolume_forward, "BuildGeometryVolume_forward");
  m.def("build_geometry_volume_backward", &BuildGeometryVolume_backward, "BuildGeometryVolume_backward");
}
