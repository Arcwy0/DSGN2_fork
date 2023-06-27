#ifndef _STACK_VOXEL_QUERY_GPU_H
#define _STACK_VOXEL_QUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/ceil_div.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

int voxel_query_wrapper_stack(int M, int R1, int R2, int R3, int nsample, float radius, 
    int z_range, int y_range, int x_range, at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, 
    at::Tensor new_coords_tensor, at::Tensor point_indices_tensor, at::Tensor idx_tensor);


void voxel_query_kernel_launcher_stack(int M, int R1, int R2, int R3, int nsample,
    float radius, int z_range, int y_range, int x_range, const float *new_xyz, 
    const float *xyz, const int *new_coords, const int *point_indices, int *idx);


#endif
