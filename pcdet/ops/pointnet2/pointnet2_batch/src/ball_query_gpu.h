#ifndef _BALL_QUERY_GPU_H
#define _BALL_QUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/ceil_div.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/DeviceUtils.cuh>
int ball_query_wrapper_fast(int b, int n, int m, float radius, int nsample, 
	at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor);

void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, 
	const float *xyz, const float *new_xyz, int *idx);

#endif
