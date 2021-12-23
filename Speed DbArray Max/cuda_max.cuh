#pragma once
#include "cu_common.cuh"

// 命名空间
using std::max;

// 定义宏
#define CUDA_MAX_RESLEN ((size_t)256)
#define BLOCK_DIMONE_x1024 ((unsigned int)1024)

// 调用函数
__host__ cudaError_t initialCuda(int device);
__host__ cudaError_t maxWithCuda(float* retValue, size_t* retLen, const float* data, size_t len);
__host__ cudaError_t releaseCuda(void);

// CUDA Kernel
__global__ void readKernel(float* retValue, const float* data);
__global__ void maxKernel(float* retValue, const float* dataA, const float* dataB);
