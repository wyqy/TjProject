#pragma once
#include "cu_common.cuh"

// 命名空间
using std::max;

// 定义宏
#define CUDA_MAX_RESLEN ((size_t)128)
#define BLOCK_DIMONE_x1024 ((unsigned int)1024)

// 调用函数
__host__ cudaError_t initialCuda(int device);
__host__ cudaError_t sortWithCuda(float* retValue, size_t* retLen, const float* data, size_t len);
__host__ cudaError_t releaseCuda(void);

// 辅助Host Function
__host__ size_t paddingSize(size_t len);
__host__ size_t fastIntLog(size_t x);

// CUDA Kernel
__global__ void readKernel(float* retValue, const float* data);
__global__ void sumKernel(float* retValue, const float* dataA, const float* dataB);
