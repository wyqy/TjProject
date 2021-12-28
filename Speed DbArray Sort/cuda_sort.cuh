#pragma once
#include "cu_common.cuh"

// 命名空间
using std::max;

// 定义宏
#define BLOCK_DIMONE_xLOG ((unsigned int)7)
#define BLOCK_DIMONE_x128 ((unsigned int)128)
#define WRAP_DIMLOG ((unsigned int)5)
#define WRAP_DIM ((unsigned int)32)

// 调用函数
__host__ cudaError_t initialCuda(int device, float* arrRaw, size_t lenRaw, float* arrLoc, size_t lenLoc);
__host__ cudaError_t sortWithCuda(float* data, size_t len);
__host__ cudaError_t releaseCuda(void);

// 辅助Host Function
__host__ size_t paddingSize(size_t len);
__host__ size_t fastIntLog(size_t x);

// 排序调用
__host__ void bitonicSort(float* data, unsigned int len);

// CUDA Kernel
__global__ void sortKernel(float* data, unsigned int iter_i, unsigned int iter_j);

