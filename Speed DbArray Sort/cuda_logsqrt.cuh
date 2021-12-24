#pragma once
#include "cu_common.cuh"

// 命名空间
using std::max;

// 定义宏
#define BLOCK_DIMONE_x1024 ((unsigned int)1024)

// 调用函数
__host__ cudaError_t logsqrtWithCuda(float* data, size_t len);

// CUDA Kernel
__global__ void readKernel(float* data);
