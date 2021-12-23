#pragma once
#include "cu_common.cuh"

// �����ռ�
using std::max;

// �����
#define CUDA_MAX_RESLEN ((size_t)128)
#define BLOCK_DIMONE_x1024 ((unsigned int)1024)

// ���ú���
__host__ cudaError_t initialCuda(int device);
__host__ cudaError_t sumWithCuda(float* retValue, size_t* retLen, const float* data, size_t len);
__host__ cudaError_t releaseCuda(void);

// CUDA Kernel
__global__ void readKernel(float* retValue, const float* data);
__global__ void sumKernel(float* retValue, const float* dataA, const float* dataB);
