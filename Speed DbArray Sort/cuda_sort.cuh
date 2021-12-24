#pragma once
#include "cu_common.cuh"

// �����ռ�
using std::max;

// �����
#define CUDA_MAX_RESLEN ((size_t)128)
#define BLOCK_DIMONE_x512 ((unsigned int)512)
#define WRAP_DIM ((unsigned int)32)

// ���ú���
__host__ cudaError_t initialCuda(int device, float* arrRaw, size_t lenRaw, float* arrLoc, size_t lenLoc);
__host__ cudaError_t sortWithCuda(float* data, size_t len);
__host__ cudaError_t releaseCuda(void);

// ����Host Function
__host__ size_t paddingSize(size_t len);
__host__ size_t fastIntLog(size_t x);

// �������
__host__ void bitonicSort(float* data, unsigned int len, cudaError_t* cudaRetValue);

// CUDA Kernel
__global__ void sortKernel(float* data, unsigned int iter_i, unsigned int iter_j);

