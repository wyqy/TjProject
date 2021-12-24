#pragma once
#include "cu_common.cuh"

// �����ռ�
using std::max;

// �����
#define BLOCK_DIMONE_x1024 ((unsigned int)1024)

// ���ú���
__host__ cudaError_t logsqrtWithCuda(float* data, size_t len);

// CUDA Kernel
__global__ void readKernel(float* data);
