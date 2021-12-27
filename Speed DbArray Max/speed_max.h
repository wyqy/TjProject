#pragma once
#include "pj_common.h"
#include "socket_common.h"
#include "cuda_max.cuh"

// ����SIMDָ���
#include <intrin.h>
// ����OMP��
#include <omp.h>
// ����ϵͳ��
#include <Windows.h>

// ��������
#define MEMORY_ALIGNED (size_t)0x20  // 0x20 for _mm256, 0x40 for _mm512
// ����CPU���к�
#define OMP_THREADS 12  // adjust!

// ����ȫ������
extern float rawFloatData[DUAL_DATANUM];  // local
extern float locFloatData[SGLE_DATANUM];  // socket_local

// �����ʼ������
float init_arithmetic(float data[], float start, const size_t len);

// ���幦�ܺ���
// ��ͳ
float maxNaive(const float data[], const size_t len);
// CUDA + AVX
float maxSpeedup_Cu(const float data[], const size_t len);
float maxWithAVX(const float* data, size_t len);
// OMP + AVX
float maxSpeedup_OP(const float data[], const size_t len);
float maxWithOMPAVX(const float* data, size_t len);
