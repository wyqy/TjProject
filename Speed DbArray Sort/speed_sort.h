#pragma once
#include "pj_common.h"
#include "socket_common.h"
#include "cuda_sort.cuh"

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
extern float rmtFloatData[SGLE_DATANUM];  // socket_remote

// �����ʼ������
float init_arithmetic(float data[], float start, const size_t len);

// ���幦�ܺ���
// ��ͳ
float sortNaive(const float data[], const size_t len);
// ����
float sortSpeedup(const float data[], const size_t len);
// ��֤
bool validSort(const float data[], const size_t len);
