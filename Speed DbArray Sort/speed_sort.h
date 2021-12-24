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
// ����STD STACK��
#include <stack>
using std::stack;

// ��������
#define MEMORY_ALIGNED (size_t)0x20  // 0x20 for _mm256, 0x40 for _mm512
// ����CPU���к�
#define OMP_THREADS 12  // adjust!

// ����ȫ������
// ���Ҫ����, ȡ���������е�ע��
// ����
#undef DUAL_DATANUM
#undef SGLE_DATANUM
#define DUAL_DATANUM TEST_DUAL_DATANUM
#define SGLE_DATANUM TEST_SGLE_DATANUM
// ����
extern float rawFloatData[DUAL_DATANUM];
extern float locFloatData[SGLE_DATANUM];
extern float rmtFloatData[SGLE_DATANUM];


// ��������������
struct quickLoc
{  // ����ҿ�
    size_t start;
    size_t end;
};

// �����ʼ������
float init_random(float data[], const size_t len);
float init_arithmetic(float data[], float start, const size_t len);

// ���幦�ܺ���
// ��ͳ
float sortNaive(float data[], size_t len);
// ����
float sortSpeedup(float data[], size_t len);
// �ϲ�
float sortMerge(float dataRes[], float dataA[], size_t lenA, float dataB[], size_t lenB);
// �������(��ԭ����任Ϊlog(sqrt()))
float postProcess(float data[], size_t len);
// ��֤
bool validSort(const float data[], size_t len);

// ��������
void atomFloatSwap(float* dataA, float* dataB);
void atomSizetSwap(size_t* dataA, size_t* dataB);
size_t quickFindIndex(float* data, size_t len);
size_t quickPartSort(float* data, size_t leftIndex, size_t rightIndex);
