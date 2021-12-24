#pragma once
#include "pj_common.h"
#include "socket_common.h"
#include "cuda_sort.cuh"
#include "cuda_logsqrt.cuh"

// 调用SIMD指令库
#include <intrin.h>
// 调用OMP库
#include <omp.h>
// 调用系统库
#include <Windows.h>
// 调用STD STACK库
#include <stack>
using std::stack;

// 定义对齐宏
#define MEMORY_ALIGNED (size_t)0x20  // 0x20 for _mm256, 0x40 for _mm512
// 定义CPU并行宏
#define OMP_THREADS 12  // adjust!

// 定义全局数组
// 如果要测试, 取消下面四行的注释
// 测试
#undef DUAL_DATANUM
#undef SGLE_DATANUM
#define DUAL_DATANUM TEST_DUAL_DATANUM
#define SGLE_DATANUM TEST_SGLE_DATANUM
// 定义
extern float disorderData[DUAL_DATANUM];
extern float rawFloatData[DUAL_DATANUM];
extern float locFloatData[SGLE_DATANUM];
extern float rmtFloatData[SGLE_DATANUM];


// 定义快速排序对象
struct quickLoc
{  // 左闭右开
    size_t start;
    size_t end;
};

// 定义初始化函数
float init_random(float data[], const size_t len);
float init_arithmetic(float data[], float start, const size_t len);

// 定义功能函数
// 传统排序
float sortNaive(float data[], size_t len);
// 传统结果处理(将原数组变换为log(sqrt()))
float postProcess(float data[], size_t len);
// 加速排序
float sortSpeedup(float data[], size_t len);
// 加速结果处理
float postSpeedup(float data[], size_t len);
// 合并
float sortMerge(float dataRes[], float dataA[], size_t lenA, float dataB[], size_t lenB);
// 验证
bool validSort(const float data[], size_t len);

// 辅助函数
void atomFloatSwap(float* dataA, float* dataB);
void atomSizetSwap(size_t* dataA, size_t* dataB);
size_t quickFindIndex(float* data, size_t len);
size_t quickPartSort(float* data, size_t leftIndex, size_t rightIndex);
