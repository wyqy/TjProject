#pragma once
#include "pj_common.h"
#include "socket_common.h"
#include "cuda_max.cuh"

// 调用SIMD指令库
#include <intrin.h>
// 调用OMP库
#include <omp.h>
// 调用系统库
#include <Windows.h>

// 定义对齐宏
#define MEMORY_ALIGNED (size_t)0x20  // 0x20 for _mm256, 0x40 for _mm512
// 定义CPU并行宏
#define OMP_THREADS 12  // adjust!

// 定义全局数组
extern float rawFloatData[DUAL_DATANUM];  // local
extern float locFloatData[SGLE_DATANUM];  // socket_local

// 定义初始化函数
float init_arithmetic(float data[], float start, const size_t len);

// 定义功能函数
// 传统
float maxNaive(const float data[], const size_t len);
// CUDA + AVX
float maxSpeedup_Cu(const float data[], const size_t len);
float maxWithAVX(const float* data, size_t len);
// OMP + AVX
float maxSpeedup_OP(const float data[], const size_t len);
float maxWithOMPAVX(const float* data, size_t len);
