#include "speed_max.h"

float init_arithmetic(float data[], float start, const size_t len)
{
    const intptr_t intlen = (intptr_t)len;
#pragma omp parallel for shared(data, intlen)
    for (intptr_t iter_series = 0; iter_series < intlen; iter_series++)
    {
        data[iter_series] = float(start + iter_series);
    }

    return data[len - 1];
}


float maxNaive(const float data[], const size_t len)
{
    float lastMaxData = 0, currentData = 0;

    for (size_t iter_max = 0; iter_max < len; iter_max++)
    {
        currentData = data[iter_max];
        currentData = log(sqrt(currentData));

        if (currentData > lastMaxData) lastMaxData = currentData;
    }

    return lastMaxData;
}


float maxSpeedup_Cu(const float data[], const size_t len)
{
    size_t* retLen = new size_t;
    float* retValueCuda = (float*)_aligned_malloc(CUDA_MAX_RESLEN * sizeof(float), MEMORY_ALIGNED);  // aligned
    float retValue;

    // CUDA
    maxWithCuda(retValueCuda, retLen, data, (unsigned int)len);
    // CPU
    retValue = maxWithAVX(retValueCuda, *retLen);

    delete retLen;  // 释放堆
    _aligned_free(retValueCuda);  // 对应对齐
    return retValue;
}


float maxWithAVX(const float* data, size_t len)
{
    // 计算
    intptr_t simd_speedup = 32 / sizeof(float);
    intptr_t m256_num_elems = len / simd_speedup;
    intptr_t rest_offest = m256_num_elems * simd_speedup;
    intptr_t naive_num_elems = len - rest_offest;
    // 初始化
    float lastRestMaxData = 1e-3f, currentRestData = 0;

    // 加速对齐的完整元素
    __declspec(align(MEMORY_ALIGNED)) __m256 simd_last_max_elem;
    __declspec(align(MEMORY_ALIGNED)) __m256 simd_current_elem;
    __m256* simdPtr_now_elems = (__m256*) data;
    // 初始化
    simd_last_max_elem = _mm256_set1_ps(0);
    // 计算
    for (intptr_t iter_simd_max = 0; iter_simd_max < m256_num_elems; iter_simd_max++)
    {
        simd_current_elem = _mm256_load_ps((const float*)(simdPtr_now_elems + iter_simd_max));
        // simd_current_elem = _mm256_log_ps(_mm256_sqrt_ps(simd_current_elem));
        simd_last_max_elem = _mm256_max_ps(simd_last_max_elem, simd_current_elem);
    }
    // 写回
    _mm256_store_ps((float*)(simdPtr_now_elems + m256_num_elems - 1), simd_last_max_elem);

    // 计算剩下的元素
    for (size_t iter_rest_max = rest_offest - simd_speedup; iter_rest_max < len; iter_rest_max++)
    {
        currentRestData = data[iter_rest_max];
        // currentRestData = log(sqrt(currentRestData));
        if (currentRestData > lastRestMaxData) lastRestMaxData = currentRestData;
    }
    
    return lastRestMaxData;
}


float maxSpeedup_OP(const float data[], const size_t len)
{
    return maxWithOMPAVX(data, len);
}


float maxWithOMPAVX(const float* data, size_t len)
{
    // 计算size
    intptr_t simd_speedup = 32 / sizeof(float);
    intptr_t m256_num_elems = len / simd_speedup;
    intptr_t rest_offest = m256_num_elems * simd_speedup;
    intptr_t naive_num_elems = len - rest_offest + (OMP_THREADS * simd_speedup);
    // 初始化
    float lastRestMaxData = 1e-3f, currentRestData = 0;

    float* restMaxData;
    if (naive_num_elems > 0) restMaxData = (float*)_aligned_malloc(naive_num_elems * sizeof(float), MEMORY_ALIGNED);
    else return lastRestMaxData;
    if (restMaxData != nullptr)  // 防止编译器报错
    {
        // 复制剩余元素
        memcpy_s(restMaxData + (OMP_THREADS * simd_speedup), (len - (size_t)rest_offest) * sizeof(float),
            data + rest_offest, (len - (size_t)rest_offest) * sizeof(float));
        // 计算为log(sqrt())
        for (intptr_t iter_logsqrt = OMP_THREADS * simd_speedup; iter_logsqrt < naive_num_elems; iter_logsqrt++)
        {
            restMaxData[iter_logsqrt] = log(sqrt(restMaxData[iter_logsqrt]));
        }
    }
    else return lastRestMaxData;

    // 加速对齐的完整元素
    __declspec(align(MEMORY_ALIGNED)) __m256 simd_last_max_elem;
    __declspec(align(MEMORY_ALIGNED)) __m256 simd_current_elem;
    __m256* simd_max_array = (__m256*) restMaxData;
    __m256* simdPtr_now_elems = (__m256*) data;
    // 初始化
    simd_last_max_elem = _mm256_set1_ps(0);
    // 计算
    // 并行化
#pragma omp parallel for num_threads(OMP_THREADS) firstprivate(simd_last_max_elem) private(simd_current_elem) shared(simdPtr_now_elems, m256_num_elems, simd_max_array)
    for (intptr_t iter_simd_max = 0; iter_simd_max < m256_num_elems; iter_simd_max++)
    {
        simd_current_elem = _mm256_load_ps((const float*)(simdPtr_now_elems + iter_simd_max));
        simd_current_elem = _mm256_log_ps(_mm256_sqrt_ps(simd_current_elem));
        simd_last_max_elem = _mm256_max_ps(simd_last_max_elem, simd_current_elem);
        // 写回(可否到最后一次再写回?)
        _mm256_store_ps((float*)(simd_max_array + omp_get_thread_num()), simd_last_max_elem);
    }

    // 计算剩下的元素, 归并法
    if (restMaxData != nullptr && naive_num_elems > 0)
    {
        for (intptr_t iter_rest_max = 0; iter_rest_max < naive_num_elems; iter_rest_max++)
        {
            currentRestData = restMaxData[iter_rest_max];
            // currentRestData = log(sqrt(currentRestData));
            if (currentRestData > lastRestMaxData) lastRestMaxData = currentRestData;
        }

        _aligned_free(restMaxData);  // 释放内存, 对应对齐
    }

    return lastRestMaxData;
}
