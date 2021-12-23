#include "speed_sum.h"

float init_arithmetic(float data[], float start, const size_t len)
{
    const intptr_t intlen = (intptr_t)len;

#pragma omp parallel for shared(data, len)
    for (intptr_t iter_series = 0; iter_series < intlen; iter_series++)
    {
        data[iter_series] = float(start + iter_series);
    }

    return data[len - 1];
}


float sumNaive(const float data[], const size_t len)
{
    float summaryData = 0, currentData = 0;

    // 由于单精度浮点数精度限制, 该实现结果不正确
    for (size_t iter_sum = 0; iter_sum < len; iter_sum++)
    {
        currentData = log(sqrt(data[iter_sum]));
        summaryData += currentData;
    }

    return summaryData;
}


float sumSpeedup_Cu(const float data[], const size_t len)
{
    size_t* retLen = new size_t;
    float* retValueCuda = (float*)_aligned_malloc(CUDA_MAX_RESLEN * sizeof(float), MEMORY_ALIGNED);  // aligned
    float retValue;

    // CUDA
    sumWithCuda(retValueCuda, retLen, data, (unsigned int)len);
    // CPU
    retValue = sumWithAVX(retValueCuda, *retLen);

    delete retLen;  // 释放堆
    _aligned_free(retValueCuda);  // 对应对齐
    return retValue;
}


float sumWithAVX(float* data, size_t len)
{
    // 计算
    intptr_t simd_speedup = 32 / sizeof(float);
    intptr_t m256_num_elems = len / simd_speedup;
    intptr_t rest_offest = (m256_num_elems - 1) * simd_speedup;
    intptr_t naive_num_elems = len - rest_offest;
    // 初始化
    float restSumData = 0.0f;

    // 加速对齐的完整元素
    __declspec(align(MEMORY_ALIGNED)) __m256 simd_sum_result;
    __declspec(align(MEMORY_ALIGNED)) __m256 simd_current_elem;
    __m256* simdPtr_now_elems = (__m256*) data;
    // 初始化
    simd_sum_result = _mm256_set1_ps(0);
    // 计算
    for (intptr_t iter_simd_max = 0; iter_simd_max < m256_num_elems; iter_simd_max++)
    {
        simd_current_elem = _mm256_load_ps((const float*)(simdPtr_now_elems + iter_simd_max));
        simd_sum_result = _mm256_add_ps(simd_sum_result, simd_current_elem);
    }
    // 写回(可直接写回!)
    _mm256_store_ps((float*)(simdPtr_now_elems + m256_num_elems - 1), simd_sum_result);

    // 计算剩下的元素
    for (intptr_t iter_rest = len - 1; iter_rest >= rest_offest; iter_rest--)
    {
        restSumData += data[iter_rest];
    }

    return restSumData;
}


float sumSpeedup_OP(const float data[], const size_t len)
{
    return sumWithOMPAVX(data, len);
}


float sumWithOMPAVX(const float* data, size_t len)
{
    // 计算size
    intptr_t simd_speedup = 32 / sizeof(float);
    intptr_t m256_num_elems = len / simd_speedup;
    intptr_t rest_offest = m256_num_elems * simd_speedup;
    intptr_t naive_num_elems = len - rest_offest + (OMP_THREADS * simd_speedup);
    // 初始化
    float restSumElem = 0.0f;
    float* restSumData;
    if (naive_num_elems > 0) restSumData = (float*)_aligned_malloc(naive_num_elems * sizeof(float), MEMORY_ALIGNED);
    else return restSumElem;
    if (restSumData != nullptr)
    {
        // 复制剩余元素
        memcpy_s(restSumData + (OMP_THREADS * simd_speedup), (len - (size_t)rest_offest) * sizeof(float),
            data + rest_offest, (len - (size_t)rest_offest) * sizeof(float));
        // 计算为log(sqrt())
        for (intptr_t iter_logsqrt = OMP_THREADS * simd_speedup; iter_logsqrt < naive_num_elems; iter_logsqrt++)
        {
            restSumData[iter_logsqrt] = log(sqrt(restSumData[iter_logsqrt]));
        }
    }
    else return restSumElem;

    // 加速对齐的完整元素
    __declspec(align(MEMORY_ALIGNED)) __m256 simd_sum_result;
    __declspec(align(MEMORY_ALIGNED)) __m256 simd_current_elem;
    __m256* simd_sum_array = (__m256*) restSumData;
    __m256* simdPtr_now_elems = (__m256*) data;
    // 初始化
    simd_sum_result = _mm256_set1_ps(0.0f);
    // 计算
    // 并行化
#pragma omp parallel for num_threads(OMP_THREADS) firstprivate(simd_sum_result) private(simd_current_elem) shared(simdPtr_now_elems, m256_num_elems)
    for (intptr_t iter_simd_max = 0; iter_simd_max < m256_num_elems; iter_simd_max++)
    {
        simd_current_elem = _mm256_load_ps((const float*)(simdPtr_now_elems + iter_simd_max));
        simd_current_elem = _mm256_log_ps(_mm256_sqrt_ps(simd_current_elem));
        simd_sum_result = _mm256_add_ps(simd_sum_result, simd_current_elem);
        // 写回(可否到最后一次再写回?)
        _mm256_store_ps((float*)(simd_sum_array + omp_get_thread_num()), simd_sum_result);
    }

    // 规约, 计算剩下的元素
    if (restSumData != nullptr && naive_num_elems > 0)
    {
        for (intptr_t iter_rest = 0; iter_rest < naive_num_elems; iter_rest++)
        {
            restSumElem += restSumData[iter_rest];
        }
        // 释放内存
        _aligned_free(restSumData);  // 对应对齐
    }
    return restSumElem;
}
