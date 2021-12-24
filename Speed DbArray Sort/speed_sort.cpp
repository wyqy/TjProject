#include "speed_sort.h"


float init_random(float data[], const size_t len)
{
    // 顺序shuffle, 随机生成交换数的index
    random_device rdev;  // 随机数生成设备
    mt19937 rengine(rdev());  // 随机数生成引擎
    uniform_int_distribution<size_t> rdistrib(0, len);  // 随机分布, 左闭右开
    size_t shuffleLoc = 0;

    for (size_t iter_rand = 0; iter_rand < len - 1; iter_rand++)
    {
        rdistrib.param(uniform_int<size_t>::param_type(iter_rand + 1, len));
        shuffleLoc = rdistrib(rengine);
        atomFloatSwap(data + iter_rand, data + shuffleLoc);
    }

    return (float)0;
}


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


float sortNaive(float data[], const size_t len)
{
    // 传统排序
    // 使用快速排序, 由于栈的限制, 使用非递归版本, 本质还是用栈(STL)

    size_t midIndex = 0;
    quickLoc putQuickLoc = { 0, len - 1 };
    quickLoc getQuickLoc = { 0, 0 };
    stack<quickLoc> recursiveStack;  // 默认使用deque实现

    // 初始入栈
    recursiveStack.push(putQuickLoc);  // 内部使用push_back, 属于值复制!

    while (!recursiveStack.empty())
    {
        // 出栈
        getQuickLoc = recursiveStack.top();  // 取值
        recursiveStack.pop();  // 弹出
        
        // 排序
        midIndex = quickPartSort(data, getQuickLoc.start, getQuickLoc.end);

        // 入栈
        if (midIndex > getQuickLoc.start + 1)
        {
            putQuickLoc = { getQuickLoc.start, midIndex - 1 };
            recursiveStack.push(putQuickLoc);
        }
        if (midIndex < getQuickLoc.end - 1)
        {
            putQuickLoc = { midIndex + 1, getQuickLoc.end };
            recursiveStack.push(putQuickLoc);
        }
    }

    return data[len - 1];
}


float sortSpeedup(float data[], const size_t len)
{
    // 使用基于CUDA的双调排序?
    return 0;
}


float sortMerge(float dataRes[], const float dataA[], const size_t lenA, const float dataB[], const size_t lenB)
{
    // 基于归并合并两个有序数组
    return 0;
}


float postProcess(float data[], const size_t len)
{
    // 简单并行
    const intptr_t intlen = (intptr_t)len;

#pragma omp parallel for shared(data, intlen)
    for (intptr_t iter_series = 0; iter_series < intlen; iter_series++)
    {
        data[iter_series] = log(sqrt(data[iter_series]));
    }

    return data[len - 1];
}


bool validSort(const float data[], const size_t len)
{
    size_t summary = 0;
    const intptr_t intlen = (intptr_t)len - 1;  // 减少一个元素
    size_t* unvalidCount = new size_t[OMP_THREADS];
    memset(unvalidCount, 0, OMP_THREADS * sizeof(size_t));

    // 默认为升序排列
#pragma omp parallel for num_threads(OMP_THREADS) shared(data, intlen)
    for (intptr_t iter_series = 0; iter_series < intlen; iter_series++)
    {
        if (data[iter_series] > data[iter_series + 1]) unvalidCount[omp_get_thread_num()] += 1;
    }

    // 归并
    for (size_t iter_sum = 0; iter_sum < OMP_THREADS; iter_sum++)
    {
        summary += unvalidCount[iter_sum];
    }

    return summary == 0;
}


inline void atomFloatSwap(float* dataA, float* dataB)
{
    float temp;
    temp = *dataA;
    *dataA = *dataB;
    *dataB = temp;
}


void atomSizetSwap(size_t* dataA, size_t* dataB)
{
    size_t temp;
    temp = *dataA;
    *dataA = *dataB;
    *dataB = temp;
}


size_t quickFindIndex(float* data, size_t len)
{
    float startVal = data[0], midVal = data[len / 2], endVal = data[len - 1];
    if ((startVal - midVal) * (startVal - endVal) < 0) return 0;
    if ((midVal - startVal) * (midVal - endVal) < 0) return len / 2;
    if ((endVal - startVal) * (endVal - midVal) < 0) return len - 1;

    return 0;
}


inline size_t quickPartSort(float* data, size_t leftIndex, size_t rightIndex)
{
    // 下面的方法必须指定第一位数据为基准
    float midValue = data[leftIndex];
    float midReadValue = log(sqrt(midValue));

    // 在中间停止
    while (leftIndex < rightIndex)
    {
        // 上升顺序
        // 右指针向左查找, 直到左右相碰或所指元素不满足要求停止
        while (leftIndex < rightIndex && log(sqrt(data[rightIndex])) >= midReadValue)
        {
            rightIndex--;
        }
        data[leftIndex] = data[rightIndex];  // 右指针元素填入左侧相应位置

        // 左指针向右查找, 直到左右相碰或所指元素不满足要求停止
        while (leftIndex < rightIndex && log(sqrt(data[leftIndex])) <= midReadValue)
        {
            leftIndex++;
        }
        data[rightIndex] = data[leftIndex];  // 左指针元素填入右侧相应位置
    }

    // 将中间元素填入相应位置
    data[leftIndex] = midValue;

    return leftIndex;  // 返回位置
}
