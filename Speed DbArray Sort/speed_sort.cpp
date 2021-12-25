#include "speed_sort.h"


float init_random(float data[], const size_t len)
{
    // ˳��shuffle, ������ɽ�������index
    random_device rdev;  // ����������豸
    mt19937 rengine(rdev());  // �������������
    uniform_int_distribution<size_t> rdistrib(0, len - 1);  // ����ֲ�, ����ұ�?
    size_t shuffleLoc = 0;

    for (size_t iter_rand = 0; iter_rand < len - 1; iter_rand++)
    {
        rdistrib.param(uniform_int<size_t>::param_type(iter_rand + 1, len - 1));
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
    // ��ͳ����
    // ʹ�ÿ�������, ����ջ������, ʹ�÷ǵݹ�汾, ���ʻ�����ջ(STL)

    size_t midIndex = 0;
    quickLoc putQuickLoc = { 0, len - 1 };
    quickLoc getQuickLoc = { 0, 0 };
    stack<quickLoc> recursiveStack;  // Ĭ��ʹ��dequeʵ��

    // ��ʼ��ջ
    recursiveStack.push(putQuickLoc);  // �ڲ�ʹ��push_back, ����ֵ����!

    while (!recursiveStack.empty())
    {
        // ��ջ
        getQuickLoc = recursiveStack.top();  // ȡֵ
        recursiveStack.pop();  // ����
        
        // ����
        midIndex = quickPartSort(data, getQuickLoc.start, getQuickLoc.end);

        // ��ջ
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


float postProcess(float data[], const size_t len)
{
    // �򵥲���
    const intptr_t intlen = (intptr_t)len;

#pragma omp parallel for shared(data, intlen)
    for (intptr_t iter_series = 0; iter_series < intlen; iter_series++)
    {
        data[iter_series] = log(sqrt(data[iter_series]));
    }

    return data[len - 1];
}


float sortSpeedup(float data[], const size_t len)
{
    sortWithCuda(data, len);  // ԭʼ����ʵ��

    return data[len - 1];
}


float postSpeedup(float data[], size_t len)
{
    logsqrtWithCuda(data, len);

    return data[len - 1];
}


float sortMerge(float dataRes[], float dataA[], size_t lenA, float dataB[], size_t lenB)
{
    // ���ڹ鲢�ϲ�������������, �򵥴����㷨
    size_t dataResOffset = 0, dataAOffset = 0, dataBOffset = 0;

    while (dataAOffset < lenA || dataBOffset < lenB)
    {
        if (dataBOffset == lenB || (dataAOffset < lenA && dataA[dataAOffset] < dataB[dataBOffset]))
        {
            dataRes[dataResOffset] = dataA[dataAOffset];
            dataResOffset += 1;
            dataAOffset += 1;
        }
        else
        {
            dataRes[dataResOffset] = dataB[dataBOffset];
            dataResOffset += 1;
            dataBOffset += 1;
        }
    }

    return dataRes[0];
}


bool validSort(const float data[], const size_t len)
{
    size_t summary = 0;
    const intptr_t intlen = (intptr_t)len - 1;  // ����һ��Ԫ��
    size_t* unvalidCount = new size_t[OMP_THREADS];
    memset(unvalidCount, 0, OMP_THREADS * sizeof(size_t));

    // Ĭ��Ϊ��������
#pragma omp parallel for num_threads(OMP_THREADS) shared(data, intlen)
    for (intptr_t iter_series = 0; iter_series < intlen; iter_series++)
    {
        if (data[iter_series] > data[iter_series + 1]) unvalidCount[omp_get_thread_num()] += 1;
    }

    // �鲢
    for (size_t iter_sum = 0; iter_sum < OMP_THREADS; iter_sum++)
    {
        summary += unvalidCount[iter_sum];
    }

    delete[] unvalidCount;
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
    // ����ķ�������ָ����һλ����Ϊ��׼
    float midValue = data[leftIndex];
    float midReadValue = log(sqrt(midValue));

    // ���м�ֹͣ
    while (leftIndex < rightIndex)
    {
        // ����˳��
        // ��ָ���������, ֱ��������������ָԪ�ز�����Ҫ��ֹͣ
        while (leftIndex < rightIndex && log(sqrt(data[rightIndex])) >= midReadValue)
        {
            rightIndex--;
        }
        data[leftIndex] = data[rightIndex];  // ��ָ��Ԫ�����������Ӧλ��

        // ��ָ�����Ҳ���, ֱ��������������ָԪ�ز�����Ҫ��ֹͣ
        while (leftIndex < rightIndex && log(sqrt(data[leftIndex])) <= midReadValue)
        {
            leftIndex++;
        }
        data[rightIndex] = data[leftIndex];  // ��ָ��Ԫ�������Ҳ���Ӧλ��
    }

    // ���м�Ԫ��������Ӧλ��
    data[leftIndex] = midValue;

    return leftIndex;  // ����λ��
}
