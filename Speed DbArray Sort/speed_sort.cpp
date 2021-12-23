#include "speed_sort.h"

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

