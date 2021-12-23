#pragma once

// ����CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// ������
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

// �����Կ�����grid, thread block����״
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability
// CC6.1 -> 32          grids               per device;
//          2^{31} -1   thread blocks       per grid    in 3 dims;
//          1024        threads per         per block   in 3 dims;
//          32          wraps               per SM;
//          32          resident blocks     per SM;
//          64          resident wraps      per SM;
//          2048        resident threads    per SM;
// ��ά�ȼ���Ϳ���ά��:
// https://blog.csdn.net/dcrmg/article/details/54867507
// ��������:
// �� CUDA ���:  https://zhuanlan.zhihu.com/p/34587739
// CUDA �ڴ����: https://zhuanlan.zhihu.com/p/146691161
// CUDA �ڴ���: https://zhuanlan.zhihu.com/p/158548901
// CUDA �̶��ڴ�: https://blog.csdn.net/Rong_Toa/article/details/78665318
// CUDA �����Ż�: https://zhuanlan.zhihu.com/p/384272799
// CUDA ����Ż�:   https://www.zhihu.com/column/c_1188568938097819648
// CUDA �����첽: https://blog.csdn.net/u010335328/article/details/52453499
// CUDA ����ͬ��: https://blog.csdn.net/weixin_37804469/article/details/103659411
// Nsight ��ʹ��: https://blog.csdn.net/yan31415/article/details/109491749

