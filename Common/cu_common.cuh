#pragma once

// 调用CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// 基本库
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

// 根据显卡定义grid, thread block的形状
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability
// CC6.1 -> 32          grids               per device;
//          2^{31} -1   thread blocks       per grid    in 3 dims;
//          1024        threads per         per block   in 3 dims;
//          32          wraps               per SM;
//          32          resident blocks     per SM;
//          64          resident wraps      per SM;
//          2048        resident threads    per SM;
// 附维度计算和可用维度:
// https://blog.csdn.net/dcrmg/article/details/54867507
// 更多资料:
// 简单 CUDA 编程:  https://zhuanlan.zhihu.com/p/34587739
// CUDA 内存管理: https://zhuanlan.zhihu.com/p/146691161
// CUDA 内存编程: https://zhuanlan.zhihu.com/p/158548901
// CUDA 固定内存: https://blog.csdn.net/Rong_Toa/article/details/78665318
// CUDA 缓存优化: https://zhuanlan.zhihu.com/p/384272799
// CUDA 编程优化:   https://www.zhihu.com/column/c_1188568938097819648
// CUDA 流与异步: https://blog.csdn.net/u010335328/article/details/52453499
// CUDA 并行同步: https://blog.csdn.net/weixin_37804469/article/details/103659411
// CUDA 配置参数: https://blog.csdn.net/fuxiaoxiaoyue/article/details/83782167
// CUDA 内存传输: https://zhuanlan.zhihu.com/p/188246455
// Nsight 的使用: https://blog.csdn.net/yan31415/article/details/109491749

