#pragma once

// 调用公共库
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
// 说明
#include <new>           // 新new语法
#include <numeric>       // 数字库
#include <algorithm>     // 数学库1
#include <cmath>         // 数学库2
#include <random>        // 随机库
#include <chrono>        // STD时间库
// 命名空间
// 输入输出部分
using std::cin;
using std::cout;
using std::wcin;
using std::wcout;
using std::fixed;
using std::endl;
using std::locale;
using std::string;
// 随机部分
using std::random_device;
using std::mt19937;
using std::uniform_int;
using std::uniform_int_distribution;
// 计时部分
using namespace std::chrono;

// 定义宏
// 乘数
#define MAX_THREADS 64
#define SGLE_SUBDATANUM 1000000
#define DUAL_SUBDATANUM 2000000
#define TEST_SGLE_SUBDATANUM 10000
#define TEST_DUAL_SUBDATANUM 20000
// 数组大小
#define SGLE_DATANUM (SGLE_SUBDATANUM * MAX_THREADS)
#define DUAL_DATANUM (DUAL_SUBDATANUM * MAX_THREADS)
#define TEST_SGLE_DATANUM (TEST_SGLE_SUBDATANUM * MAX_THREADS)
#define TEST_DUAL_DATANUM (TEST_DUAL_SUBDATANUM * MAX_THREADS)
