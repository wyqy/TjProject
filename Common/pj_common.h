#pragma once

// ���ù�����
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
// ˵��
#include <new>           // ��new�﷨
#include <numeric>       // ���ֿ�
#include <algorithm>     // ��ѧ��1
#include <cmath>         // ��ѧ��2
#include <random>        // �����
#include <chrono>        // STDʱ���
// �����ռ�
using std::cin;
using std::cout;
using std::wcin;
using std::wcout;
using std::fixed;
using std::endl;
using std::locale;
using std::string;
using std::random_device;
using std::mt19937;
using std::uniform_real_distribution;
using namespace std::chrono;

// �����
#define MAX_THREADS 64
#define SGLE_SUBDATANUM 1000000
#define DUAL_SUBDATANUM 2000000
#define SGLE_DATANUM (SGLE_SUBDATANUM * MAX_THREADS)
#define DUAL_DATANUM (DUAL_SUBDATANUM * MAX_THREADS)