#include "cuda_sort.cuh"

// 理解: https://www.cnblogs.com/marsggbo/p/10215830.html
// GPU版:https://blog.csdn.net/abcjennifer/article/details/47110991
// n != 2^k时的排序: https://hellolzc.github.io/2018/04/bitonic-sort-without-padding/
// https://www.sakuratears.top/blog/%E6%8E%92%E5%BA%8F%E7%AE%97%E6%B3%95%EF%BC%88%E4%BA%94%EF%BC%89-%E5%8F%8C%E8%B0%83%E6%8E%92%E5%BA%8F.html
// 此处我们直接将非2^k的padding为2^k大小!

__host__ cudaError_t initialCuda(int device)
{
    // 初始化CUDA设备, 线程级别!
    cudaError_t cudaStatus;

    // 清除遗留错误
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] last execution failed: %s!\n", cudaGetErrorString(cudaStatus));
    }

    // 确定CUDA设备, 默认只选中第一个设备
    cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
    }

    return cudaStatus;
}


__host__ cudaError_t sortWithCuda(float* retValue, size_t* retLen, const float* data, size_t len)
{
    float* lastResult = nullptr;
    float* nowDataA = nullptr;
    float* nowDataB = nullptr;
    float* nowResult = nullptr;
    bool   isLastResultAMalloc = false;
    bool   isNowResultMalloc = false;

    bool isOdd = false;
    size_t paddedDataLen = len;

    dim3 dimBlock(BLOCK_DIMONE_x1024);
    dim3 dimGrid(1);

    cudaError_t cudaStatus;

    // 清除遗留错误
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] last execution failed: %s!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }


    // 计算所需显存
    // 计算内存size
    paddedDataLen = paddingSize(len);


    // 拷入操作数据: host -> device
    // 初始化内存
    cudaStatus = cudaMalloc((void**)&lastResult, paddedDataLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMalloc failed for source!\n");
        goto Error;
    }
    else isLastResultAMalloc = true;
    // 拷贝内存
    cudaStatus = cudaMemcpy(lastResult, data, len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMemcpy failed when copy mem data for the first time!\n");
        goto Error;
    }
    // 初始化padding部分为0
    cudaStatus = cudaMemset(lastResult + len, 0, (paddedDataLen - len) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMemset failed when init for the first time!\n");
        goto Error;
    }
    // ********************************到这里为止
    // 两个循环:
    // 一个大循环, 确定当前运算范围
    // 一个小循环, 在当前运算范围内从大到小运算
    // 注意设定kernel函数运算范围!

    // 开始求和
    // 归并法: 不断缩小!
    while (paddedDataLen > CUDA_MAX_RESLEN)
    {
        // 计算数据size
        isOdd = (paddedDataLen % 2 != 0);
        opDataLen = paddedDataLen / 2;
        resDataLen = isOdd ? opDataLen + 1 : opDataLen;
        // 计算grid和blocks的size
        dimGrid.x = ((unsigned int)opDataLen % BLOCK_DIMONE_x1024 != 0) ? ((unsigned int)opDataLen / BLOCK_DIMONE_x1024 + 1) : ((unsigned int)opDataLen / BLOCK_DIMONE_x1024);
        if ((unsigned int)opDataLen < BLOCK_DIMONE_x1024) dimBlock.x = (unsigned int)opDataLen;
        else dimBlock.x = BLOCK_DIMONE_x1024;
        // 计算内存size
        resMemLen = max((size_t)(dimGrid.x * dimBlock.x), resDataLen);

        // 操作数内存处理
        // 释放lastResult内存
        cudaStatus = cudaFree(lastResult);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[Error] cudaFree failed for lastResult!\n");
            goto Error;
        }
        else isLastResultAMalloc = false;
        // 重定向内存
        lastResult = nowResult;
        nowResult = nullptr;
        isNowResultMalloc = false;
        // 操作数处理
        nowDataA = lastResult;
        nowDataB = lastResult + opDataLen;
        // 初始化nowResult内存
        cudaStatus = cudaMalloc((void**)&nowResult, resMemLen * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[Error] cudaMalloc failed for result!\n");
            goto Error;
        }
        else isNowResultMalloc = true;
        // 检查程序准备错误
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[Error] device initialization failed: %s!\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // 运行sum kernel
        sumKernel << <dimGrid, dimBlock >> > (nowResult, nowDataA, nowDataB);
        // 检查kernel执行错误
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[Error] kernel launching failed: %s @ %d, %d!\n", cudaGetErrorString(cudaStatus), dimGrid.x, dimBlock.x);
            goto Error;
        }
        // 等待kernel执行完成
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[Error] cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
            goto Error;
        }

        // 奇数处理
        if (isOdd)
        {
            // 拷贝单个内存
            cudaStatus = cudaMemcpy(nowResult + opDataLen, lastResult + 2 * opDataLen, sizeof(float), cudaMemcpyDeviceToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "\n[Error] cudaMemcpy failed during odd padding!\n");
                goto Error;
            }
        }

        // 循环操作
        paddedDataLen = resDataLen;
    }


    // 拷回执行结果: device -> host
    if (nowResult != nullptr)
    {
        *retLen = paddedDataLen;

        cudaStatus = cudaMemcpy(retValue, nowResult, paddedDataLen * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[Error] cudaMemcpy failed when writting back to host!\n");
            goto Error;
        }
    }


    // 释放指针所指内存!
Error:
    if (isLastResultAMalloc) cudaFree(lastResult);
    if (isNowResultMalloc) cudaFree(nowResult);

    return cudaStatus;
}


__host__ cudaError_t releaseCuda(void)
{
    // 重置CUDA设备, 进程级别!
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaDeviceReset failed!\n");
    }

    return cudaStatus;
}


__host__ size_t paddingSize(size_t len)
{
    // 返回满足2^k >= n的最小整数k对应的size!
    if (len == 0) return 0;
    size_t retValue = fastIntLog(len);
    if (abs((((size_t)1) << retValue) - (double)len) > 1e-1) retValue += 1;
    return ((size_t)1) << retValue;
}


__host__ size_t fastIntLog(size_t x)
{
    // 快速取整数的2的对数
    float fx;
    unsigned long ix, exp;

    fx = (float)x;
    ix = *(unsigned long*)&fx;
    exp = (ix >> 23) & 0xFF;

    return (size_t)(exp - 127);
}


__global__ void readKernel(float* retValue, const float* data)
{
    // 定位
    unsigned int thIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算
    retValue[thIndexX] = log(sqrt(data[thIndexX]));
}


__global__ void sumKernel(float* retValue, const float* dataA, const float* dataB)
{
    // 定位
    unsigned int thIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算
    retValue[thIndexX] = dataA[thIndexX] + dataB[thIndexX];
}


