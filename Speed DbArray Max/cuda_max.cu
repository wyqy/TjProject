#include "cuda_max.cuh"

__host__ cudaError_t initialCuda(int device, float* arrRaw, size_t lenRaw, float* arrLoc, size_t lenLoc)
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

    // 初始化锁页内存
    cudaStatus = cudaHostRegister(arrRaw, lenRaw * sizeof(float), cudaHostAllocMapped);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaHostRegister for arrRaw failed!\n");
    }
    cudaStatus = cudaHostRegister(arrLoc, lenLoc * sizeof(float), cudaHostAllocMapped);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaHostRegister for arrLoc failed!\n");
    }

    return cudaStatus;
}


__host__ cudaError_t maxWithCuda(float* retValue, size_t* retLen, const float* data, size_t len)
{
    float* lastResult = nullptr;
    float* nowDataA = nullptr;
    float* nowDataB = nullptr;
    float* nowResult = nullptr;
    bool   isLastResultMalloc = false;
    bool   isNowResultMalloc = false;

    bool isOdd = false;
    size_t wholeDataLen = len;
    size_t opDataLen = 0;
    size_t resDataLen = 0;
    size_t resMemLen = 0;

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
    // 计算数据size
    resDataLen = wholeDataLen;
    // 计算grid和blocks的size
    dimGrid.x = ((unsigned int)resDataLen % BLOCK_DIMONE_x1024 != 0) ? ((unsigned int)resDataLen / BLOCK_DIMONE_x1024 + 1) : ((unsigned int)resDataLen / BLOCK_DIMONE_x1024);
    if ((unsigned int)resDataLen < BLOCK_DIMONE_x1024) dimBlock.x = (unsigned int)resDataLen;
    else dimBlock.x = BLOCK_DIMONE_x1024;
    // 计算内存size
    resMemLen = max((size_t)(dimGrid.x * dimBlock.x), resDataLen);


    // 拷入操作数据: host -> device
    // 初始化lastResult内存
    cudaStatus = cudaMalloc((void**)&lastResult, resMemLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMalloc failed for source!\n");
        goto Error;
    }
    else isLastResultMalloc = true;
    // 拷贝lastResult内存
    cudaStatus = cudaMemcpy(lastResult, data, wholeDataLen * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMemcpy failed when copying for the first time!\n");
        goto Error;
    }
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


    // 运行read kernel
    readKernel <<<dimGrid, dimBlock >>> (nowResult, lastResult);
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


    // 开始比较
    // 二分法: 不断缩小!
    while (wholeDataLen > CUDA_MAX_RESLEN)
    {
        // 计算数据size
        isOdd = (wholeDataLen % 2 != 0);
        opDataLen = wholeDataLen / 2;
        resDataLen = isOdd ? opDataLen + 1 : opDataLen;
        // 计算grid和blocks的size
        dimGrid.x = ((unsigned int)opDataLen % BLOCK_DIMONE_x1024 != 0)? ((unsigned int)opDataLen / BLOCK_DIMONE_x1024 + 1) : ((unsigned int)opDataLen / BLOCK_DIMONE_x1024);
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
        else isLastResultMalloc = false;
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

        // 运行max kernel
        maxKernel<<<dimGrid, dimBlock>>> (nowResult, nowDataA, nowDataB);
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
            cudaStatus = cudaMemcpy(nowResult + opDataLen, lastResult + 2*opDataLen, sizeof(float), cudaMemcpyDeviceToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "\n[Error] cudaMemcpy failed during odd padding!\n");
                goto Error;
            }
        }

        // 循环操作
        wholeDataLen = resDataLen;
    }


    // 拷回执行结果: device -> host
    if (nowResult != nullptr)
    {
        *retLen = wholeDataLen;

        cudaStatus = cudaMemcpy(retValue, nowResult, wholeDataLen * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[Error] cudaMemcpy failed when writting back to host!\n");
            goto Error;
        }
    }


    // 释放指针所指内存!
Error:
    if (isLastResultMalloc) cudaFree(lastResult);
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


__global__ void readKernel(float* retValue, const float* data)
{
    // 定位
    unsigned int thIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算
    float elem = data[thIndexX];
    retValue[thIndexX] = log(sqrt(elem));
}


__global__ void maxKernel(float* retValue, const float* dataA, const float* dataB)
{
    // 定位
    unsigned int thIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算
    float originalA = dataA[thIndexX];
    float originalB = dataB[thIndexX];
    retValue[thIndexX] = originalA > originalB ? originalA : originalB;
}

