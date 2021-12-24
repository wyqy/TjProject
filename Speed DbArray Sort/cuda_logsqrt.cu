#include "cuda_logsqrt.cuh"

__host__ cudaError_t logsqrtWithCuda(float* data, size_t len)
{
    float* gpuMemPtr = nullptr;
    bool   isGpuMemMalloc = false;

    bool isOdd = false;
    size_t opMemLen = 0;

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
    // 计算grid和blocks的size
    dimGrid.x = ((unsigned int)len % BLOCK_DIMONE_x1024 != 0) ? ((unsigned int)len / BLOCK_DIMONE_x1024 + 1) : ((unsigned int)len / BLOCK_DIMONE_x1024);
    if ((unsigned int)len < BLOCK_DIMONE_x1024) dimBlock.x = (unsigned int)len;
    else dimBlock.x = BLOCK_DIMONE_x1024;
    // 计算内存size
    opMemLen = max((size_t)(dimGrid.x * dimBlock.x), len);


    // 拷入操作数据: host -> device
    // 初始化内存
    cudaStatus = cudaMalloc((void**)&gpuMemPtr, opMemLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMalloc failed for source!\n");
        goto Error;
    }
    else isGpuMemMalloc = true;
    // 拷贝内存
    cudaStatus = cudaMemcpy(gpuMemPtr, data, len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMemcpy failed when copy dataA for the first time!\n");
        goto Error;
    }


    // 运行read kernel
    readKernel <<<dimGrid, dimBlock >>> (gpuMemPtr);
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


    // 拷回执行结果: device -> host
    if (gpuMemPtr != nullptr)
    {
        cudaStatus = cudaMemcpy(data, gpuMemPtr, len * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[Error] cudaMemcpy failed when writting back to host!\n");
            goto Error;
        }
    }


    // 释放指针所指内存!
Error:
    if (isGpuMemMalloc) cudaFree(gpuMemPtr);

    return cudaStatus;
}


__global__ void readKernel(float* data)
{
    // 定位
    unsigned int thIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算
    data[thIndexX] = log(sqrt(data[thIndexX]));
}

