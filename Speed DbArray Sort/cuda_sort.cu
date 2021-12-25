#include "cuda_sort.cuh"

// 理解: https://www.cnblogs.com/marsggbo/p/10215830.html
// GPU版:https://blog.csdn.net/abcjennifer/article/details/47110991
// n != 2^k时的排序: https://hellolzc.github.io/2018/04/bitonic-sort-without-padding/
// https://www.sakuratears.top/blog/%E6%8E%92%E5%BA%8F%E7%AE%97%E6%B3%95%EF%BC%88%E4%BA%94%EF%BC%89-%E5%8F%8C%E8%B0%83%E6%8E%92%E5%BA%8F.html
// 此处我们直接将非2^k的padding为2^k大小!

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
    cudaStatus = cudaHostRegister(arrRaw, lenRaw * sizeof(float), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaHostRegister for arrRaw failed!\n");
    }
    cudaStatus = cudaHostRegister(arrLoc, lenLoc * sizeof(float), cudaHostAllocDefault);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaHostRegister for arrLoc failed!\n");
    }

    return cudaStatus;
}


__host__ cudaError_t sortWithCuda(float* data, size_t len)
{
    float* gpuMem = nullptr;
    bool   isGpuMemMalloc = false;
    size_t paddedDataLen = len;

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
    cudaStatus = cudaMalloc((void**)&gpuMem, paddedDataLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMalloc failed for source!\n");
        goto Error;
    }
    else isGpuMemMalloc = true;
    // 拷贝内存
    cudaStatus = cudaMemcpy(gpuMem, data, len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMemcpy failed when copy mem data for the first time!\n");
        goto Error;
    }
    // 初始化padding部分为0
    cudaStatus = cudaMemset(gpuMem + len, 0, (paddedDataLen - len) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMemset failed when init for the first time!\n");
        goto Error;
    }


    // 算法简要描述:
    // 原理: Batcher定理, 0-1序列...
    // 假设: 原序列长度是2^n, n为整数!
    // 结构: (假设先增后减)
    //  对于运行时参数问题, 需要结合内存局部性问题考虑!
    //      循环1: 决定每个双调序列的半长度, 设为iter_i, 从2开始, 每次<<1, 到len结束(此时len长度为单调, 等价双调序列长度为2*len);
    //          循环2: 决定为了使每个iter_i长的序列有序, 需要进行的log2(iter_i)轮排序中, 每轮排序的单位长度(下面会说明),
    //                 设为iter_j, 从iter_i>>1开始, 每次>>1, 到1结束;
    //              操作3: 决定参与排序的并发线程;
    //                     1.   为了方便调度, 令全部元素都作为一个线程参与排序, 但在排序线程函数中判断是否真正参与排序,
    //                          可以使用位运算判断!
    //                     2.   iter_i长的双调序列的iter_j长的单位排序长度的有效排序线程个数为低半(或高半)iter_j>>1个(另一半是被排序的),
    //                          每个线程只比较自己所在位置和另一半的对应位置的元素的相对大小;
    //                     3.   min放在低半区, max放在高半区(需要交换元素!);
    //                     


    // 开始排序
    // 运行sort辅助函数
    bitonicSort(gpuMem, (unsigned int)paddedDataLen, &cudaStatus);
    // 检查函数执行错误
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] sort failed!\n");
        goto Error;
    }


    // 拷回执行结果: device -> host
    if (gpuMem != nullptr)
    {
        cudaStatus = cudaMemcpy(data, gpuMem + paddedDataLen - len, len * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[Error] cudaMemcpy failed when writting back to host!\n");
            goto Error;
        }
    }


    // 释放指针所指内存!
Error:
    if (isGpuMemMalloc) cudaFree(gpuMem);

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
    // 返回内存大小
    size_t retValue;

    // 考虑基础双调排序的要求, 扩充为2的整数幂
    if (len == 0) return 0;
    // 快速整数对数
    retValue = fastIntLog(len);
    // 计算满足2^k >= n的最小整数k
    if ((((size_t)1 << retValue) - len) != 0) retValue += 1;
    // 计算此时尺寸
    retValue = ((size_t)1) << retValue;

    // 考虑thread block的大小, 扩充为至少一个block
    // 计算内存size
    retValue = retValue > BLOCK_DIMONE_x512 ? retValue : BLOCK_DIMONE_x512;

    // 返回
    return retValue;
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


__host__ void bitonicSort(float* data, unsigned int len, cudaError_t* cudaRetValue)
{
    unsigned int iter_i, iter_j;

    dim3 dimGrid(1);
    dim3 dimBlock(BLOCK_DIMONE_x512);
    unsigned int sharedMemSize = BLOCK_DIMONE_x512 * sizeof(float);  // 只缓存(操作数的)一半

    cudaError_t cudaStatus;

    // 计算运行配置数(确定是512的倍数)
    dimGrid.x = len / BLOCK_DIMONE_x512;

    
    // 排序(原理详见前)
    for (iter_i = 2; iter_i <= len; iter_i <<= 1)  // 循环1
    {
        for (iter_j = iter_i >> 1; iter_j > 0; iter_j >>= 1)  // 循环2
        {
            // 操作3
            // 运行sort kernel
            sortKernel <<<dimGrid , dimBlock, sharedMemSize>>> (data, iter_i, iter_j);
            // 检查kernel执行错误
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "\n[Error] kernel launching failed: %s @ %d, %d!\n", cudaGetErrorString(cudaStatus), dimGrid.x, dimBlock.x);
                *cudaRetValue = cudaErrorInvalidValue;
            }
            // 等待kernel执行完成
            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "\n[Error] cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
                *cudaRetValue = cudaErrorInvalidValue;
            }
        }
    }
}


__global__ void sortKernel(float* data, unsigned int iter_i, unsigned int iter_j)
{
    extern __shared__ int sharedMem[];  // 共享内存
    unsigned int share_k = threadIdx.x * WRAP_DIM;
                 share_k = share_k / BLOCK_DIMONE_x512 + share_k % BLOCK_DIMONE_x512;  // 计算共享内存调用位置, 避免bank conflict
    unsigned int iter_k = blockIdx.x * blockDim.x + threadIdx.x;  // 获得当前元素
    unsigned int dual_k = iter_j ^ iter_k;  // 或运算, 获得对偶元素位置
        // 由于iter_j只有一位为1, 其它均为0, 异或0得原元素, 异或1得反元素, 因此正好为需要变换的元素地址对应位

    sharedMem[share_k] = data[dual_k];  // 初始化对应位置的共享内存
    __syncthreads();  // 确保线程块的共享内存同步

    // 当前元素和对偶元素比较
    if (dual_k > iter_k)  // 确保当前元素在低半区(高半区是被比较的)
    {
        if ((iter_k & iter_i) == 0)  // 与运算, 得到现在是双调的哪一单调区
            // 由于iter_i只有一位为1, 因此若结果为0则原元素在低双调区, 为1则在高双调区
        {  // 低双调区, 设为MIN区, 则排序结果单调递增
            if (logf(sqrtf(data[iter_k])) > logf(sqrtf(sharedMem[share_k])))
            {
                // 交换元素位置
                float temp = *(data + iter_k);
                *(data + iter_k) = sharedMem[share_k];
                *(data + dual_k) = temp;
            }
        }
        else
        {  // 高双调区, 设为MAX区, 则排序结果单调递增
            if (logf(sqrtf(data[iter_k])) < logf(sqrtf(sharedMem[share_k])))
            {
                // 交换元素位置
                float temp = *(data + iter_k);
                *(data + iter_k) = sharedMem[share_k];
                *(data + dual_k) = temp;
            }
        }
    }
}

