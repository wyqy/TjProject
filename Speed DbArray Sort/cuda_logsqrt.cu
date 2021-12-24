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

    // �����������
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] last execution failed: %s!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }


    // ���������Դ�
    // ��������size
    // ����grid��blocks��size
    dimGrid.x = ((unsigned int)len % BLOCK_DIMONE_x1024 != 0) ? ((unsigned int)len / BLOCK_DIMONE_x1024 + 1) : ((unsigned int)len / BLOCK_DIMONE_x1024);
    if ((unsigned int)len < BLOCK_DIMONE_x1024) dimBlock.x = (unsigned int)len;
    else dimBlock.x = BLOCK_DIMONE_x1024;
    // �����ڴ�size
    opMemLen = max((size_t)(dimGrid.x * dimBlock.x), len);


    // �����������: host -> device
    // ��ʼ���ڴ�
    cudaStatus = cudaMalloc((void**)&gpuMemPtr, opMemLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMalloc failed for source!\n");
        goto Error;
    }
    else isGpuMemMalloc = true;
    // �����ڴ�
    cudaStatus = cudaMemcpy(gpuMemPtr, data, len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMemcpy failed when copy dataA for the first time!\n");
        goto Error;
    }


    // ����read kernel
    readKernel <<<dimGrid, dimBlock >>> (gpuMemPtr);
    // ���kernelִ�д���
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] kernel launching failed: %s @ %d, %d!\n", cudaGetErrorString(cudaStatus), dimGrid.x, dimBlock.x);
        goto Error;
    }
    // �ȴ�kernelִ�����
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
        goto Error;
    }


    // ����ִ�н��: device -> host
    if (gpuMemPtr != nullptr)
    {
        cudaStatus = cudaMemcpy(data, gpuMemPtr, len * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[Error] cudaMemcpy failed when writting back to host!\n");
            goto Error;
        }
    }


    // �ͷ�ָ����ָ�ڴ�!
Error:
    if (isGpuMemMalloc) cudaFree(gpuMemPtr);

    return cudaStatus;
}


__global__ void readKernel(float* data)
{
    // ��λ
    unsigned int thIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    // ����
    data[thIndexX] = log(sqrt(data[thIndexX]));
}

