#include "cuda_sum.cuh"

__host__ cudaError_t initialCuda(int device)
{
    // ��ʼ��CUDA�豸, �̼߳���!
    cudaError_t cudaStatus;

    // �����������
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] last execution failed: %s!\n", cudaGetErrorString(cudaStatus));
    }

    // ȷ��CUDA�豸, Ĭ��ֻѡ�е�һ���豸
    cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
    }

    return cudaStatus;
}


__host__ cudaError_t sumWithCuda(float* retValue, size_t* retLen, const float* data, size_t len)
{
    float* lastResult = nullptr;
    float* nowDataA = nullptr;
    float* nowDataB = nullptr;
    float* nowResult = nullptr;
    bool   isLastResultAMalloc = false;
    bool   isNowResultMalloc = false;

    bool isOdd = false;
    size_t wholeDataLen = len;
    size_t opDataLen = 0;
    size_t resDataLen = 0;
    size_t resMemLen = 0;

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
    resDataLen = wholeDataLen;
    // ����grid��blocks��size
    dimGrid.x = ((unsigned int)resDataLen % BLOCK_DIMONE_x1024 != 0) ? ((unsigned int)resDataLen / BLOCK_DIMONE_x1024 + 1) : ((unsigned int)resDataLen / BLOCK_DIMONE_x1024);
    if ((unsigned int)resDataLen < BLOCK_DIMONE_x1024) dimBlock.x = (unsigned int)resDataLen;
    else dimBlock.x = BLOCK_DIMONE_x1024;
    // �����ڴ�size
    resMemLen = max((size_t)(dimGrid.x * dimBlock.x), resDataLen);


    // �����������: host -> device
    // ��ʼ��lastResult�ڴ�
    cudaStatus = cudaMalloc((void**)&lastResult, resMemLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMalloc failed for source!\n");
        goto Error;
    }
    else isLastResultAMalloc = true;
    // ����lastResult�ڴ�
    cudaStatus = cudaMemcpy(lastResult, data, wholeDataLen * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMemcpy failed when copy dataA for the first time!\n");
        goto Error;
    }
    // ����������
    nowDataA = lastResult;
    nowDataB = lastResult + opDataLen;
    // ��ʼ��nowResult�ڴ�
    cudaStatus = cudaMalloc((void**)&nowResult, resMemLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMalloc failed for result!\n");
        goto Error;
    }
    else isNowResultMalloc = true;


    // ����read kernel
    readKernel <<<dimGrid, dimBlock>>> (nowResult, lastResult);
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


    // ��ʼ���
    // �鲢��: ������С!
    while (wholeDataLen > CUDA_MAX_RESLEN)
    {
        // ��������size
        isOdd = (wholeDataLen % 2 != 0);
        opDataLen = wholeDataLen / 2;
        resDataLen = isOdd ? opDataLen + 1 : opDataLen;
        // ����grid��blocks��size
        dimGrid.x = ((unsigned int)opDataLen % BLOCK_DIMONE_x1024 != 0) ? ((unsigned int)opDataLen / BLOCK_DIMONE_x1024 + 1) : ((unsigned int)opDataLen / BLOCK_DIMONE_x1024);
        if ((unsigned int)opDataLen < BLOCK_DIMONE_x1024) dimBlock.x = (unsigned int)opDataLen;
        else dimBlock.x = BLOCK_DIMONE_x1024;
        // �����ڴ�size
        resMemLen = max((size_t)(dimGrid.x * dimBlock.x), resDataLen);

        // �������ڴ洦��
        // �ͷ�lastResult�ڴ�
        cudaStatus = cudaFree(lastResult);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[Error] cudaFree failed for lastResult!\n");
            goto Error;
        }
        else isLastResultAMalloc = false;
        // �ض����ڴ�
        lastResult = nowResult;
        nowResult = nullptr;
        isNowResultMalloc = false;
        // ����������
        nowDataA = lastResult;
        nowDataB = lastResult + opDataLen;
        // ��ʼ��nowResult�ڴ�
        cudaStatus = cudaMalloc((void**)&nowResult, resMemLen * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[Error] cudaMalloc failed for result!\n");
            goto Error;
        }
        else isNowResultMalloc = true;
        // ������׼������
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[Error] device initialization failed: %s!\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // ����sum kernel
        sumKernel <<<dimGrid, dimBlock>>> (nowResult, nowDataA, nowDataB);
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

        // ��������
        if (isOdd)
        {
            // ���������ڴ�
            cudaStatus = cudaMemcpy(nowResult + opDataLen, lastResult + 2 * opDataLen, sizeof(float), cudaMemcpyDeviceToDevice);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "\n[Error] cudaMemcpy failed during odd padding!\n");
                goto Error;
            }
        }

        // ѭ������
        wholeDataLen = resDataLen;
    }


    // ����ִ�н��: device -> host
    if (nowResult != nullptr)
    {
        *retLen = wholeDataLen;

        cudaStatus = cudaMemcpy(retValue, nowResult, wholeDataLen * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[Error] cudaMemcpy failed when writting back to host!\n");
            goto Error;
        }
    }


    // �ͷ�ָ����ָ�ڴ�!
Error:
    if (isLastResultAMalloc) cudaFree(lastResult);
    if (isNowResultMalloc) cudaFree(nowResult);

    return cudaStatus;
}


__host__ cudaError_t releaseCuda(void)
{
    // ����CUDA�豸, ���̼���!
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaDeviceReset failed!\n");
    }

    return cudaStatus;
}


__global__ void readKernel(float* retValue, const float* data)
{
    // ��λ
    unsigned int thIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    // ����
    retValue[thIndexX] = log(sqrt(data[thIndexX]));
}


__global__ void sumKernel(float* retValue, const float* dataA, const float* dataB)
{
    // ��λ
    unsigned int thIndexX = blockIdx.x * blockDim.x + threadIdx.x;
    // ����
    retValue[thIndexX] = dataA[thIndexX] + dataB[thIndexX];
}

