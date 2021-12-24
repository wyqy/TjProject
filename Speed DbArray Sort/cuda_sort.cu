#include "cuda_sort.cuh"

// ���: https://www.cnblogs.com/marsggbo/p/10215830.html
// GPU��:https://blog.csdn.net/abcjennifer/article/details/47110991
// n != 2^kʱ������: https://hellolzc.github.io/2018/04/bitonic-sort-without-padding/
// https://www.sakuratears.top/blog/%E6%8E%92%E5%BA%8F%E7%AE%97%E6%B3%95%EF%BC%88%E4%BA%94%EF%BC%89-%E5%8F%8C%E8%B0%83%E6%8E%92%E5%BA%8F.html
// �˴�����ֱ�ӽ���2^k��paddingΪ2^k��С!

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

    // �����������
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] last execution failed: %s!\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }


    // ���������Դ�
    // �����ڴ�size
    paddedDataLen = paddingSize(len);


    // �����������: host -> device
    // ��ʼ���ڴ�
    cudaStatus = cudaMalloc((void**)&lastResult, paddedDataLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMalloc failed for source!\n");
        goto Error;
    }
    else isLastResultAMalloc = true;
    // �����ڴ�
    cudaStatus = cudaMemcpy(lastResult, data, len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMemcpy failed when copy mem data for the first time!\n");
        goto Error;
    }
    // ��ʼ��padding����Ϊ0
    cudaStatus = cudaMemset(lastResult + len, 0, (paddedDataLen - len) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMemset failed when init for the first time!\n");
        goto Error;
    }
    // ********************************������Ϊֹ
    // ����ѭ��:
    // һ����ѭ��, ȷ����ǰ���㷶Χ
    // һ��Сѭ��, �ڵ�ǰ���㷶Χ�ڴӴ�С����
    // ע���趨kernel�������㷶Χ!

    // ��ʼ���
    // �鲢��: ������С!
    while (paddedDataLen > CUDA_MAX_RESLEN)
    {
        // ��������size
        isOdd = (paddedDataLen % 2 != 0);
        opDataLen = paddedDataLen / 2;
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
        sumKernel << <dimGrid, dimBlock >> > (nowResult, nowDataA, nowDataB);
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
        paddedDataLen = resDataLen;
    }


    // ����ִ�н��: device -> host
    if (nowResult != nullptr)
    {
        *retLen = paddedDataLen;

        cudaStatus = cudaMemcpy(retValue, nowResult, paddedDataLen * sizeof(float), cudaMemcpyDeviceToHost);
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


__host__ size_t paddingSize(size_t len)
{
    // ��������2^k >= n����С����k��Ӧ��size!
    if (len == 0) return 0;
    size_t retValue = fastIntLog(len);
    if (abs((((size_t)1) << retValue) - (double)len) > 1e-1) retValue += 1;
    return ((size_t)1) << retValue;
}


__host__ size_t fastIntLog(size_t x)
{
    // ����ȡ������2�Ķ���
    float fx;
    unsigned long ix, exp;

    fx = (float)x;
    ix = *(unsigned long*)&fx;
    exp = (ix >> 23) & 0xFF;

    return (size_t)(exp - 127);
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


