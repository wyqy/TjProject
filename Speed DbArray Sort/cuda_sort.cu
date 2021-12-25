#include "cuda_sort.cuh"

// ���: https://www.cnblogs.com/marsggbo/p/10215830.html
// GPU��:https://blog.csdn.net/abcjennifer/article/details/47110991
// n != 2^kʱ������: https://hellolzc.github.io/2018/04/bitonic-sort-without-padding/
// https://www.sakuratears.top/blog/%E6%8E%92%E5%BA%8F%E7%AE%97%E6%B3%95%EF%BC%88%E4%BA%94%EF%BC%89-%E5%8F%8C%E8%B0%83%E6%8E%92%E5%BA%8F.html
// �˴�����ֱ�ӽ���2^k��paddingΪ2^k��С!

__host__ cudaError_t initialCuda(int device, float* arrRaw, size_t lenRaw, float* arrLoc, size_t lenLoc)
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

    // ��ʼ����ҳ�ڴ�
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
    cudaStatus = cudaMalloc((void**)&gpuMem, paddedDataLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMalloc failed for source!\n");
        goto Error;
    }
    else isGpuMemMalloc = true;
    // �����ڴ�
    cudaStatus = cudaMemcpy(gpuMem, data, len * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMemcpy failed when copy mem data for the first time!\n");
        goto Error;
    }
    // ��ʼ��padding����Ϊ0
    cudaStatus = cudaMemset(gpuMem + len, 0, (paddedDataLen - len) * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] cudaMemset failed when init for the first time!\n");
        goto Error;
    }


    // �㷨��Ҫ����:
    // ԭ��: Batcher����, 0-1����...
    // ����: ԭ���г�����2^n, nΪ����!
    // �ṹ: (�����������)
    //  ��������ʱ��������, ��Ҫ����ڴ�ֲ������⿼��!
    //      ѭ��1: ����ÿ��˫�����еİ볤��, ��Ϊiter_i, ��2��ʼ, ÿ��<<1, ��len����(��ʱlen����Ϊ����, �ȼ�˫�����г���Ϊ2*len);
    //          ѭ��2: ����Ϊ��ʹÿ��iter_i������������, ��Ҫ���е�log2(iter_i)��������, ÿ������ĵ�λ����(�����˵��),
    //                 ��Ϊiter_j, ��iter_i>>1��ʼ, ÿ��>>1, ��1����;
    //              ����3: ������������Ĳ����߳�;
    //                     1.   Ϊ�˷������, ��ȫ��Ԫ�ض���Ϊһ���̲߳�������, ���������̺߳������ж��Ƿ�������������,
    //                          ����ʹ��λ�����ж�!
    //                     2.   iter_i����˫�����е�iter_j���ĵ�λ���򳤶ȵ���Ч�����̸߳���Ϊ�Ͱ�(��߰�)iter_j>>1��(��һ���Ǳ������),
    //                          ÿ���߳�ֻ�Ƚ��Լ�����λ�ú���һ��Ķ�Ӧλ�õ�Ԫ�ص���Դ�С;
    //                     3.   min���ڵͰ���, max���ڸ߰���(��Ҫ����Ԫ��!);
    //                     


    // ��ʼ����
    // ����sort��������
    bitonicSort(gpuMem, (unsigned int)paddedDataLen, &cudaStatus);
    // ��麯��ִ�д���
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "\n[Error] sort failed!\n");
        goto Error;
    }


    // ����ִ�н��: device -> host
    if (gpuMem != nullptr)
    {
        cudaStatus = cudaMemcpy(data, gpuMem + paddedDataLen - len, len * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "\n[Error] cudaMemcpy failed when writting back to host!\n");
            goto Error;
        }
    }


    // �ͷ�ָ����ָ�ڴ�!
Error:
    if (isGpuMemMalloc) cudaFree(gpuMem);

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
    // �����ڴ��С
    size_t retValue;

    // ���ǻ���˫�������Ҫ��, ����Ϊ2��������
    if (len == 0) return 0;
    // ������������
    retValue = fastIntLog(len);
    // ��������2^k >= n����С����k
    if ((((size_t)1 << retValue) - len) != 0) retValue += 1;
    // �����ʱ�ߴ�
    retValue = ((size_t)1) << retValue;

    // ����thread block�Ĵ�С, ����Ϊ����һ��block
    // �����ڴ�size
    retValue = retValue > BLOCK_DIMONE_x512 ? retValue : BLOCK_DIMONE_x512;

    // ����
    return retValue;
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


__host__ void bitonicSort(float* data, unsigned int len, cudaError_t* cudaRetValue)
{
    unsigned int iter_i, iter_j;

    dim3 dimGrid(1);
    dim3 dimBlock(BLOCK_DIMONE_x512);
    unsigned int sharedMemSize = BLOCK_DIMONE_x512 * sizeof(float);  // ֻ����(��������)һ��

    cudaError_t cudaStatus;

    // ��������������(ȷ����512�ı���)
    dimGrid.x = len / BLOCK_DIMONE_x512;

    
    // ����(ԭ�����ǰ)
    for (iter_i = 2; iter_i <= len; iter_i <<= 1)  // ѭ��1
    {
        for (iter_j = iter_i >> 1; iter_j > 0; iter_j >>= 1)  // ѭ��2
        {
            // ����3
            // ����sort kernel
            sortKernel <<<dimGrid , dimBlock, sharedMemSize>>> (data, iter_i, iter_j);
            // ���kernelִ�д���
            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "\n[Error] kernel launching failed: %s @ %d, %d!\n", cudaGetErrorString(cudaStatus), dimGrid.x, dimBlock.x);
                *cudaRetValue = cudaErrorInvalidValue;
            }
            // �ȴ�kernelִ�����
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
    extern __shared__ int sharedMem[];  // �����ڴ�
    unsigned int share_k = threadIdx.x * WRAP_DIM;
                 share_k = share_k / BLOCK_DIMONE_x512 + share_k % BLOCK_DIMONE_x512;  // ���㹲���ڴ����λ��, ����bank conflict
    unsigned int iter_k = blockIdx.x * blockDim.x + threadIdx.x;  // ��õ�ǰԪ��
    unsigned int dual_k = iter_j ^ iter_k;  // ������, ��ö�żԪ��λ��
        // ����iter_jֻ��һλΪ1, ������Ϊ0, ���0��ԭԪ��, ���1�÷�Ԫ��, �������Ϊ��Ҫ�任��Ԫ�ص�ַ��Ӧλ

    sharedMem[share_k] = data[dual_k];  // ��ʼ����Ӧλ�õĹ����ڴ�
    __syncthreads();  // ȷ���߳̿�Ĺ����ڴ�ͬ��

    // ��ǰԪ�غͶ�żԪ�رȽ�
    if (dual_k > iter_k)  // ȷ����ǰԪ���ڵͰ���(�߰����Ǳ��Ƚϵ�)
    {
        if ((iter_k & iter_i) == 0)  // ������, �õ�������˫������һ������
            // ����iter_iֻ��һλΪ1, ��������Ϊ0��ԭԪ���ڵ�˫����, Ϊ1���ڸ�˫����
        {  // ��˫����, ��ΪMIN��, ����������������
            if (logf(sqrtf(data[iter_k])) > logf(sqrtf(sharedMem[share_k])))
            {
                // ����Ԫ��λ��
                float temp = *(data + iter_k);
                *(data + iter_k) = sharedMem[share_k];
                *(data + dual_k) = temp;
            }
        }
        else
        {  // ��˫����, ��ΪMAX��, ����������������
            if (logf(sqrtf(data[iter_k])) < logf(sqrtf(sharedMem[share_k])))
            {
                // ����Ԫ��λ��
                float temp = *(data + iter_k);
                *(data + iter_k) = sharedMem[share_k];
                *(data + dual_k) = temp;
            }
        }
    }
}

