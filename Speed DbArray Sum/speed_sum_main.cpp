#include "speed_sum.h"

__declspec(align(MEMORY_ALIGNED)) float rawFloatData[DUAL_DATANUM];
__declspec(align(MEMORY_ALIGNED)) float locFloatData[SGLE_DATANUM];

int main()
{
    int main_entered;
    float retValue = 0;
    double durationSecond = 0;

    system_clock::time_point start_time;
    system_clock::time_point end_time;
    duration<double> diff;
    wcout.imbue(locale("chs"));

    float locRetValue = 0;
    float rmtRetValue = 0;

    int* recvTypeHeap = new int;
    int* recvLenHeap = new int;
    int* socketIntHeap = new int;
    float* socketFloatHeap = new float;
    *recvTypeHeap = 0;
    *recvLenHeap = 0;
    *socketIntHeap = 0;
    *socketFloatHeap = 0.0f;

    cudaError_t cudaStatus;

    wcout << L"��ʼ��, ���Ժ�...";
    // ����׼��
    retValue = init_arithmetic(rawFloatData, (float)1, DUAL_DATANUM);  // ��ʼ����������!
    system("cls");

    while (true)
    {
        wcout << L"\n������������ͻ�׼ʱ�����(5 ��): \n\
                   1. ԭʼ����ʵ��(������); \n\
                   2. CUDA + AVX����ʵ��(�����С); \n\
                   3. OMP + AVX����ʵ��(�������); \n\
                   4. CUDA + AVX + Socket����ʵ��; \n\
                   5. OMP + AVX + Socket����ʵ��; \n\
                   6. �˳�. \n\
                   ������ѡ��: ";
        wcin >> main_entered;
        if (main_entered == 6) break;

        switch (main_entered)
        {
        case 1:
            for (size_t iter = 1; iter <= 5; iter++)
            {
                // ִ��
                start_time = system_clock::now();
                retValue = sumNaive(rawFloatData, (size_t)DUAL_DATANUM);  // ԭʼ����ʵ��
                end_time = system_clock::now();
                diff = end_time - start_time;
                durationSecond = diff.count();
                // ��ʾ
                wcout << L"��" << iter << L"��: ��ʱ(��): " << durationSecond;
                wcout << L"; �ܺ�Ϊ: " << fixed << retValue << endl;
            }
            break;
        case 2:  // 2, 3 ������ͬ, ��˼�
            if (main_entered == 2) cudaStatus = initialCuda(0);  // ���ڶ�GPUϵͳ���޸�!
            if (cudaStatus != cudaSuccess) goto InitError;
            [[fallthrough]];
        case 3:
            for (size_t iter = 1; iter <= 5; iter++)
            {
                // ִ��
                start_time = system_clock::now();
                if (main_entered == 2) retValue = sumSpeedup_Cu(rawFloatData, (size_t)DUAL_DATANUM);  // CUDA����
                else if (main_entered == 3) retValue = sumSpeedup_OP(rawFloatData, (size_t)DUAL_DATANUM);  // OMP����
                end_time = system_clock::now();
                diff = end_time - start_time;
                durationSecond = diff.count();
                // ��ʾ
                wcout << L"��" << iter << L"��: ��ʱ(��): " << durationSecond;
                wcout << L"; �ܺ�Ϊ: " << fixed << retValue << endl;
            }
            // �ر�Cuda
            if (main_entered == 2) cudaStatus = releaseCuda();  // �ͷ�CUDA
            break;
        case 4:  // 4, 5 ������ͬ, ��˼�
            if (main_entered == 4) cudaStatus = initialCuda(0);  // ���ڶ�GPUϵͳ���޸�!
            if (cudaStatus != cudaSuccess) goto InitError;
            [[fallthrough]];
        case 5:
            // ���׼��
            int socket_type;                        // ͨ������: 1 = ������; 2 = �ͻ���
            char* socket_ip;                        // Socket IP
            u_short socket_port;                    // Socket Port
            double sockLatency;
            SocketConfig* socketconfig;

            wcout << L"\n������ͨ������: \n\
                       1. ���ض�(Server); \n\
                       2. �ܿض�(Client). \n\
                       ������ѡ��: ";
            wcin >> socket_type;
            wcout << L"\n������IP: ";
            socket_ip = new char[16];
            cin.get();
            cin.get(socket_ip, 16);
            wcout << L"\n������˿ں�: ";
            wcin >> socket_port;

            wcout << L"��ʼ��socket, ���Ժ�...";
            // ��ʼ��Socket�����
            socketconfig = InitSocket(socket_type, socket_ip, socket_port);
            // �ȴ��ӳٲ���
            sockLatency = InitLatency(socketconfig);
            // ��ʾ�ӳ�
            wcout << L"�ɹ�! ˫�������ӳ�(����): " << sockLatency * 1000.0f << endl;

            wcout << L"��ʼ���ֲ�ʽ����, ���Ժ�..." << endl;

            if (socket_type == 1)
            {
                // ���׼��
                // ���ͳ�ʼ���������ͻ���
                *socketIntHeap = 1000;
                SendCommand(socketconfig, COMMAND_INT, sizeof(int), socketIntHeap);
                // ������ʼ��
                retValue = init_arithmetic(locFloatData, (float)1, SGLE_DATANUM);  // ��ʼ���������!
                // �ȴ��Է����
                RecvCommand(socketconfig, recvTypeHeap, recvLenHeap, socketIntHeap);
                if (*socketIntHeap != 1000) break;

                for (size_t iter = 1; iter <= 5; iter++)
                {
                    // ִ��
                    start_time = system_clock::now();
                    // ����ִ���������ͻ���
                    *socketIntHeap = 1001;
                    SendCommand(socketconfig, COMMAND_INT, sizeof(int), socketIntHeap);
                    // ִ�м�������(��������)
                    if (main_entered == 4) locRetValue = sumSpeedup_Cu(locFloatData, (size_t)SGLE_DATANUM);  // CUDA����
                    else if (main_entered == 5) locRetValue = sumSpeedup_OP(locFloatData, (size_t)SGLE_DATANUM);  // OMP����
                    // ���ͼ�������
                    *socketFloatHeap = locRetValue;
                    SendCommand(socketconfig, COMMAND_FLOAT, sizeof(float), socketFloatHeap);
                    // �ȴ��Է����
                    RecvCommand(socketconfig, recvTypeHeap, recvLenHeap, socketFloatHeap);
                    rmtRetValue = *socketFloatHeap;
                    // �ϲ���������
                    retValue = locRetValue + rmtRetValue;
                    // �������
                    end_time = system_clock::now();
                    diff = end_time - start_time;
                    durationSecond = diff.count();
                    // ��ʾ
                    wcout << L"��" << iter << L"��: ��ʱ(��): " << durationSecond;
                    wcout << L"; �ܺ�Ϊ: " << fixed << retValue << endl;
                    // ����ͬ��ָ��
                    RecvCommand(socketconfig, recvTypeHeap, recvLenHeap, socketIntHeap);
                    if (*socketIntHeap != 1002) break;
                }
            }
            else if (socket_type == 2)
            {
                // ���׼��
                // �������Է������ĳ�ʼ������
                RecvCommand(socketconfig, recvTypeHeap, recvLenHeap, socketIntHeap);
                if (*socketIntHeap != 1000) break;
                // ������ʼ��
                retValue = init_arithmetic(locFloatData, (float)SGLE_DATANUM, SGLE_DATANUM);  // ��ʼ���������!
                // ���ͼ������
                *socketIntHeap = 1000;
                SendCommand(socketconfig, COMMAND_INT, sizeof(int), socketIntHeap);

                for (size_t iter = 1; iter <= 5; iter++)
                {
                    // ִ��
                    start_time = system_clock::now();
                    // �������Է�������ִ������
                    RecvCommand(socketconfig, recvTypeHeap, recvLenHeap, socketIntHeap);
                    // ִ�м�������(��������)
                    if (*socketIntHeap == 1001)
                    {
                        if (main_entered == 4) locRetValue = sumSpeedup_Cu(locFloatData, (size_t)SGLE_DATANUM);  // CUDA����
                        else if (main_entered == 5) locRetValue = sumSpeedup_OP(locFloatData, (size_t)SGLE_DATANUM);  // OMP����
                    }
                    // ���ͼ�������
                    *socketFloatHeap = locRetValue;
                    SendCommand(socketconfig, COMMAND_FLOAT, sizeof(float), socketFloatHeap);
                    // �ȴ��Է����
                    RecvCommand(socketconfig, recvTypeHeap, recvLenHeap, socketFloatHeap);
                    rmtRetValue = *socketFloatHeap;
                    // �ϲ���������
                    retValue = locRetValue + rmtRetValue;
                    // �������
                    end_time = system_clock::now();
                    diff = end_time - start_time;
                    durationSecond = diff.count();
                    // ��ʾ
                    wcout << L"��" << iter << L"��: ��ʱ(��): " << durationSecond;
                    wcout << L"; �ܺ�Ϊ: " << fixed << retValue << endl;
                    // ����ͬ��ָ��
                    *socketIntHeap = 1002;
                    SendCommand(socketconfig, COMMAND_INT, sizeof(int), socketIntHeap);
                }
            }

            // �ͷŶ��ڴ�
            delete[] socket_ip;
            // �ر�����
            EndCommand(socketconfig);
            // ����Socket�����
            CloseSocket(socketconfig);
            // �ر�Cuda
            if (main_entered == 4) cudaStatus = releaseCuda();  // �ͷ�CUDA
            break;
        default:
            durationSecond = 0;
            retValue = 0;
        }
    }

InitError:
    delete recvTypeHeap;
    delete recvLenHeap;
    delete socketIntHeap;
    delete socketFloatHeap;
    system("pause");
    return 0;
}
