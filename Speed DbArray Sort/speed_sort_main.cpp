#include "speed_sort.h"

__declspec(align(MEMORY_ALIGNED)) float rawFloatData[DUAL_DATANUM];
__declspec(align(MEMORY_ALIGNED)) float locFloatData[SGLE_DATANUM];
__declspec(align(MEMORY_ALIGNED)) float rmtFloatData[SGLE_DATANUM];

int main()
{
    // ���������������
    int main_entered;
    float retValue = 0;
    double durationSecond = 0;
    wcout.imbue(locale("chs"));

    // ��ʱ����
    system_clock::time_point start_time;
    system_clock::time_point end_time;
    duration<double> diff;

    // �������
    float locRetValue = 0.0f;
    float rmtRetValue = 0.0f;
    int socketRetStatus = 0;

    // cuda����
    cudaError_t cudaStatus;

    wcout << L"��ʼ��, ���Ժ�...";
    // ����׼��
    retValue = init_arithmetic(rawFloatData, (float)1, DUAL_DATANUM);  // ��ʼ����������!
    system("cls");

    while (true)
    {
        // wcout << L"\n���������������׼ʱ�����(5 ��): \n\
        //            1. ԭʼ����ʵ��; \n\
        //            2. ��������ʵ��(�����С); \n\
        //            1. ˫������ʵ��; \n\
        //            4. �˳�. \n\
        //            ������ѡ��: ";
        wcout << L"���鴫�����: \n\
                    1. ����; \n\
                    2. �˳�. \n\
                    ������ѡ��: ";
        wcin >> main_entered;
        if (main_entered == 2) break;

        switch (main_entered)
        {
        // case 1:
        //     for (size_t iter = 1; iter <= 5; iter++)
        //     {
        //         // ִ��
        //         start_time = system_clock::now();
        //         // retValue = sumNaive(rawFloatData, (size_t)DUAL_DATANUM);  // ԭʼ����ʵ��
        //         end_time = system_clock::now();
        //         diff = end_time - start_time;
        //         durationSecond = diff.count();
        //         // ��ʾ
        //         wcout << L"��" << iter << L"��: ��ʱ(��): " << durationSecond;
        //         wcout << L"; ���ֵΪ: " << fixed << retValue << endl;
        //     }
        //     break;
        // case 2:
        //     break;
        case 1:
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
                SendInt(socketconfig, 1000);
                // ������ʼ��
                retValue = init_arithmetic(locFloatData, (float)1, SGLE_DATANUM);  // ��ʼ���������!
                // �ȴ��Է����
                socketRetStatus = RecvInt(socketconfig);
                if (socketRetStatus != 1000) break;

                for (size_t iter = 1; iter <= 2; iter++)
                {
                    // ִ��
                    start_time = system_clock::now();
                    // ����ִ���������ͻ���
                    SendInt(socketconfig, 1001);
                    // ִ�м�������(��������)
                    // ��
                    // ���ͼ�������
                    SendFloatPtr(socketconfig, locFloatData, SGLE_DATANUM);
                    // �ȴ��Է����
                    socketRetStatus = RecvFloatPtr(socketconfig, rmtFloatData);
                    // ��֤���һ����
                    retValue = rmtFloatData[SGLE_DATANUM - 1];
                    // �������
                    end_time = system_clock::now();
                    diff = end_time - start_time;
                    durationSecond = diff.count();
                    // ��ʾ
                    wcout << L"��" << iter << L"��: ��ʱ(��): " << durationSecond;
                    wcout << L"; ���Ϊ(Ӧ��Ϊ128M): " << fixed << retValue << endl;
                    // ����ͬ��ָ��
                    socketRetStatus = RecvInt(socketconfig);
                    if (socketRetStatus != 1002) break;
                }
            }
            else if (socket_type == 2)
            {
                // ���׼��
                // �������Է������ĳ�ʼ������
                socketRetStatus = RecvInt(socketconfig);
                if (socketRetStatus != 1000) break;
                // ������ʼ��
                retValue = init_arithmetic(locFloatData, (float)SGLE_DATANUM, SGLE_DATANUM);  // ��ʼ���������!
                // ���ͼ������
                SendInt(socketconfig, 1000);

                for (size_t iter = 1; iter <= 2; iter++)
                {
                    // ִ��
                    start_time = system_clock::now();
                    // �������Է�������ִ������
                    socketRetStatus = RecvInt(socketconfig);
                    // ִ�м�������(��������)
                    // ��
                    // ���ͼ�������
                    SendFloatPtr(socketconfig, locFloatData, SGLE_DATANUM);
                    // �ȴ��Է����
                    socketRetStatus = RecvFloatPtr(socketconfig, rmtFloatData);
                    // ��֤���һ����
                    retValue = rmtFloatData[SGLE_DATANUM - 1];
                    // �������
                    end_time = system_clock::now();
                    diff = end_time - start_time;
                    durationSecond = diff.count();
                    // ��ʾ
                    wcout << L"��" << iter << L"��: ��ʱ(��): " << durationSecond;
                    wcout << L"; ���Ϊ(Ӧ��Ϊ64M): " << fixed << retValue << endl;
                    // ����ͬ��ָ��
                    SendInt(socketconfig, 1002);
                }
            }

            // �ͷŶ��ڴ�
            delete[] socket_ip;
            // �ر�Socket
            CloseSocket(socketconfig);
            // �ر�Cuda
            // if (main_entered == 4) cudaStatus = releaseCuda();  // �ͷ�CUDA
            break;
        default:
            durationSecond = 0;
            retValue = 0;
        }
    }

InitError:
    system("pause");
    return 0;
}