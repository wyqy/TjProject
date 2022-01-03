#include "speed_sort.h"

__declspec(align(MEMORY_ALIGNED)) float disorderData[DUAL_DATANUM];
__declspec(align(MEMORY_ALIGNED)) float rawFloatData[DUAL_DATANUM];
__declspec(align(MEMORY_ALIGNED)) float locFloatData[SGLE_DATANUM];
__declspec(align(MEMORY_ALIGNED)) float rmtFloatData[SGLE_DATANUM];

int main()
{
    // ���������������
    int main_entered;
    bool isValid = 0;
    float retValue = 0;
    double durationSecond = 0;
    double computationSecond = 0;

    // ��ʱ����
    system_clock::time_point start_time;
    system_clock::time_point end_time;
    duration<double> diff;
    wcout.imbue(locale("chs"));

    // �������
    int socket_type;                        // ͨ������: 1 = ������; 2 = �ͻ���
    char* socket_ip;                        // Socket IP
    u_short socket_port;                    // Socket Port
    double sockLatency;
    SocketConfig* socketconfig;

    // �������
    float rmtDurationSecond = 0.0;
    int socketRetStatus = 0;

    // cuda����
    cudaError_t cudaStatus;


    // ����׼��
    wcout << L"��ʼ��, ���Ժ�...";
    cudaStatus = initialCuda(0, rawFloatData, DUAL_DATANUM, locFloatData, SGLE_DATANUM);  // ��ʼ��CUDA, ���ڶ�GPUϵͳ���޸�!
    if (cudaStatus != cudaSuccess) goto InitError;
    system("cls");

    while (true)
    {
        wcout << L"�����ȸ����������������: \n\
                    1. ԭʼ����; \n\
                    2. ��������; \n\
                    3. �������; \n\
                    4. ����������; \n\
                    5. �˳�. \n\
                    ������ѡ��: ";
        wcin >> main_entered;
        if (main_entered == 5) break;

        switch (main_entered)
        {
        case 1:
            // �������, ���ֻʹ�ô˴ε������������ ,��ͬ
            wcout << L"�������, ���Ժ�...";
            retValue = init_arithmetic(disorderData, (float)1, DUAL_DATANUM);
            retValue = init_random(disorderData, DUAL_DATANUM);  // �����ʼ������!
            wcout << L"���!" << endl;
            for (size_t iter = 1; iter <= 5; iter++)
            {
                // ��ԭ˳��
                wcout << L"��������, ���Ժ�...";
                memcpy(rawFloatData, disorderData, DUAL_DATANUM * sizeof(float));
                // ִ��
                wcout << L"���! ������... ";
                start_time = system_clock::now();
                retValue = sortNaive(rawFloatData, (size_t)DUAL_DATANUM);  // ԭʼ����ʵ��
                end_time = system_clock::now();
                diff = end_time - start_time;
                durationSecond = diff.count();
                wcout << L"��" << iter << L"��: ��ʱ(��): " << durationSecond << L"; ��֤��...";
                // ��֤
                retValue = postProcess(rawFloatData, DUAL_DATANUM);
                isValid = validSort(rawFloatData, (size_t)DUAL_DATANUM);  // ��֤
                if (isValid) wcout << L"���! ��������ȷ! " << fixed << retValue << endl;
                else wcout << L"���! ����������! " << fixed << retValue << endl;
            }
            break;
        case 2:
            // �������
            wcout << L"�������, ���Ժ�...";
            retValue = init_arithmetic(disorderData, (float)1, DUAL_DATANUM);
            retValue = init_random(disorderData, DUAL_DATANUM);  // �����ʼ������!
            wcout << L"���!" << endl;
            for (size_t iter = 1; iter <= 5; iter++)
            {
                // ��ԭ˳��
                wcout << L"��������, ���Ժ�...";
                memcpy(rawFloatData, disorderData, DUAL_DATANUM * sizeof(float));
                // ִ��
                wcout << L"���! ������... ";
                start_time = system_clock::now();
                retValue = sortSpeedup(rawFloatData, (size_t)DUAL_DATANUM);  // ���ٴ���ʵ��
                end_time = system_clock::now();
                diff = end_time - start_time;
                durationSecond = diff.count();
                wcout << L"��" << iter << L"��: ��ʱ(��): " << durationSecond << L"; ��֤��...";
                // ��֤
                retValue = postSpeedup(rawFloatData, DUAL_DATANUM);
                isValid = validSort(rawFloatData, (size_t)DUAL_DATANUM);  // ��֤
                if (isValid) wcout << L"���! ��������ȷ! " << fixed << retValue << endl;
                else wcout << L"���! ����������! " << fixed << retValue << endl;
            }
            break;
        case 3:
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

            // �������
            wcout << L"�������, ���Ժ�...";
            if (socket_type == 1) retValue = init_arithmetic(disorderData, (float)1, SGLE_DATANUM);
            else if(socket_type == 2) retValue = init_arithmetic(disorderData, (float)SGLE_DATANUM, SGLE_DATANUM);
            retValue = init_random(disorderData, SGLE_DATANUM);  // �����ʼ������!
            wcout << L"���!" << endl;

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
                memcpy(locFloatData, disorderData, SGLE_DATANUM * sizeof(float));
                // �ȴ��Է����
                socketRetStatus = RecvInt(socketconfig);
                if (socketRetStatus != 1000) break;

                for (size_t iter = 1; iter <= 5; iter++)
                {
                    // ִ��
                    start_time = system_clock::now();
                    // ����ִ���������ͻ���
                    SendInt(socketconfig, 1001);
                    // ִ�м�������
                    retValue = sortSpeedup(locFloatData, (size_t)SGLE_DATANUM);  // ���ٴ���ʵ��
                    // ��¼���ؼ���ʱ��
                    end_time = system_clock::now();
                    diff = end_time - start_time;
                    computationSecond = diff.count();
                    // �ȴ��Է����
                    socketRetStatus = RecvFloatPtr(socketconfig, rmtFloatData);
                    // �ϲ�
                    retValue = sortMerge(rawFloatData, locFloatData, SGLE_DATANUM, rmtFloatData, SGLE_DATANUM);
                    // �������
                    end_time = system_clock::now();
                    diff = end_time - start_time;
                    durationSecond = diff.count();
                    // ��ʾ�ٶ�
                    wcout << L"��" << iter << L"��: ���ؼ����ʱ(��): " << computationSecond << L"; �ܺ�ʱ(��): " << durationSecond << L"; ��֤��...";
                    // ��֤���
                    retValue = postSpeedup(rawFloatData, DUAL_DATANUM);
                    isValid = validSort(rawFloatData, (size_t)DUAL_DATANUM);  // ��֤
                    if (isValid) wcout << L"���! ��������ȷ! " << fixed << retValue << endl;
                    else wcout << L"���! ����������! " << fixed << retValue << endl;
                    // �������ս��
                    SendFloat(socketconfig, (float)durationSecond);
                    if (isValid) SendInt(socketconfig, 1501);
                    else SendInt(socketconfig, 1502);
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
                memcpy(locFloatData, disorderData, SGLE_DATANUM * sizeof(float));
                // ���ͼ������
                SendInt(socketconfig, 1000);

                for (size_t iter = 1; iter <= 5; iter++)
                {
                    // ִ��
                    start_time = system_clock::now();
                    // �������Է�������ִ������
                    socketRetStatus = RecvInt(socketconfig);
                    if (socketRetStatus != 1001) break;
                    // ִ�м�������
                    retValue = sortSpeedup(locFloatData, (size_t)SGLE_DATANUM);  // ���ٴ���ʵ��
                    // ���ͼ�������
                    SendFloatPtr(socketconfig, locFloatData, SGLE_DATANUM);
                    // �������
                    end_time = system_clock::now();
                    diff = end_time - start_time;
                    durationSecond = diff.count();
                    // ��ʾ�����ٶ�
                    wcout << L"��" << iter << L"��: ������ʱ(��): " << durationSecond;
                    // �ȴ��Է����
                    rmtDurationSecond = RecvFloat(socketconfig);
                    socketRetStatus = RecvInt(socketconfig);
                    // ��ʾ���ս��
                    wcout << L"; �����ܺ�ʱ: " << fixed << rmtDurationSecond;
                    if (socketRetStatus == 1501) wcout << L", ��������ȷ! " << endl;
                    else if (socketRetStatus == 1502) wcout << L", ����������! " << endl;
                    // ����ͬ��ָ��
                    SendInt(socketconfig, 1002);
                }
            }

            delete[] socket_ip;  // �ͷŶ��ڴ�
            CloseSocket(socketconfig);  // �ر�Socket
            break;
        case 4:
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
                SendInt(socketconfig, 2000);
                printf("\nSuccess Send 2000");
                // ������ʼ��
                retValue = init_arithmetic(locFloatData, (float)1, SGLE_DATANUM);  // ��ʼ���������!
                printf("\nSuccess Inited");
                // �ȴ��Է����
                socketRetStatus = RecvInt(socketconfig);
                if (socketRetStatus != 2000) break;
                printf("\nSuccess Recv 2000");

                for (size_t iter = 1; iter <= 2; iter++)
                {
                    // ִ��
                    start_time = system_clock::now();
                    // ����ִ���������ͻ���
                    SendInt(socketconfig, 2001);
                    printf("\nSuccess Send 2001");
                    // ���ͼ�������
                    SendFloatPtr(socketconfig, locFloatData, SGLE_DATANUM);
                    printf("\nSuccess Send!");
                    // �ȴ��Է����
                    socketRetStatus = RecvFloatPtr(socketconfig, rmtFloatData);
                    printf("\nSuccess Recv!\n");
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
                    if (socketRetStatus != 2002) break;
                    printf("\nSuccess Recv 2002");
                }
            }
            else if (socket_type == 2)
            {
                // ���׼��
                // �������Է������ĳ�ʼ������
                socketRetStatus = RecvInt(socketconfig);
                if (socketRetStatus != 2000) break;
                printf("\nSuccess Recv 2000");
                // ������ʼ��
                retValue = init_arithmetic(locFloatData, (float)SGLE_DATANUM, SGLE_DATANUM);  // ��ʼ���������!
                // ���ͼ������
                SendInt(socketconfig, 2000);
                printf("\nSuccess Send 2000");

                for (size_t iter = 1; iter <= 2; iter++)
                {
                    // ִ��
                    start_time = system_clock::now();
                    // �������Է�������ִ������
                    socketRetStatus = RecvInt(socketconfig);
                    if (socketRetStatus != 2001) break;
                    printf("\nSuccess Recv 2001");
                    // ���ͼ�������
                    SendFloatPtr(socketconfig, locFloatData, SGLE_DATANUM);
                    printf("\nSuccess Send!");
                    // �ȴ��Է����
                    socketRetStatus = RecvFloatPtr(socketconfig, rmtFloatData);
                    printf("\nSuccess Recv!\n");
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
                    SendInt(socketconfig, 2002);
                    printf("\nSuccess Send 2002");
                }
            }

            delete[] socket_ip;  // �ͷŶ��ڴ�
            CloseSocket(socketconfig);  // �ر�Socket
            printf("\n");
            break;
        default:
            durationSecond = 0;
            retValue = 0;
        }
    }

InitError:
    cudaStatus = releaseCuda();  // �ͷ�CUDA
    system("pause");
    return 0;
}