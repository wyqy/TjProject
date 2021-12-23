#include "socket_sum.h"
#pragma comment(lib,"ws2_32.lib")


// ��������
// ����
SocketConfig::SocketConfig(int type, const char* ip, u_short port)
{
    // Socket����
    socket_type = type;
    socket_port = port;
    // ��ȫ�ַ���
    socket_ip = new char[16];  // 15 + \0 + 2
    strcpy_s(socket_ip, 16, ip);
    socket_ip[15] = '\0';
    // ��ʼ������
    // ******************
    sockInitSignal = nullptr;
    sockSendSignal = nullptr;
    sockRecvSignal = nullptr;
    // ******************
    sockCommandSignal = nullptr;
    sockPauseRecvSignal = nullptr;
    socketThread = nullptr;
    recvThread = nullptr;
    // ******************
    sockLatency = 0.0;
    sendDataType = 0;
    sendDataLen = 0;
    recvDataType = 0;
    recvDataLen = 0;
    // ******************
    socketThreadID = 0;
    recvThreadID = 0;
    isSelfSockBackendInited = false;
    isClntSockBackendInited = false;
    selfSockBackend = 0;
    clntSockBackend = 0;
    memset(&(selfSockAddr), 0, sizeof(selfSockAddr));  // pad with 0
    memset(&(clntSockAddr), 0, sizeof(clntSockAddr));  // pad with 0
}
// ����
SocketConfig::~SocketConfig(void)
{
    delete[] socket_ip;
}
// ��ȡ
int SocketConfig::getType(void)
{
    return socket_type;
}
u_short SocketConfig::getPort(void)
{
    return socket_port;
}
const char* SocketConfig::getIP(void)
{
    return (const char*)socket_ip;
}


// ���߳� - ����&����
DWORD WINAPI socketThreadFunc(LPVOID thParameter)
{
    SocketConfig* para = (SocketConfig*)thParameter;

    int socketSendRet;
    int winsockStatus;
    BOOL winapiStatus;
    DWORD threadStatus;

    bool isWorking = true;
    int mainCommand;

    // ��ʼ��DLL
    WSADATA wsaData;
    WORD DllVersion = MAKEWORD(2, 2);
    winsockStatus = WSAStartup(DllVersion, &wsaData);
    if (winsockStatus) SocketErrorExit(para);
    // �����ź���
    para->sockCommandSignal = CreateSemaphore(NULL, 0, 1, NULL);
    para->sockPauseRecvSignal = CreateSemaphore(NULL, 0, 1, NULL);

    // �����׽���
    para->selfSockBackend = socket(
        PF_INET,        // IPv4
        SOCK_STREAM,    // oriented for connection
        IPPROTO_TCP);   // TCP
    // �����׽���(IPv4)
    para->selfSockAddr.sin_family = PF_INET;                                                  // IPv4
    winsockStatus = inet_pton(PF_INET, para->getIP(), &(para->selfSockAddr.sin_addr.s_addr));  // IP
    if (winsockStatus != 1) SocketErrorExit(para);
    para->selfSockAddr.sin_port = htons(para->getPort());                                    // port
    // �������߳�
    para->recvThread = CreateThread(
        NULL,                           // default security attributes
        0,                              // use default stack size
        recvThreadFunc,                 // thread function
        para,                           // argument to thread function
        CREATE_SUSPENDED,               // use default creation flags. 0 means the thread will be run at once CREATE_SUSPENDED
        &(para->recvThreadID));    // default thread id

    // ��������: �Ӳ�����;
    // �����ĸ��ź���: ���ͺͽ���, ���̻߳��Ѻ��ⲿ��ȡ;
    // ע���ź����ɴ����߸�������, �����ⲿ�����ͷ�!
    // ����һ��ֻ����, ������������;
    // ����һ��ֻд��, ����֪ͨ��Ϣ��ʽ;
    // 
    // ��ʼ����Ͻ��ָ�����(recv)�ź���
    // ƽʱ���ڲ��Ͻ���(����)��״̬, ������Ӧ����(˫��?)
    // ˫�߳�: ���������߳�, ���������߳�, �������̻߳�������(���˳�ʼ���ͽ���ͨ��)
    //      ���߳�: ƽʱ��ͣ�߳�, ����������������,
    //              ������ϻָ����Ͳ����ź���;
    //              ����ʱ�ȴ���������ٽ������߳�, Ȼ���ͷ���Դ���˳�����;
    //      ���߳�: ƽʱ��������, �����յ���Ϣ,
    //              ���趨��Ϣ��ʽ, Ȼ�������Ϣ,
    //              ������ϻָ����Ͳ����ź���;
    // 
    // ����socket����
    if (para->getType() == 1)  // ������
    {
        // ��ʼ��������, ����ͻ��˽�������
        initServer(para);
        // ׼�����߳�
        threadStatus = ResumeThread(para->recvThread);
        if (threadStatus == -1) SocketErrorExit(para);
        // �ͷ��ź���
        winapiStatus = ReleaseSemaphore(para->sockSendSignal, (LONG)1, NULL);
        if (!winapiStatus) SocketErrorExit(para);
        winapiStatus = ReleaseSemaphore(
            para->sockInitSignal,   // signal
            (LONG)1,                // incremental
            NULL);                  // last value
        if (!winapiStatus) SocketErrorExit(para);

        // ����ѭ��
        while (isWorking)
        {
            // �ȴ������
            threadStatus = WaitForSingleObject(para->sockCommandSignal, INFINITE);
            if (threadStatus) SocketErrorExit(para);

            // ���Ѻ�
            // �����ź���
            para->sockCommandSignal = SocketResetSemaphore(para, para->sockCommandSignal, 0, 1);
            // ��ȡ����
            mainCommand = para->sendDataType;
            switch (mainCommand)
            {
            case COMMAND_FLOAT:  // ����float
                // ��������ͷ
                socketSendRet = send(para->clntSockBackend, (const char*)&mainCommand, sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // �������ݳ���
                socketSendRet = send(para->clntSockBackend, (const char*)&(para->sendDataLen), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // ����float��ʽbuffer����������
                socketSendRet = send(para->clntSockBackend, (const char*)&(para->sendBuffer_float), sizeof(float), NULL);
                if (socketSendRet != sizeof(float)) SocketErrorExit(para);
                // �ͷ��ź���
                winapiStatus = ReleaseSemaphore(para->sockSendSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                break;
            case COMMAND_INT:  // ����int
                // ��������ͷ
                socketSendRet = send(para->clntSockBackend, (const char*)&mainCommand, sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // �������ݳ���
                socketSendRet = send(para->clntSockBackend, (const char*)&(para->sendDataLen), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // ����int��ʽbuffer����������
                socketSendRet = send(para->clntSockBackend, (const char*)&(para->sendBuffer_int), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // �ͷ��ź���
                winapiStatus = ReleaseSemaphore(para->sockSendSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                break;
            case COMMAND_RETEND:  // ���ͽ���ȷ��
                // ���ͽ���ȷ�ϰ�
                socketSendRet = send(para->clntSockBackend, (const char*)&mainCommand, sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // �ͷ��ź���
                winapiStatus = ReleaseSemaphore(para->sockSendSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                break;
            case COMMAND_ACKEND:  // ����ִ��
                // ����ѭ��
                isWorking = false;
                break;
            }
        }
    }
    else if (para->getType() == 2)  // �ͻ���
    {
        // ��ʼ���ͻ���, �����������������
        initClient(para);
        // ׼�����߳�
        threadStatus = ResumeThread(para->recvThread);
        if (threadStatus == -1) SocketErrorExit(para);
        // �ͷ��ź���
        winapiStatus = ReleaseSemaphore(para->sockSendSignal, (LONG)1, NULL);
        if (!winapiStatus) SocketErrorExit(para);
        winapiStatus = ReleaseSemaphore(para->sockInitSignal, (LONG)1, NULL);
        if (!winapiStatus) SocketErrorExit(para);

        // ����ѭ��
        while (isWorking)
        {
            // �ȴ������
            threadStatus = WaitForSingleObject(para->sockCommandSignal, INFINITE);
            if (threadStatus) SocketErrorExit(para);

            // ���Ѻ�
            // �����ź���
            para->sockCommandSignal = SocketResetSemaphore(para, para->sockCommandSignal, 0, 1);
            // ��ȡ����
            mainCommand = para->sendDataType;
            switch (mainCommand)
            {
            case COMMAND_FLOAT:  // ����float
                // ��������ͷ
                socketSendRet = send(para->selfSockBackend, (const char*)&mainCommand, sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // �������ݳ���
                socketSendRet = send(para->selfSockBackend, (const char*)&(para->sendDataLen), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // ����float��ʽbuffer����������
                socketSendRet = send(para->selfSockBackend, (const char*)&(para->sendBuffer_float), sizeof(float), NULL);
                if (socketSendRet != sizeof(float)) SocketErrorExit(para);
                // �ͷ��ź���
                winapiStatus = ReleaseSemaphore(para->sockSendSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                break;
            case COMMAND_INT:  // ����int
                // ��������ͷ
                socketSendRet = send(para->selfSockBackend, (const char*)&mainCommand, sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // �������ݳ���
                socketSendRet = send(para->selfSockBackend, (const char*)&(para->sendDataLen), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // ����int��ʽbuffer����������
                socketSendRet = send(para->selfSockBackend, (const char*)&(para->sendBuffer_int), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // �ͷ��ź���
                winapiStatus = ReleaseSemaphore(para->sockSendSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                break;
            case COMMAND_REQEND:  // ���ͽ�������
                // ���ͽ��������
                socketSendRet = send(para->selfSockBackend, (const char*)&mainCommand, sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // �ͷ��ź���
                winapiStatus = ReleaseSemaphore(para->sockSendSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                break;
            case COMMAND_ACKEND:  // ����ִ��
                // ����ѭ��
                isWorking = false;
                break;
            }
        }
    }
    else ExitProcess((DWORD)-1);

    // ����FIN��
    // shutdown(para->selfSockBackend, SD_BOTH);
    // �ر��ź���
    winapiStatus = CloseHandle(para->sockCommandSignal);
    if (!winapiStatus) SocketErrorExit(para);
    winapiStatus = CloseHandle(para->sockPauseRecvSignal);
    if (!winapiStatus) SocketErrorExit(para);
    // �رվ������
    winapiStatus = CloseHandle(para->recvThread);
    if (!winapiStatus) SocketErrorExit(para);
    // �ر��׽���
    if (para->isSelfSockBackendInited) closesocket(para->selfSockBackend);
    if (para->isClntSockBackendInited) closesocket(para->clntSockBackend);
    para->isSelfSockBackendInited = false;
    para->isClntSockBackendInited = false;

    // �ͷ�DLL
    WSACleanup();
    return (DWORD)0;
}


// ���߳� - ����
DWORD WINAPI recvThreadFunc(LPVOID thParameter)
{
    SocketConfig* para = (SocketConfig*)thParameter;

    int socketSendRet;
    BOOL winapiStatus;
    DWORD threadStatus;

    bool isWorking = true;
    int recvType;

    // Ĭ���Ѿ���ʼ�����
    // ��Ҫ��ʼ���κζ��ڴ�, ���߶���
    // ����socket����
    if (para->getType() == 1)  // ������
    {
        // ����ѭ��
        while (isWorking)
        {
            // �����������
            socketSendRet = recv(para->clntSockBackend, (char*)&recvType, sizeof(int), NULL);
            if (socketSendRet == 0 || socketSendRet == -1) goto RECV_END;
            // else if (socketSendRet != sizeof(int)) SocketErrorExit(para);
            // ���պ�
            // ��λ֪ͨλ
            para->recvDataType = recvType;
            // ��ȡ��ʽ
            switch (recvType)
            {
            case COMMAND_FLOAT:  // ����float
                // �������ݳ���
                socketSendRet = recv(para->clntSockBackend, (char*)&(para->recvDataLen), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // ����ָ�����ȵ����ݵ�float��ʽ��buffer
                socketSendRet = recv(para->clntSockBackend, (char*)&(para->recvBuffer_float), para->recvDataLen, NULL);
                if (socketSendRet != para->recvDataLen) SocketErrorExit(para);
                // �ͷ��ź���
                winapiStatus = ReleaseSemaphore(para->sockRecvSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                // �ȴ������̴������
                threadStatus = WaitForSingleObject(para->sockPauseRecvSignal, INFINITE);
                if (threadStatus) SocketErrorExit(para);
                para->sockPauseRecvSignal = SocketResetSemaphore(para, para->sockPauseRecvSignal, 0, 1);
                break;
            case COMMAND_INT:  // ����int
                // �������ݳ���
                socketSendRet = recv(para->clntSockBackend, (char*)&(para->recvDataLen), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // ����ָ�����ȵ����ݵ�int��ʽ��buffer
                socketSendRet = recv(para->clntSockBackend, (char*)&(para->recvBuffer_int), para->recvDataLen, NULL);
                if (socketSendRet != para->recvDataLen) SocketErrorExit(para);
                // �ͷ��ź���
                winapiStatus = ReleaseSemaphore(para->sockRecvSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                // �ȴ������̴������
                threadStatus = WaitForSingleObject(para->sockPauseRecvSignal, INFINITE);
                if (threadStatus) SocketErrorExit(para);
                para->sockPauseRecvSignal = SocketResetSemaphore(para, para->sockPauseRecvSignal, 0, 1);
                break;
            case COMMAND_REQEND:  // ��������ִ��
                // �ͷ��ź���
                winapiStatus = ReleaseSemaphore(para->sockRecvSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                // ����ѭ��
                isWorking = false;
                break;
            }
        }
    }
    else if (para->getType() == 2)  // �ͻ���
    {
        // ����ѭ��
        while (isWorking)
        {
            // �����������
            socketSendRet = recv(para->selfSockBackend, (char*)&recvType, sizeof(int), NULL);
            if (socketSendRet == 0 || socketSendRet == -1) goto RECV_END;
            // else if (socketSendRet != sizeof(int)) SocketErrorExit(para);
            // ���պ�
            // ��λ֪ͨλ
            para->recvDataType = recvType;
            // ��ȡ��ʽ
            switch (recvType)
            {
            case COMMAND_FLOAT:  // ����float
                // �������ݳ���
                socketSendRet = recv(para->selfSockBackend, (char*)&(para->recvDataLen), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // ����ָ�����ȵ����ݵ�float��ʽ��buffer
                socketSendRet = recv(para->selfSockBackend, (char*)&(para->recvBuffer_float), para->recvDataLen, NULL);
                if (socketSendRet != para->recvDataLen) SocketErrorExit(para);
                // �ͷ��ź���
                winapiStatus = ReleaseSemaphore(para->sockRecvSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                // �ȴ������̴������
                threadStatus = WaitForSingleObject(para->sockPauseRecvSignal, INFINITE);
                if (threadStatus) SocketErrorExit(para);
                para->sockPauseRecvSignal = SocketResetSemaphore(para, para->sockPauseRecvSignal, 0, 1);
                break;
            case COMMAND_INT:  // ����int
                // �������ݳ���
                socketSendRet = recv(para->selfSockBackend, (char*)&(para->recvDataLen), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // ����ָ�����ȵ����ݵ�int��ʽ��buffer
                socketSendRet = recv(para->selfSockBackend, (char*)&(para->recvBuffer_int), para->recvDataLen, NULL);
                if (socketSendRet != para->recvDataLen) SocketErrorExit(para);
                // �ͷ��ź���
                winapiStatus = ReleaseSemaphore(para->sockRecvSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                // �ȴ������̴������
                threadStatus = WaitForSingleObject(para->sockPauseRecvSignal, INFINITE);
                if (threadStatus) SocketErrorExit(para);
                para->sockPauseRecvSignal = SocketResetSemaphore(para, para->sockPauseRecvSignal, 0, 1);
                break;
            case COMMAND_RETEND:  // ��������ִ��
                // �ͷ��ź���
                winapiStatus = ReleaseSemaphore(para->sockRecvSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                // ����ѭ��
                isWorking = false;
                break;
            }
        }
    }
    else ExitProcess((DWORD)-1);

RECV_END:
    return (DWORD)0;
}


SocketConfig* InitSocket(int type, const char* ip, u_short port)
{
    // ����
    SocketConfig* retConfig = new SocketConfig(type, ip, port);
    // ׼���ź���
    retConfig->sockInitSignal = CreateSemaphore(NULL, 0, 1, NULL);
    retConfig->sockSendSignal = CreateSemaphore(NULL, 0, 1, NULL);
    retConfig->sockRecvSignal = CreateSemaphore(NULL, 0, 1, NULL);
    // ׼�����߳�
    retConfig->socketThread = CreateThread(NULL, 0, socketThreadFunc, retConfig, CREATE_SUSPENDED, &(retConfig->socketThreadID));
    if (retConfig->socketThread == 0) SocketErrorExit(retConfig);
    else
    {
        DWORD threadStatus = ResumeThread(retConfig->socketThread);
        if (threadStatus == -1) SocketErrorExit(retConfig);
    }
    return retConfig;
}


double InitLatency(SocketConfig* socketconfig)
{
    WaitAndResetInitSig(socketconfig);
    return socketconfig->sockLatency;
}


void SendCommand(SocketConfig* socketconfig, int command, int len, void* contentPtr)  // nullptr
{
    // ȷ���ϴη��ͳɹ�
    WaitAndResetSendSig(socketconfig);

    // ���д˴η���
    socketconfig->sendDataType = command;
    switch (socketconfig->sendDataType)
    {
    case COMMAND_FLOAT:
        // ����
        socketconfig->sendDataLen = sizeof(float);
        // ��Ϣ
        if (contentPtr != nullptr) socketconfig->sendBuffer_float = *((float*)contentPtr);
        break;
    case COMMAND_INT:
        // ����
        socketconfig->sendDataLen = sizeof(int);
        // ��Ϣ
        if (contentPtr != nullptr) socketconfig->sendBuffer_int = *((int*)contentPtr);
        break;
    case COMMAND_REQEND:
        break;
    }
    BOOL winapiStatus = ReleaseSemaphore(socketconfig->sockCommandSignal, (LONG)1, NULL);
    if (!winapiStatus) SocketErrorExit(socketconfig);
}


void RecvCommand(SocketConfig* socketconfig, int* type, int* len, void* contentPtr)  // nullptr
{
    // ��������ȴ�����
    WaitAndResetRecvSig(socketconfig);

    // �������ݸ�ʽ, ��С
    *type = socketconfig->recvDataType;
    *len = socketconfig->recvDataLen;
    switch (socketconfig->recvDataType)
    {
    case COMMAND_FLOAT:
        if (contentPtr != nullptr && socketconfig->recvDataLen == sizeof(float)) *(float*)contentPtr = socketconfig->recvBuffer_float;
        break;
    case COMMAND_INT:
        if (contentPtr != nullptr && socketconfig->recvDataLen == sizeof(int)) *(int*)contentPtr = socketconfig->recvBuffer_int;
        break;
    case COMMAND_REQEND:
        break;
    }
    BOOL winapiStatus = ReleaseSemaphore(socketconfig->sockPauseRecvSignal, (LONG)1, NULL);
    if (!winapiStatus) SocketErrorExit(socketconfig);
}


void EndCommand(SocketConfig* socketconfig)
{
    BOOL winapiStatus;

    // ����ϴη����ź���
    WaitAndResetSendSig(socketconfig);

    // �����
    if (socketconfig->getType() == 1)  // ������
    {
        // �ȴ��ͻ��˷��ͽ��������, �Խ������߳�
        WaitAndResetRecvSig(socketconfig);
        if (socketconfig->recvDataType == COMMAND_REQEND)
        {
            // ���ͽ���ȷ�ϰ�
            socketconfig->sendDataType = COMMAND_RETEND;
            winapiStatus = ReleaseSemaphore(socketconfig->sockCommandSignal, (LONG)1, NULL);
            if (!winapiStatus) SocketErrorExit(socketconfig);
            // ȷ�����ͳɹ�
            WaitAndResetSendSig(socketconfig);
            // �������߳�
            socketconfig->sendDataType = COMMAND_ACKEND;
            winapiStatus = ReleaseSemaphore(socketconfig->sockCommandSignal, (LONG)1, NULL);
            if (!winapiStatus) SocketErrorExit(socketconfig);
        }
    }
    else if (socketconfig->getType() == 2)  // �ͻ���
    {
        // ���ͽ��������
        socketconfig->sendDataType = COMMAND_REQEND;
        winapiStatus = ReleaseSemaphore(socketconfig->sockCommandSignal, (LONG)1, NULL);
        if (!winapiStatus) SocketErrorExit(socketconfig);
        // ȷ�����ͳɹ�
        WaitAndResetSendSig(socketconfig);
        // �ȴ����������ͽ���ȷ�ϰ�, �Խ������߳�
        WaitAndResetRecvSig(socketconfig);
        if (socketconfig->recvDataType == COMMAND_RETEND)
        {
            // �������߳�
            socketconfig->sendDataType = COMMAND_ACKEND;
            winapiStatus = ReleaseSemaphore(socketconfig->sockCommandSignal, (LONG)1, NULL);
            if (!winapiStatus) SocketErrorExit(socketconfig);
        }
    }
    else ExitProcess((DWORD)-1);
}


void CloseSocket(SocketConfig* socketconfig)
{
    BOOL winapiStatus;
    // �ر��ź���
    winapiStatus = CloseHandle(socketconfig->sockInitSignal);
    if (!winapiStatus) SocketErrorExit(socketconfig);
    winapiStatus = CloseHandle(socketconfig->sockSendSignal);
    if (!winapiStatus) SocketErrorExit(socketconfig);
    winapiStatus = CloseHandle(socketconfig->sockRecvSignal);
    if (!winapiStatus) SocketErrorExit(socketconfig);
    // �ر��߳�
    winapiStatus = CloseHandle(socketconfig->socketThread);
    if (!winapiStatus) SocketErrorExit(socketconfig);
    // ���ٶ���
    delete socketconfig;
}


void WaitAndResetInitSig(SocketConfig* socketconfig)
{
    DWORD threadStatus = WaitForSingleObject(socketconfig->sockInitSignal, MAXWAITING_MS);
    if (threadStatus) SocketErrorExit(socketconfig);
    socketconfig->sockInitSignal = SocketResetSemaphore(socketconfig, socketconfig->sockInitSignal, 0, 1);
}


void WaitAndResetSendSig(SocketConfig* socketconfig)
{
    DWORD threadStatus = WaitForSingleObject(socketconfig->sockSendSignal, MAXWAITING_MS);
    if (threadStatus) SocketErrorExit(socketconfig);
    socketconfig->sockSendSignal = SocketResetSemaphore(socketconfig, socketconfig->sockSendSignal, 0, 1);
}


void WaitAndResetRecvSig(SocketConfig* socketconfig)
{
    DWORD threadStatus = WaitForSingleObject(socketconfig->sockRecvSignal, MAXWAITING_MS);
    if (threadStatus) SocketErrorExit(socketconfig);
    socketconfig->sockRecvSignal = SocketResetSemaphore(socketconfig, socketconfig->sockRecvSignal, 0, 1);
}


void initServer(SocketConfig* socketconfig)
{
    system_clock::time_point start_time;
    system_clock::time_point end_time;
    duration<double> diff;
    int winsockStatus;

    // ���׽���
    winsockStatus = bind(socketconfig->selfSockBackend, (SOCKADDR*)&(socketconfig->selfSockAddr), sizeof(SOCKADDR));
    if (winsockStatus == -1) SocketErrorExit(socketconfig);
    socketconfig->isSelfSockBackendInited = true;
    // �������״̬
    winsockStatus = listen(socketconfig->selfSockBackend, BACKLOG);
    if (winsockStatus == -1) SocketErrorExit(socketconfig);
    // ���տͻ�������
    int nSize = sizeof(SOCKADDR);
    socketconfig->clntSockBackend = accept(socketconfig->selfSockBackend, (SOCKADDR*)&(socketconfig->clntSockAddr), &nSize);
    socketconfig->isClntSockBackendInited = true;
    // ��ʼ��ʱ
    start_time = system_clock::now();
    // ��ͻ��˷��ͳ�ʼ������
    int handServer = 123;
    send(socketconfig->clntSockBackend, (const char*)&handServer, sizeof(int), NULL);
    // ���տͻ�����Ӧ
    recv(socketconfig->clntSockBackend, (char*)&handServer, sizeof(int), NULL);
    // ֹͣ��ʱ
    end_time = system_clock::now();
    // ��ͻ����ٴη�������
    send(socketconfig->clntSockBackend, (const char*)&handServer, sizeof(int), NULL);

    diff = end_time - start_time;
    socketconfig->sockLatency = diff.count();
}


void initClient(SocketConfig* socketconfig)
{
    system_clock::time_point start_time;
    system_clock::time_point end_time;
    duration<double> diff;
    int winsockStatus;

    // ���������������
    winsockStatus = connect(socketconfig->selfSockBackend, (SOCKADDR*)&(socketconfig->selfSockAddr), sizeof(SOCKADDR));
    if (winsockStatus == -1) SocketErrorExit(socketconfig);
    socketconfig->isSelfSockBackendInited = true;
    // ���շ�������ʼ������
    int handClient = 0;
    recv(socketconfig->selfSockBackend, (char*)&handClient, sizeof(int), NULL);
    // ��ʼ��ʱ
    start_time = system_clock::now();
    // �������������Ӧ
    send(socketconfig->selfSockBackend, (const char*)&handClient, sizeof(int), NULL);
    // �ٴν��շ���������
    recv(socketconfig->selfSockBackend, (char*)&handClient, sizeof(int), NULL);
    // ֹͣ��ʱ
    end_time = system_clock::now();

    diff = end_time - start_time;
    socketconfig->sockLatency = diff.count();
}


void SocketErrorExit(SocketConfig* socketconfig)
{
    // https://blog.csdn.net/makenothing/article/details/51198006

    LPVOID lpMsgBuf;
    HLOCAL lpDisplayBuf;

    // Retrieve the system error message for the last-error code
    DWORD dw = GetLastError();
    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR)&lpMsgBuf,
        0, NULL);

    // Display the error message and exit the process
    lpDisplayBuf = LocalAlloc(LMEM_ZEROINIT,
        (lstrlen((LPCTSTR)lpMsgBuf) + (size_t)40) * sizeof(TCHAR));
    if (lpDisplayBuf != NULL)
    {
        StringCchPrintf((LPTSTR)lpDisplayBuf,
            LocalSize(lpDisplayBuf),
            TEXT("[Error] failed with error %d: %s!"),
            dw, (LPCTSTR)lpMsgBuf);
        MessageBox(NULL, (LPCTSTR)lpDisplayBuf, TEXT("Error"), MB_OK);
    }

    // Release Sockets
    if (socketconfig->isSelfSockBackendInited) closesocket(socketconfig->selfSockBackend);
    if (socketconfig->isClntSockBackendInited) closesocket(socketconfig->clntSockBackend);

    LocalFree(lpMsgBuf);
    LocalFree(lpDisplayBuf);
    ExitProcess(dw);
}


HANDLE SocketResetSemaphore(SocketConfig* socketconfig, HANDLE handle, LONG sigInitialCount, LONG sigMaximumCount)
{
    // PHANDLE retPHandle = new HANDLE;
    if (handle != nullptr)
    {
        BOOL winapiStatus = CloseHandle(handle);
        if (!winapiStatus) SocketErrorExit(socketconfig);
    }
    HANDLE retHandle = CreateSemaphore(
        NULL,                    // lpSemaphoreAttributes
        sigInitialCount,         // lInitialCount
        sigMaximumCount,         // lMaximumCount
        NULL);                   // lpName  // CreateEvent(NULL,TRUE,FALSE)
    return retHandle;
}
