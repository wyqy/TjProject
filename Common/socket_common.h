#pragma once
#include "pj_common.h"

// ����ϵͳ��
#include <strsafe.h>
// �����׽��ֿ�
#include <WinSock2.h>
#include <ws2tcpip.h>
// ����ϵͳ��
#include <Windows.h>

// �����
// socket ����
#define BACKLOG 0x1000  // #define BACKLOG SOMAXCONN
// ���߳�ͬ�����Ź�
#define MAXWAITING_MS  ((DWORD)0xffff)
// ����/�������ݸ�ʽ
#define COMMAND_FLOAT   ((int)1812)
#define COMMAND_INT     ((int)1824)
#define COMMAND_FLARRAY ((int)2012)
#define COMMAND_REQEND  ((int)4396)
#define COMMAND_RETEND  ((int)4397)
#define COMMAND_ACKEND  ((int)4398)
// ���黺������С
#define BUFFER_FLARRAY  ((size_t)64*1200000)


// ���崫������
class SocketConfig
{
private:
    // ******************
    // �ⲿ�̴߳����ʼ������
    int socket_type;                        // ͨ������: 1 = ������; 2 = �ͻ���
    char* socket_ip;                        // Socket IP
    u_short socket_port;                    // Socket Port
    // ******************
public:
    // ******************
    // �ⲿ�߳�ͬ������
    HANDLE sockInitSignal;                  // ��ʼ�����ź���, �ⲿ���������, �ڲ��ָ�
    HANDLE sockSendSignal;                  // ���Ͳ����ź���, �ⲿ���������, �ڲ��ָ�
    HANDLE sockRecvSignal;                  // ���ղ����ź���, �ⲿ���������, �ڲ��ָ�
    // �ڲ��߳�ͬ������
    HANDLE sockCommandSignal;               // ������ź���, �ڲ����������, �ⲿ�ָ�
    HANDLE sockPauseRecvSignal;             // �ⲿ��ȡ�ź���, �ڲ����������, �ⲿ�ָ�
    HANDLE socketThread;                    // Socket���߳�, �ⲿֻ��
    HANDLE recvThread;                      // Socket���߳�, �ⲿֻ��
    // ******************
    // �����������
    double sockLatency;                     // ��ʼ��ʱ���ӳ�ָʾ, �ⲿֻ��
    int sendDataType;                       // ������Ϣ��ʽ, �ⲿҪ�ڻָ����̻߳����ź���ǰд��
    int sendDataLen;                        // ������Ϣ����, �ⲿҪ�ڻָ����̻߳����ź���ǰд��
    int recvDataType;                       // ֪ͨ���յ���Ϣ��ʽ, �ⲿֻ��
    int recvDataLen;                        // ֪ͨ���յ���Ϣ����, �ⲿֻ��
    // ******************
    // �ڲ��̲߳�������
    DWORD socketThreadID;                   // ���߳�Thread ID
    DWORD recvThreadID;                     // ���߳�THread ID
    bool isSelfSockBackendInited;           // ����Socket��Ϣ���ü���
    bool isClntSockBackendInited;           // (������) �ͻ���Socket��Ϣ���ü���
    SOCKET selfSockBackend;                 // ����Socket��Ϣ
    SOCKET clntSockBackend;                 // (������) �ͻ���Socket��Ϣ
    sockaddr_in selfSockAddr;               // ����Socket Addr��Ϣ
    SOCKADDR clntSockAddr;                  // (������) �ͻ���Socket Addr��Ϣ
    // ******************
    // ����buffer
    // ��������buffer
    float recvBuffer_float;
    int recvBuffer_int;
    float* recvBuffer_flarray;
    // ��������buffer
    float sendBuffer_float;
    int sendBuffer_int;
    const float* sendBuffer_flptr;
    // float* sendBuffer_flarray;
    // extern float sendBuffer_floatArray[SGLE_DATANUM];
    // ******************

    // ����
    SocketConfig(int type, const char* ip, u_short port);
    // ����
    ~SocketConfig(void);
    // ��ȡ
    int getType(void);
    u_short getPort(void);
    const char* getIP(void);
};

// �����ڲ��߳���ں���
DWORD WINAPI socketThreadFunc(LPVOID thParameter);
DWORD WINAPI recvThreadFunc(LPVOID thParameter);

// ******************
// ʹ��˵��:
// 1. �������򴦶��������ⲿȫ�ֱ���.
// 2. ������˳��ʹ��:
//      InitSocket();                               // ��ʼ��Socket�����
//      WaitAndResetInitSig();                      // �ȴ��ӳٲ���(�������ӳٲ������)
//      <�ɶ�ȡsockLatency����>
//      SendCommand(COMMAND_XXX, contentPtr);       // ��������(����������)
//      RecvCommand(COMMAND_XXX, len, contentPtr);  // ��������(������ȫ�����ݽ������)
//      <����ָ����˳������>
//      EndCommand();                               // �ر�����(����˻�ȴ��ͻ���, �ͻ��˼���������)
//      CloseSocket();                              // ����Socket�����
// 3. ��ò�Ҫ��ȡ / д�������ⲿȫ�ֱ���
// ******************
// ��ʽ˵��:
// ÿ�η��͵���Ϣ�ĵ�һ��int��ʾ��Ϣ��ʽ:
// (int)        ����
// 1812         float
// 1824         int
// �ͷ�����������ͬ!
// ******************
// ÿ�η��͵���Ϣ�ĵڶ���size_t��ʾ��Ϣ����;
// ******************
// ��������:
// (int)        ����
// 1812         ����float��
// 1824         ����int��
// ******************(����ʹ����������)******************
// 4396         ���ͽ��������
// 4397         ���ͽ���ȷ�ϰ�
// 4398         ��ʽ�˳�
// ����궨��
// ����(ȷ��ͬ����)�ⲿ����
// ***********************************************************************************************
// ************************************ֻʹ�ÿ��ڵİ˸�����!!!************************************
// ****************** ��ʼ��
SocketConfig* InitSocket(int type, const char* ip, u_short port);  // ��ʼ��, ����Socket�Ĺ�����ָ��, ��Ҫ��ǰ����(������Ҫ��ʼ��)һ��SocketConfig*���ͱ�������
double InitLatency(SocketConfig* socketconfig);  // ���˫���ӳ�
// ****************** ����, ��Ҫ���б�֤�����ڷ���ʱ������ָ��ָ����ڴ�(���������)
void SendFloat(SocketConfig* socketconfig, float source);
void SendInt(SocketConfig* socketconfig, int source);
void SendFloatPtr(SocketConfig* socketconfig, const float* source, size_t length);
// ****************** ����, ��Ҫ���б�֤�����������յ�����ͬ��������! ����ᷢ������!
float RecvFloat(SocketConfig* socketconfig);
int RecvInt(SocketConfig* socketconfig);
int RecvFloatPtr(SocketConfig* socketconfig, float* target);  // ���Ƶ�ָ��ָ�����ڴ�, ͬʱ���������С, ���б�֤�ռ��㹻
int RecvFloatPtr(SocketConfig* socketconfig);  // ֱ�ӷ��������С, ���б�֤ʹ��ʱ������յ�����!
// ****************** ����
void CloseSocket(SocketConfig* socketconfig);  // �ر�Socket, �ͷ��ڴ�(��˲���Ҫ�ͷ�SocketConfig*ָ��)
// ************************************ֻʹ�ÿ��ڵİ˸�����!!!************************************
// ***********************************************************************************************
// �����ⲿ������������
void SendCommand(SocketConfig* socketconfig, int command, int len, void* contentPtr = nullptr);  // ��������(��������), ����!
void RecvCommand(SocketConfig* socketconfig, int* type, int* len, void* contentPtr = nullptr);  // ��������(�Դ�����), ����!
void EndCommand(SocketConfig* socketconfig);  // ����ͨ��(˫������Ҫʹ��! ֱ��˫��������ͨ�Ų�ֹͣ����)
void WaitAndResetInitSig(SocketConfig* socketconfig);  // ��ʼ��ʱ����
void WaitAndResetSendSig(SocketConfig* socketconfig);  // ����ʱ����
void WaitAndResetRecvSig(SocketConfig* socketconfig);  // ����ʱ����

// ����ͨ�ź���
void initServer(SocketConfig* socketconfig);
void initClient(SocketConfig* socketconfig);

// �����̹߳�������
void SocketErrorExit(SocketConfig* socketconfig);
HANDLE SocketResetSemaphore(SocketConfig* socketconfig, HANDLE handle, LONG sigInitialCount, LONG sigMaximumCount);