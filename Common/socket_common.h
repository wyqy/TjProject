#pragma once
#include "pj_common.h"

// 调用系统库
#include <strsafe.h>
// 调用套接字库
#include <WinSock2.h>
#include <ws2tcpip.h>
// 调用系统库
#include <Windows.h>

// 定义宏
// socket 参数
#define BACKLOG 0x1000  // #define BACKLOG SOMAXCONN
// 多线程同步看门狗
#define MAXWAITING_MS  ((DWORD)0xfffff)
// 发送/接收数据格式
#define COMMAND_FLOAT   ((int)1812)
#define COMMAND_INT     ((int)1824)
#define COMMAND_FLARRAY ((int)2012)
#define COMMAND_REQEND  ((int)4396)
#define COMMAND_RETEND  ((int)4397)
#define COMMAND_ACKEND  ((int)4398)
// 数组缓冲区大小
#define BUFFER_FLARRAY  ((size_t)64*1200000)


// 定义传入类型
class SocketConfig
{
private:
    // ******************
    // 外部线程传入初始化参数
    int socket_type;                        // 通信类型: 1 = 服务器; 2 = 客户端
    char* socket_ip;                        // Socket IP
    u_short socket_port;                    // Socket Port
    // ******************
public:
    // ******************
    // 外部线程同步变量
    HANDLE sockInitSignal;                  // 初始操作信号量, 外部定义和重置, 内部恢复
    HANDLE sockSendSignal;                  // 发送操作信号量, 外部定义和重置, 内部恢复
    HANDLE sockRecvSignal;                  // 接收操作信号量, 外部定义和重置, 内部恢复
    // 内部线程同步变量
    HANDLE sockCommandSignal;               // 命令处理信号量, 内部定义和重置, 外部恢复
    HANDLE sockPauseRecvSignal;             // 外部读取信号量, 内部定义和重置, 外部恢复
    HANDLE socketThread;                    // Socket主线程, 外部只读
    HANDLE recvThread;                      // Socket子线程, 外部只读
    // ******************
    // 内外操作变量
    double sockLatency;                     // 初始化时的延迟指示, 外部只读
    int sendDataType;                       // 发送信息格式, 外部要在恢复主线程唤醒信号量前写入
    int sendDataLen;                        // 发送信息长度, 外部要在恢复主线程唤醒信号量前写入
    int recvDataType;                       // 通知接收的信息格式, 外部只读
    int recvDataLen;                        // 通知接收的信息长度, 外部只读
    // ******************
    // 内部线程操作数据
    DWORD socketThreadID;                   // 主线程Thread ID
    DWORD recvThreadID;                     // 子线程THread ID
    bool isSelfSockBackendInited;           // 自身Socket信息引用计数
    bool isClntSockBackendInited;           // (服务器) 客户端Socket信息引用计数
    SOCKET selfSockBackend;                 // 自身Socket信息
    SOCKET clntSockBackend;                 // (服务器) 客户端Socket信息
    sockaddr_in selfSockAddr;               // 自身Socket Addr信息
    SOCKADDR clntSockAddr;                  // (服务器) 客户端Socket Addr信息
    // ******************
    // 传输buffer
    // 接收数据buffer
    float recvBuffer_float;
    int recvBuffer_int;
    float* recvBuffer_flarray;
    // 发送数据buffer
    float sendBuffer_float;
    int sendBuffer_int;
    const float* sendBuffer_flptr;
    // float* sendBuffer_flarray;
    // extern float sendBuffer_floatArray[SGLE_DATANUM];
    // ******************

    // 构造
    SocketConfig(int type, const char* ip, u_short port);
    // 析构
    ~SocketConfig(void);
    // 读取
    int getType(void);
    u_short getPort(void);
    const char* getIP(void);
};

// 定义内部线程入口函数
DWORD WINAPI socketThreadFunc(LPVOID thParameter);
DWORD WINAPI recvThreadFunc(LPVOID thParameter);

// ******************
// 使用说明:
// 1. 在主程序处定义上述外部全局变量.
// 2. 按如下顺序使用:
//      InitSocket();                               // 初始化Socket相关量
//      WaitAndResetInitSig();                      // 等待延迟测试(阻塞到延迟测试完毕)
//      <可读取sockLatency变量>
//      SendCommand(COMMAND_XXX, contentPtr);       // 发送数据(几乎无阻塞)
//      RecvCommand(COMMAND_XXX, len, contentPtr);  // 接收数据(阻塞到全部数据接收完毕)
//      <上述指令无顺序限制>
//      EndCommand();                               // 关闭连接(服务端会等待客户端, 客户端几乎无阻塞)
//      CloseSocket();                              // 清理Socket相关量
// 3. 最好不要读取 / 写入上述外部全局变量
// ******************
// 格式说明:
// 每次发送的信息的第一个int表示信息格式:
// (int)        类型
// 1812         float
// 1824         int
// 和发送命令里相同!
// ******************
// 每次发送的信息的第二个size_t表示信息长度;
// ******************
// 主机命令:
// (int)        类型
// 1812         发送float包
// 1824         发送int包
// ******************(请勿使用如下命令)******************
// 4396         发送结束请求包
// 4397         发送结束确认包
// 4398         正式退出
// 详见宏定义
// 定义(确保同步的)外部函数
// ***********************************************************************************************
// ************************************只使用框内的八个函数!!!************************************
// ****************** 初始化
SocketConfig* InitSocket(int type, const char* ip, u_short port);  // 初始化, 返回Socket的管理类指针, 需要提前声明(但不需要初始化)一个SocketConfig*类型变量保存
double InitLatency(SocketConfig* socketconfig);  // 检测双向延迟
// ****************** 发送, 需要自行保证不会在发送时操作该指针指向的内存(对数组而言)
void SendFloat(SocketConfig* socketconfig, float source);
void SendInt(SocketConfig* socketconfig, int source);
void SendFloatPtr(SocketConfig* socketconfig, const float* source, size_t length);
// ****************** 接收, 需要自行保证不会连续接收到两次同类型数据! 否则会发生覆盖!
float RecvFloat(SocketConfig* socketconfig);
int RecvInt(SocketConfig* socketconfig);
int RecvFloatPtr(SocketConfig* socketconfig, float* target);  // 复制到指针指定的内存, 同时返回数组大小, 自行保证空间足够
int RecvFloatPtr(SocketConfig* socketconfig);  // 直接返回数组大小, 自行保证使用时不会接收到数据!
// ****************** 结束
void CloseSocket(SocketConfig* socketconfig);  // 关闭Socket, 释放内存(因此不需要释放SocketConfig*指针)
// ************************************只使用框内的八个函数!!!************************************
// ***********************************************************************************************
// 定义外部操作辅助函数
void SendCommand(SocketConfig* socketconfig, int command, int len, void* contentPtr = nullptr);  // 发送数据(无需阻塞), 废弃!
void RecvCommand(SocketConfig* socketconfig, int* type, int* len, void* contentPtr = nullptr);  // 接收数据(自带阻塞), 废弃!
void EndCommand(SocketConfig* socketconfig);  // 结束通信(双方都需要使用! 直到双方都结束通信才停止阻塞)
void WaitAndResetInitSig(SocketConfig* socketconfig);  // 初始化时阻塞
void WaitAndResetSendSig(SocketConfig* socketconfig);  // 发送时阻塞
void WaitAndResetRecvSig(SocketConfig* socketconfig);  // 接收时阻塞

// 定义通信函数
void initServer(SocketConfig* socketconfig);
void initClient(SocketConfig* socketconfig);

// 定义线程管理函数
void SocketErrorExit(SocketConfig* socketconfig);
HANDLE SocketResetSemaphore(SocketConfig* socketconfig, HANDLE handle, LONG sigInitialCount, LONG sigMaximumCount);
