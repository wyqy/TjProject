#include "socket_sum.h"
#pragma comment(lib,"ws2_32.lib")


// 传入类型
// 构造
SocketConfig::SocketConfig(int type, const char* ip, u_short port)
{
    // Socket数据
    socket_type = type;
    socket_port = port;
    // 安全字符串
    socket_ip = new char[16];  // 15 + \0 + 2
    strcpy_s(socket_ip, 16, ip);
    socket_ip[15] = '\0';
    // 初始化数据
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
// 析构
SocketConfig::~SocketConfig(void)
{
    delete[] socket_ip;
}
// 读取
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


// 主线程 - 管理&发送
DWORD WINAPI socketThreadFunc(LPVOID thParameter)
{
    SocketConfig* para = (SocketConfig*)thParameter;

    int socketSendRet;
    int winsockStatus;
    BOOL winapiStatus;
    DWORD threadStatus;

    bool isWorking = true;
    int mainCommand;

    // 初始化DLL
    WSADATA wsaData;
    WORD DllVersion = MAKEWORD(2, 2);
    winsockStatus = WSAStartup(DllVersion, &wsaData);
    if (winsockStatus) SocketErrorExit(para);
    // 创建信号量
    para->sockCommandSignal = CreateSemaphore(NULL, 0, 1, NULL);
    para->sockPauseRecvSignal = CreateSemaphore(NULL, 0, 1, NULL);

    // 创建套接字
    para->selfSockBackend = socket(
        PF_INET,        // IPv4
        SOCK_STREAM,    // oriented for connection
        IPPROTO_TCP);   // TCP
    // 设置套接字(IPv4)
    para->selfSockAddr.sin_family = PF_INET;                                                  // IPv4
    winsockStatus = inet_pton(PF_INET, para->getIP(), &(para->selfSockAddr.sin_addr.s_addr));  // IP
    if (winsockStatus != 1) SocketErrorExit(para);
    para->selfSockAddr.sin_port = htons(para->getPort());                                    // port
    // 创建子线程
    para->recvThread = CreateThread(
        NULL,                           // default security attributes
        0,                              // use default stack size
        recvThreadFunc,                 // thread function
        para,                           // argument to thread function
        CREATE_SUSPENDED,               // use default creation flags. 0 means the thread will be run at once CREATE_SUSPENDED
        &(para->recvThreadID));    // default thread id

    // 工作过程: 从不阻塞;
    // 设置四个信号量: 发送和接收, 主线程唤醒和外部读取;
    // 注意信号量由创建者负责重置, 但由外部负责释放!
    // 设置一个只读量, 用来接收命令;
    // 设置一个只写量, 用来通知信息格式;
    // 
    // 初始化完毕将恢复接收(recv)信号量
    // 平时处于不断接收(阻塞)的状态, 但能响应发送(双工?)
    // 双线程: 发送在主线程, 接收在子线程, 且两个线程互不干涉(除了初始化和结束通信)
    //      主线程: 平时暂停线程, 若被唤醒则处理命令,
    //              发送完毕恢复发送操作信号量;
    //              结束时等待传输完毕再结束子线程, 然后释放资源并退出自身;
    //      子线程: 平时阻塞自身, 若接收到信息,
    //              先设定信息格式, 然后接收信息,
    //              接收完毕恢复发送操作信号量;
    // 
    // 决定socket类型
    if (para->getType() == 1)  // 服务器
    {
        // 初始化服务器, 并与客户端建立连接
        initServer(para);
        // 准备子线程
        threadStatus = ResumeThread(para->recvThread);
        if (threadStatus == -1) SocketErrorExit(para);
        // 释放信号量
        winapiStatus = ReleaseSemaphore(para->sockSendSignal, (LONG)1, NULL);
        if (!winapiStatus) SocketErrorExit(para);
        winapiStatus = ReleaseSemaphore(
            para->sockInitSignal,   // signal
            (LONG)1,                // incremental
            NULL);                  // last value
        if (!winapiStatus) SocketErrorExit(para);

        // 工作循环
        while (isWorking)
        {
            // 等待命令唤醒
            threadStatus = WaitForSingleObject(para->sockCommandSignal, INFINITE);
            if (threadStatus) SocketErrorExit(para);

            // 唤醒后
            // 重置信号量
            para->sockCommandSignal = SocketResetSemaphore(para, para->sockCommandSignal, 0, 1);
            // 读取命令
            mainCommand = para->sendDataType;
            switch (mainCommand)
            {
            case COMMAND_FLOAT:  // 发送float
                // 发送数据头
                socketSendRet = send(para->clntSockBackend, (const char*)&mainCommand, sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // 发送数据长度
                socketSendRet = send(para->clntSockBackend, (const char*)&(para->sendDataLen), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // 发送float格式buffer给出的数据
                socketSendRet = send(para->clntSockBackend, (const char*)&(para->sendBuffer_float), sizeof(float), NULL);
                if (socketSendRet != sizeof(float)) SocketErrorExit(para);
                // 释放信号量
                winapiStatus = ReleaseSemaphore(para->sockSendSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                break;
            case COMMAND_INT:  // 发送int
                // 发送数据头
                socketSendRet = send(para->clntSockBackend, (const char*)&mainCommand, sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // 发送数据长度
                socketSendRet = send(para->clntSockBackend, (const char*)&(para->sendDataLen), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // 发送int格式buffer给出的数据
                socketSendRet = send(para->clntSockBackend, (const char*)&(para->sendBuffer_int), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // 释放信号量
                winapiStatus = ReleaseSemaphore(para->sockSendSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                break;
            case COMMAND_RETEND:  // 发送结束确认
                // 发送结束确认包
                socketSendRet = send(para->clntSockBackend, (const char*)&mainCommand, sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // 释放信号量
                winapiStatus = ReleaseSemaphore(para->sockSendSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                break;
            case COMMAND_ACKEND:  // 结束执行
                // 结束循环
                isWorking = false;
                break;
            }
        }
    }
    else if (para->getType() == 2)  // 客户端
    {
        // 初始化客户端, 并与服务器建立连接
        initClient(para);
        // 准备子线程
        threadStatus = ResumeThread(para->recvThread);
        if (threadStatus == -1) SocketErrorExit(para);
        // 释放信号量
        winapiStatus = ReleaseSemaphore(para->sockSendSignal, (LONG)1, NULL);
        if (!winapiStatus) SocketErrorExit(para);
        winapiStatus = ReleaseSemaphore(para->sockInitSignal, (LONG)1, NULL);
        if (!winapiStatus) SocketErrorExit(para);

        // 工作循环
        while (isWorking)
        {
            // 等待命令唤醒
            threadStatus = WaitForSingleObject(para->sockCommandSignal, INFINITE);
            if (threadStatus) SocketErrorExit(para);

            // 唤醒后
            // 重置信号量
            para->sockCommandSignal = SocketResetSemaphore(para, para->sockCommandSignal, 0, 1);
            // 读取命令
            mainCommand = para->sendDataType;
            switch (mainCommand)
            {
            case COMMAND_FLOAT:  // 发送float
                // 发送数据头
                socketSendRet = send(para->selfSockBackend, (const char*)&mainCommand, sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // 发送数据长度
                socketSendRet = send(para->selfSockBackend, (const char*)&(para->sendDataLen), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // 发送float格式buffer给出的数据
                socketSendRet = send(para->selfSockBackend, (const char*)&(para->sendBuffer_float), sizeof(float), NULL);
                if (socketSendRet != sizeof(float)) SocketErrorExit(para);
                // 释放信号量
                winapiStatus = ReleaseSemaphore(para->sockSendSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                break;
            case COMMAND_INT:  // 发送int
                // 发送数据头
                socketSendRet = send(para->selfSockBackend, (const char*)&mainCommand, sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // 发送数据长度
                socketSendRet = send(para->selfSockBackend, (const char*)&(para->sendDataLen), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // 发送int格式buffer给出的数据
                socketSendRet = send(para->selfSockBackend, (const char*)&(para->sendBuffer_int), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // 释放信号量
                winapiStatus = ReleaseSemaphore(para->sockSendSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                break;
            case COMMAND_REQEND:  // 发送结束请求
                // 发送结束请求包
                socketSendRet = send(para->selfSockBackend, (const char*)&mainCommand, sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // 释放信号量
                winapiStatus = ReleaseSemaphore(para->sockSendSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                break;
            case COMMAND_ACKEND:  // 结束执行
                // 结束循环
                isWorking = false;
                break;
            }
        }
    }
    else ExitProcess((DWORD)-1);

    // 发送FIN包
    // shutdown(para->selfSockBackend, SD_BOTH);
    // 关闭信号量
    winapiStatus = CloseHandle(para->sockCommandSignal);
    if (!winapiStatus) SocketErrorExit(para);
    winapiStatus = CloseHandle(para->sockPauseRecvSignal);
    if (!winapiStatus) SocketErrorExit(para);
    // 关闭句柄引用
    winapiStatus = CloseHandle(para->recvThread);
    if (!winapiStatus) SocketErrorExit(para);
    // 关闭套接字
    if (para->isSelfSockBackendInited) closesocket(para->selfSockBackend);
    if (para->isClntSockBackendInited) closesocket(para->clntSockBackend);
    para->isSelfSockBackendInited = false;
    para->isClntSockBackendInited = false;

    // 释放DLL
    WSACleanup();
    return (DWORD)0;
}


// 子线程 - 接收
DWORD WINAPI recvThreadFunc(LPVOID thParameter)
{
    SocketConfig* para = (SocketConfig*)thParameter;

    int socketSendRet;
    BOOL winapiStatus;
    DWORD threadStatus;

    bool isWorking = true;
    int recvType;

    // 默认已经初始化完成
    // 不要初始化任何堆内存, 或者对象
    // 决定socket类型
    if (para->getType() == 1)  // 服务器
    {
        // 工作循环
        while (isWorking)
        {
            // 进入接收阻塞
            socketSendRet = recv(para->clntSockBackend, (char*)&recvType, sizeof(int), NULL);
            if (socketSendRet == 0 || socketSendRet == -1) goto RECV_END;
            // else if (socketSendRet != sizeof(int)) SocketErrorExit(para);
            // 接收后
            // 置位通知位
            para->recvDataType = recvType;
            // 读取格式
            switch (recvType)
            {
            case COMMAND_FLOAT:  // 接收float
                // 接收数据长度
                socketSendRet = recv(para->clntSockBackend, (char*)&(para->recvDataLen), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // 接收指定长度的数据到float格式的buffer
                socketSendRet = recv(para->clntSockBackend, (char*)&(para->recvBuffer_float), para->recvDataLen, NULL);
                if (socketSendRet != para->recvDataLen) SocketErrorExit(para);
                // 释放信号量
                winapiStatus = ReleaseSemaphore(para->sockRecvSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                // 等待主进程处理完毕
                threadStatus = WaitForSingleObject(para->sockPauseRecvSignal, INFINITE);
                if (threadStatus) SocketErrorExit(para);
                para->sockPauseRecvSignal = SocketResetSemaphore(para, para->sockPauseRecvSignal, 0, 1);
                break;
            case COMMAND_INT:  // 接收int
                // 接收数据长度
                socketSendRet = recv(para->clntSockBackend, (char*)&(para->recvDataLen), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // 接收指定长度的数据到int格式的buffer
                socketSendRet = recv(para->clntSockBackend, (char*)&(para->recvBuffer_int), para->recvDataLen, NULL);
                if (socketSendRet != para->recvDataLen) SocketErrorExit(para);
                // 释放信号量
                winapiStatus = ReleaseSemaphore(para->sockRecvSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                // 等待主进程处理完毕
                threadStatus = WaitForSingleObject(para->sockPauseRecvSignal, INFINITE);
                if (threadStatus) SocketErrorExit(para);
                para->sockPauseRecvSignal = SocketResetSemaphore(para, para->sockPauseRecvSignal, 0, 1);
                break;
            case COMMAND_REQEND:  // 结束自身执行
                // 释放信号量
                winapiStatus = ReleaseSemaphore(para->sockRecvSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                // 结束循环
                isWorking = false;
                break;
            }
        }
    }
    else if (para->getType() == 2)  // 客户端
    {
        // 工作循环
        while (isWorking)
        {
            // 进入接收阻塞
            socketSendRet = recv(para->selfSockBackend, (char*)&recvType, sizeof(int), NULL);
            if (socketSendRet == 0 || socketSendRet == -1) goto RECV_END;
            // else if (socketSendRet != sizeof(int)) SocketErrorExit(para);
            // 接收后
            // 置位通知位
            para->recvDataType = recvType;
            // 读取格式
            switch (recvType)
            {
            case COMMAND_FLOAT:  // 接收float
                // 接收数据长度
                socketSendRet = recv(para->selfSockBackend, (char*)&(para->recvDataLen), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // 接收指定长度的数据到float格式的buffer
                socketSendRet = recv(para->selfSockBackend, (char*)&(para->recvBuffer_float), para->recvDataLen, NULL);
                if (socketSendRet != para->recvDataLen) SocketErrorExit(para);
                // 释放信号量
                winapiStatus = ReleaseSemaphore(para->sockRecvSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                // 等待主进程处理完毕
                threadStatus = WaitForSingleObject(para->sockPauseRecvSignal, INFINITE);
                if (threadStatus) SocketErrorExit(para);
                para->sockPauseRecvSignal = SocketResetSemaphore(para, para->sockPauseRecvSignal, 0, 1);
                break;
            case COMMAND_INT:  // 接收int
                // 接收数据长度
                socketSendRet = recv(para->selfSockBackend, (char*)&(para->recvDataLen), sizeof(int), NULL);
                if (socketSendRet != sizeof(int)) SocketErrorExit(para);
                // 接收指定长度的数据到int格式的buffer
                socketSendRet = recv(para->selfSockBackend, (char*)&(para->recvBuffer_int), para->recvDataLen, NULL);
                if (socketSendRet != para->recvDataLen) SocketErrorExit(para);
                // 释放信号量
                winapiStatus = ReleaseSemaphore(para->sockRecvSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                // 等待主进程处理完毕
                threadStatus = WaitForSingleObject(para->sockPauseRecvSignal, INFINITE);
                if (threadStatus) SocketErrorExit(para);
                para->sockPauseRecvSignal = SocketResetSemaphore(para, para->sockPauseRecvSignal, 0, 1);
                break;
            case COMMAND_RETEND:  // 结束自身执行
                // 释放信号量
                winapiStatus = ReleaseSemaphore(para->sockRecvSignal, (LONG)1, NULL);
                if (!winapiStatus) SocketErrorExit(para);
                // 结束循环
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
    // 构造
    SocketConfig* retConfig = new SocketConfig(type, ip, port);
    // 准备信号量
    retConfig->sockInitSignal = CreateSemaphore(NULL, 0, 1, NULL);
    retConfig->sockSendSignal = CreateSemaphore(NULL, 0, 1, NULL);
    retConfig->sockRecvSignal = CreateSemaphore(NULL, 0, 1, NULL);
    // 准备主线程
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
    // 确保上次发送成功
    WaitAndResetSendSig(socketconfig);

    // 进行此次发送
    socketconfig->sendDataType = command;
    switch (socketconfig->sendDataType)
    {
    case COMMAND_FLOAT:
        // 长度
        socketconfig->sendDataLen = sizeof(float);
        // 信息
        if (contentPtr != nullptr) socketconfig->sendBuffer_float = *((float*)contentPtr);
        break;
    case COMMAND_INT:
        // 长度
        socketconfig->sendDataLen = sizeof(int);
        // 信息
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
    // 阻塞自身等待数据
    WaitAndResetRecvSig(socketconfig);

    // 返回数据格式, 大小
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

    // 清除上次发送信号量
    WaitAndResetSendSig(socketconfig);

    // 分情况
    if (socketconfig->getType() == 1)  // 服务器
    {
        // 等待客户端发送结束请求包, 以结束子线程
        WaitAndResetRecvSig(socketconfig);
        if (socketconfig->recvDataType == COMMAND_REQEND)
        {
            // 发送结束确认包
            socketconfig->sendDataType = COMMAND_RETEND;
            winapiStatus = ReleaseSemaphore(socketconfig->sockCommandSignal, (LONG)1, NULL);
            if (!winapiStatus) SocketErrorExit(socketconfig);
            // 确保发送成功
            WaitAndResetSendSig(socketconfig);
            // 结束主线程
            socketconfig->sendDataType = COMMAND_ACKEND;
            winapiStatus = ReleaseSemaphore(socketconfig->sockCommandSignal, (LONG)1, NULL);
            if (!winapiStatus) SocketErrorExit(socketconfig);
        }
    }
    else if (socketconfig->getType() == 2)  // 客户端
    {
        // 发送结束请求包
        socketconfig->sendDataType = COMMAND_REQEND;
        winapiStatus = ReleaseSemaphore(socketconfig->sockCommandSignal, (LONG)1, NULL);
        if (!winapiStatus) SocketErrorExit(socketconfig);
        // 确保发送成功
        WaitAndResetSendSig(socketconfig);
        // 等待服务器发送结束确认包, 以结束子线程
        WaitAndResetRecvSig(socketconfig);
        if (socketconfig->recvDataType == COMMAND_RETEND)
        {
            // 结束主线程
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
    // 关闭信号量
    winapiStatus = CloseHandle(socketconfig->sockInitSignal);
    if (!winapiStatus) SocketErrorExit(socketconfig);
    winapiStatus = CloseHandle(socketconfig->sockSendSignal);
    if (!winapiStatus) SocketErrorExit(socketconfig);
    winapiStatus = CloseHandle(socketconfig->sockRecvSignal);
    if (!winapiStatus) SocketErrorExit(socketconfig);
    // 关闭线程
    winapiStatus = CloseHandle(socketconfig->socketThread);
    if (!winapiStatus) SocketErrorExit(socketconfig);
    // 销毁对象
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

    // 绑定套接字
    winsockStatus = bind(socketconfig->selfSockBackend, (SOCKADDR*)&(socketconfig->selfSockAddr), sizeof(SOCKADDR));
    if (winsockStatus == -1) SocketErrorExit(socketconfig);
    socketconfig->isSelfSockBackendInited = true;
    // 进入监听状态
    winsockStatus = listen(socketconfig->selfSockBackend, BACKLOG);
    if (winsockStatus == -1) SocketErrorExit(socketconfig);
    // 接收客户端请求
    int nSize = sizeof(SOCKADDR);
    socketconfig->clntSockBackend = accept(socketconfig->selfSockBackend, (SOCKADDR*)&(socketconfig->clntSockAddr), &nSize);
    socketconfig->isClntSockBackendInited = true;
    // 开始计时
    start_time = system_clock::now();
    // 向客户端发送初始化数据
    int handServer = 123;
    send(socketconfig->clntSockBackend, (const char*)&handServer, sizeof(int), NULL);
    // 接收客户端响应
    recv(socketconfig->clntSockBackend, (char*)&handServer, sizeof(int), NULL);
    // 停止计时
    end_time = system_clock::now();
    // 向客户端再次发送数据
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

    // 向服务器发起请求
    winsockStatus = connect(socketconfig->selfSockBackend, (SOCKADDR*)&(socketconfig->selfSockAddr), sizeof(SOCKADDR));
    if (winsockStatus == -1) SocketErrorExit(socketconfig);
    socketconfig->isSelfSockBackendInited = true;
    // 接收服务器初始化数据
    int handClient = 0;
    recv(socketconfig->selfSockBackend, (char*)&handClient, sizeof(int), NULL);
    // 开始计时
    start_time = system_clock::now();
    // 向服务器发送响应
    send(socketconfig->selfSockBackend, (const char*)&handClient, sizeof(int), NULL);
    // 再次接收服务器数据
    recv(socketconfig->selfSockBackend, (char*)&handClient, sizeof(int), NULL);
    // 停止计时
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
