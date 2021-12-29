#include "speed_sort.h"

__declspec(align(MEMORY_ALIGNED)) float disorderData[DUAL_DATANUM];
__declspec(align(MEMORY_ALIGNED)) float rawFloatData[DUAL_DATANUM];
__declspec(align(MEMORY_ALIGNED)) float locFloatData[SGLE_DATANUM];
__declspec(align(MEMORY_ALIGNED)) float rmtFloatData[SGLE_DATANUM];

int main()
{
    // 界面输入输出变量
    int main_entered;
    bool isValid = 0;
    float retValue = 0;
    double durationSecond = 0;

    // 计时变量
    system_clock::time_point start_time;
    system_clock::time_point end_time;
    duration<double> diff;
    wcout.imbue(locale("chs"));

    // 多机配置
    int socket_type;                        // 通信类型: 1 = 服务器; 2 = 客户端
    char* socket_ip;                        // Socket IP
    u_short socket_port;                    // Socket Port
    double sockLatency;
    SocketConfig* socketconfig;

    // 多机变量
    float rmtDurationSecond = 0.0;
    int socketRetStatus = 0;

    // cuda变量
    cudaError_t cudaStatus;


    // 本机准备
    wcout << L"初始化, 请稍候...";
    cudaStatus = initialCuda(0, rawFloatData, DUAL_DATANUM, locFloatData, SGLE_DATANUM);  // 初始化CUDA, 对于多GPU系统请修改!
    if (cudaStatus != cudaSuccess) goto InitError;
    system("cls");

    while (true)
    {
        wcout << L"单精度浮点数数组排序测试: \n\
                    1. 原始做法; \n\
                    2. 本机加速; \n\
                    3. 多机加速; \n\
                    4. 多机传输测试; \n\
                    5. 退出. \n\
                    请输入选项: ";
        wcin >> main_entered;
        if (main_entered == 5) break;

        switch (main_entered)
        {
        case 1:
            // 随机排列, 随后只使用此次的随机排列数据 ,下同
            wcout << L"随机排序, 请稍候...";
            retValue = init_arithmetic(disorderData, (float)1, DUAL_DATANUM);
            retValue = init_random(disorderData, DUAL_DATANUM);  // 随机初始化数据!
            wcout << L"完成!" << endl;
            for (size_t iter = 1; iter <= 5; iter++)
            {
                // 还原顺序
                wcout << L"复制排序, 请稍候...";
                memcpy(rawFloatData, disorderData, DUAL_DATANUM * sizeof(float));
                // 执行
                wcout << L"完成! 排序中... ";
                start_time = system_clock::now();
                retValue = sortNaive(rawFloatData, (size_t)DUAL_DATANUM);  // 原始代码实现
                end_time = system_clock::now();
                diff = end_time - start_time;
                durationSecond = diff.count();
                wcout << L"第" << iter << L"次: 耗时(秒): " << durationSecond << L"; 验证中...";
                // 验证
                retValue = postProcess(rawFloatData, DUAL_DATANUM);
                isValid = validSort(rawFloatData, (size_t)DUAL_DATANUM);  // 验证
                if (isValid) wcout << L"完成! 排序结果正确! " << fixed << retValue << endl;
                else wcout << L"完成! 排序结果错误! " << fixed << retValue << endl;
            }
            break;
        case 2:
            // 随机排列
            wcout << L"随机排序, 请稍候...";
            retValue = init_arithmetic(disorderData, (float)1, DUAL_DATANUM);
            retValue = init_random(disorderData, DUAL_DATANUM);  // 随机初始化数据!
            wcout << L"完成!" << endl;
            for (size_t iter = 1; iter <= 5; iter++)
            {
                // 还原顺序
                wcout << L"复制排序, 请稍候...";
                memcpy(rawFloatData, disorderData, DUAL_DATANUM * sizeof(float));
                // 执行
                wcout << L"完成! 排序中... ";
                start_time = system_clock::now();
                retValue = sortSpeedup(rawFloatData, (size_t)DUAL_DATANUM);  // 加速代码实现
                end_time = system_clock::now();
                diff = end_time - start_time;
                durationSecond = diff.count();
                wcout << L"第" << iter << L"次: 耗时(秒): " << durationSecond << L"; 验证中...";
                // 验证
                retValue = postSpeedup(rawFloatData, DUAL_DATANUM);
                isValid = validSort(rawFloatData, (size_t)DUAL_DATANUM);  // 验证
                if (isValid) wcout << L"完成! 排序结果正确! " << fixed << retValue << endl;
                else wcout << L"完成! 排序结果错误! " << fixed << retValue << endl;
            }
            break;
        case 3:
            wcout << L"\n请输入通信类型: \n\
                       1. 主控端(Server); \n\
                       2. 受控端(Client). \n\
                       请输入选项: ";
            wcin >> socket_type;
            wcout << L"\n请输入IP: ";
            socket_ip = new char[16];
            cin.get();
            cin.get(socket_ip, 16);
            wcout << L"\n请输入端口号: ";
            wcin >> socket_port;

            // 随机排列
            wcout << L"随机排序, 请稍候...";
            if (socket_type == 1) retValue = init_arithmetic(disorderData, (float)1, SGLE_DATANUM);
            else if(socket_type == 2) retValue = init_arithmetic(disorderData, (float)SGLE_DATANUM, SGLE_DATANUM);
            retValue = init_random(disorderData, SGLE_DATANUM);  // 随机初始化数据!
            wcout << L"完成!" << endl;

            wcout << L"初始化socket, 请稍候...";
            // 初始化Socket相关量
            socketconfig = InitSocket(socket_type, socket_ip, socket_port);
            // 等待延迟测试
            sockLatency = InitLatency(socketconfig);
            // 显示延迟
            wcout << L"成功! 双向连接延迟(毫秒): " << sockLatency * 1000.0f << endl;

            wcout << L"初始化分布式数组, 请稍候..." << endl;

            if (socket_type == 1)
            {
                // 多机准备
                // 发送初始化命令至客户端
                SendInt(socketconfig, 1000);
                // 己方初始化
                memcpy(locFloatData, disorderData, SGLE_DATANUM * sizeof(float));
                // 等待对方结果
                socketRetStatus = RecvInt(socketconfig);
                if (socketRetStatus != 1000) break;

                for (size_t iter = 1; iter <= 5; iter++)
                {
                    // 执行
                    start_time = system_clock::now();
                    // 发送执行命令至客户端
                    SendInt(socketconfig, 1001);
                    // 执行己方计算
                    retValue = sortSpeedup(locFloatData, (size_t)SGLE_DATANUM);  // 加速代码实现
                    // 等待对方结果
                    socketRetStatus = RecvFloatPtr(socketconfig, rmtFloatData);
                    // 合并
                    retValue = sortMerge(rawFloatData, locFloatData, SGLE_DATANUM, rmtFloatData, SGLE_DATANUM);
                    // 计算完毕
                    end_time = system_clock::now();
                    diff = end_time - start_time;
                    durationSecond = diff.count();
                    // 显示速度
                    wcout << L"第" << iter << L"次: 耗时(秒): " << durationSecond << L"; 验证中...";
                    // 验证结果
                    retValue = postSpeedup(rawFloatData, DUAL_DATANUM);
                    isValid = validSort(rawFloatData, (size_t)DUAL_DATANUM);  // 验证
                    if (isValid) wcout << L"完成! 排序结果正确! " << fixed << retValue << endl;
                    else wcout << L"完成! 排序结果错误! " << fixed << retValue << endl;
                    // 发送最终结果
                    SendFloat(socketconfig, (float)durationSecond);
                    if (isValid) SendInt(socketconfig, 1501);
                    else SendInt(socketconfig, 1502);
                    // 接收同步指令
                    socketRetStatus = RecvInt(socketconfig);
                    if (socketRetStatus != 1002) break;
                }
            }
            else if (socket_type == 2)
            {
                // 多机准备
                // 接收来自服务器的初始化命令
                socketRetStatus = RecvInt(socketconfig);
                if (socketRetStatus != 1000) break;
                // 己方初始化
                memcpy(locFloatData, disorderData, SGLE_DATANUM * sizeof(float));
                // 发送己方结果
                SendInt(socketconfig, 1000);

                for (size_t iter = 1; iter <= 5; iter++)
                {
                    // 执行
                    start_time = system_clock::now();
                    // 接收来自服务器的执行命令
                    socketRetStatus = RecvInt(socketconfig);
                    if (socketRetStatus != 1001) break;
                    // 执行己方计算
                    retValue = sortSpeedup(locFloatData, (size_t)SGLE_DATANUM);  // 加速代码实现
                    // 发送己方数据
                    SendFloatPtr(socketconfig, locFloatData, SGLE_DATANUM);
                    // 计算完毕
                    end_time = system_clock::now();
                    diff = end_time - start_time;
                    durationSecond = diff.count();
                    // 显示本机速度
                    wcout << L"第" << iter << L"次: 本机耗时(秒): " << durationSecond;
                    // 等待对方结果
                    rmtDurationSecond = RecvFloat(socketconfig);
                    socketRetStatus = RecvInt(socketconfig);
                    // 显示最终结果
                    wcout << L"; 最终总耗时: " << fixed << rmtDurationSecond;
                    if (socketRetStatus == 1501) wcout << L", 排序结果正确! " << endl;
                    else if (socketRetStatus == 1502) wcout << L", 排序结果错误! " << endl;
                    // 发送同步指令
                    SendInt(socketconfig, 1002);
                }
            }

            delete[] socket_ip;  // 释放堆内存
            CloseSocket(socketconfig);  // 关闭Socket
            break;
        case 4:
            wcout << L"\n请输入通信类型: \n\
                       1. 主控端(Server); \n\
                       2. 受控端(Client). \n\
                       请输入选项: ";
            wcin >> socket_type;
            wcout << L"\n请输入IP: ";
            socket_ip = new char[16];
            cin.get();
            cin.get(socket_ip, 16);
            wcout << L"\n请输入端口号: ";
            wcin >> socket_port;

            wcout << L"初始化socket, 请稍候...";
            // 初始化Socket相关量
            socketconfig = InitSocket(socket_type, socket_ip, socket_port);
            // 等待延迟测试
            sockLatency = InitLatency(socketconfig);
            // 显示延迟
            wcout << L"成功! 双向连接延迟(毫秒): " << sockLatency * 1000.0f << endl;

            wcout << L"初始化分布式数组, 请稍候..." << endl;

            if (socket_type == 1)
            {
                // 多机准备
                // 发送初始化命令至客户端
                SendInt(socketconfig, 2000);
                printf("\nSuccess Send 2000");
                // 己方初始化
                retValue = init_arithmetic(locFloatData, (float)1, SGLE_DATANUM);  // 初始化多机数据!
                printf("\nSuccess Inited");
                // 等待对方结果
                socketRetStatus = RecvInt(socketconfig);
                if (socketRetStatus != 2000) break;
                printf("\nSuccess Recv 2000");

                for (size_t iter = 1; iter <= 2; iter++)
                {
                    // 执行
                    start_time = system_clock::now();
                    // 发送执行命令至客户端
                    SendInt(socketconfig, 2001);
                    printf("\nSuccess Send 2001");
                    // 发送己方数据
                    SendFloatPtr(socketconfig, locFloatData, SGLE_DATANUM);
                    printf("\nSuccess Send!");
                    // 等待对方结果
                    socketRetStatus = RecvFloatPtr(socketconfig, rmtFloatData);
                    printf("\nSuccess Recv!\n");
                    // 验证最后一个数
                    retValue = rmtFloatData[SGLE_DATANUM - 1];
                    // 计算完毕
                    end_time = system_clock::now();
                    diff = end_time - start_time;
                    durationSecond = diff.count();
                    // 显示
                    wcout << L"第" << iter << L"次: 耗时(秒): " << durationSecond;
                    wcout << L"; 结果为(应当为128M): " << fixed << retValue << endl;
                    // 接收同步指令
                    socketRetStatus = RecvInt(socketconfig);
                    if (socketRetStatus != 2002) break;
                    printf("\nSuccess Recv 2002");
                }
            }
            else if (socket_type == 2)
            {
                // 多机准备
                // 接收来自服务器的初始化命令
                socketRetStatus = RecvInt(socketconfig);
                if (socketRetStatus != 2000) break;
                printf("\nSuccess Recv 2000");
                // 己方初始化
                retValue = init_arithmetic(locFloatData, (float)SGLE_DATANUM, SGLE_DATANUM);  // 初始化多机数据!
                // 发送己方结果
                SendInt(socketconfig, 2000);
                printf("\nSuccess Send 2000");

                for (size_t iter = 1; iter <= 2; iter++)
                {
                    // 执行
                    start_time = system_clock::now();
                    // 接收来自服务器的执行命令
                    socketRetStatus = RecvInt(socketconfig);
                    if (socketRetStatus != 2001) break;
                    printf("\nSuccess Recv 2001");
                    // 发送己方数据
                    SendFloatPtr(socketconfig, locFloatData, SGLE_DATANUM);
                    printf("\nSuccess Send!");
                    // 等待对方结果
                    socketRetStatus = RecvFloatPtr(socketconfig, rmtFloatData);
                    printf("\nSuccess Recv!\n");
                    // 验证最后一个数
                    retValue = rmtFloatData[SGLE_DATANUM - 1];
                    // 计算完毕
                    end_time = system_clock::now();
                    diff = end_time - start_time;
                    durationSecond = diff.count();
                    // 显示
                    wcout << L"第" << iter << L"次: 耗时(秒): " << durationSecond;
                    wcout << L"; 结果为(应当为64M): " << fixed << retValue << endl;
                    // 发送同步指令
                    SendInt(socketconfig, 2002);
                    printf("\nSuccess Send 2002");
                }
            }

            delete[] socket_ip;  // 释放堆内存
            CloseSocket(socketconfig);  // 关闭Socket
            printf("\n");
            break;
        default:
            durationSecond = 0;
            retValue = 0;
        }
    }

InitError:
    cudaStatus = releaseCuda();  // 释放CUDA
    system("pause");
    return 0;
}