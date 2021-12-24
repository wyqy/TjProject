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

    float locRetValue = 0.0f;
    float rmtRetValue = 0.0f;
    int socketRetStatus = 0;

    cudaError_t cudaStatus;

    wcout << L"初始化, 请稍候...";
    // 本机准备
    retValue = init_arithmetic(rawFloatData, (float)1, DUAL_DATANUM);  // 初始化本机数据!
    cudaStatus = initialCuda(0, rawFloatData, DUAL_DATANUM, locFloatData, SGLE_DATANUM);  // 初始化CUDA, 对于多GPU系统请修改!
    if (cudaStatus != cudaSuccess) goto InitError;

    system("cls");

    while (true)
    {
        wcout << L"\n单精度数组求和基准时间测试(5 次): \n\
                   1. 原始代码实现(误差最大); \n\
                   2. CUDA + AVX加速实现(误差最小); \n\
                   3. OMP + AVX加速实现(误差适中); \n\
                   4. CUDA + AVX + Socket加速实现; \n\
                   5. OMP + AVX + Socket加速实现; \n\
                   6. 退出. \n\
                   请输入选项: ";
        wcin >> main_entered;
        if (main_entered == 6) break;

        switch (main_entered)
        {
        case 1:
            for (size_t iter = 1; iter <= 5; iter++)
            {
                // 执行
                start_time = system_clock::now();
                retValue = sumNaive(rawFloatData, (size_t)DUAL_DATANUM);  // 原始代码实现
                end_time = system_clock::now();
                diff = end_time - start_time;
                durationSecond = diff.count();
                // 显示
                wcout << L"第" << iter << L"次: 耗时(秒): " << durationSecond;
                wcout << L"; 总和为: " << fixed << retValue << endl;
            }
            break;
        case 2:  // 2, 3 代码雷同, 因此简化
            // [[fallthrough]];
        case 3:
            for (size_t iter = 1; iter <= 5; iter++)
            {
                // 执行
                start_time = system_clock::now();
                if (main_entered == 2) retValue = sumSpeedup_Cu(rawFloatData, (size_t)DUAL_DATANUM);  // CUDA加速
                else if (main_entered == 3) retValue = sumSpeedup_OP(rawFloatData, (size_t)DUAL_DATANUM);  // OMP加速
                end_time = system_clock::now();
                diff = end_time - start_time;
                durationSecond = diff.count();
                // 显示
                wcout << L"第" << iter << L"次: 耗时(秒): " << durationSecond;
                wcout << L"; 总和为: " << fixed << retValue << endl;
            }
            break;
        case 4:  // 4, 5 代码雷同, 因此简化
            // [[fallthrough]];
        case 5:
            // 多机准备
            int socket_type;                        // 通信类型: 1 = 服务器; 2 = 客户端
            char* socket_ip;                        // Socket IP
            u_short socket_port;                    // Socket Port
            double sockLatency;
            SocketConfig* socketconfig;

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
                SendInt(socketconfig, 1000);
                // 己方初始化
                retValue = init_arithmetic(locFloatData, (float)1, SGLE_DATANUM);  // 初始化多机数据!
                // 等待对方结果
                socketRetStatus = RecvInt(socketconfig);
                if (socketRetStatus != 1000) break;

                for (size_t iter = 1; iter <= 5; iter++)
                {
                    // 执行
                    start_time = system_clock::now();
                    // 发送执行命令至客户端
                    SendInt(socketconfig, 1001);
                    // 执行己方计算(分类讨论)
                    if (main_entered == 4) locRetValue = sumSpeedup_Cu(locFloatData, (size_t)SGLE_DATANUM);  // CUDA加速
                    else if (main_entered == 5) locRetValue = sumSpeedup_OP(locFloatData, (size_t)SGLE_DATANUM);  // OMP加速
                    // 发送己方数据
                    SendFloat(socketconfig, locRetValue);
                    // 等待对方结果
                    rmtRetValue = RecvFloat(socketconfig);
                    // 合并二者数据
                    retValue = locRetValue + rmtRetValue;
                    // 计算完毕
                    end_time = system_clock::now();
                    diff = end_time - start_time;
                    durationSecond = diff.count();
                    // 显示
                    wcout << L"第" << iter << L"次: 耗时(秒): " << durationSecond;
                    wcout << L"; 总和为: " << fixed << retValue << endl;
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
                retValue = init_arithmetic(locFloatData, (float)SGLE_DATANUM, SGLE_DATANUM);  // 初始化多机数据!
                // 发送己方结果
                SendInt(socketconfig, 1000);

                for (size_t iter = 1; iter <= 5; iter++)
                {
                    // 执行
                    start_time = system_clock::now();
                    // 接收来自服务器的执行命令
                    socketRetStatus = RecvInt(socketconfig);
                    // 执行己方计算(分类讨论)
                    if (socketRetStatus == 1001)
                    {
                        if (main_entered == 4) locRetValue = sumSpeedup_Cu(locFloatData, (size_t)SGLE_DATANUM);  // CUDA加速
                        else if (main_entered == 5) locRetValue = sumSpeedup_OP(locFloatData, (size_t)SGLE_DATANUM);  // OMP加速
                    }
                    // 发送己方数据
                    SendFloat(socketconfig, locRetValue);
                    // 等待对方结果
                    rmtRetValue = RecvFloat(socketconfig);
                    // 合并二者数据
                    retValue = locRetValue + rmtRetValue;
                    // 计算完毕
                    end_time = system_clock::now();
                    diff = end_time - start_time;
                    durationSecond = diff.count();
                    // 显示
                    wcout << L"第" << iter << L"次: 耗时(秒): " << durationSecond;
                    wcout << L"; 总和为: " << fixed << retValue << endl;
                    // 发送同步指令
                    SendInt(socketconfig, 1002);
                }
            }

            // 释放堆内存
            delete[] socket_ip;
            // 关闭Socket
            CloseSocket(socketconfig);
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
