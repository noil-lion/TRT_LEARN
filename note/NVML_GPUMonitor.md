# NVML_GPUMonitor
NVIDIA Management Library（NVML）是基于C API的用于监视和管理各种状态的编程接口，是nvidia-smi工具的底层，支持库它提供的API可以监控显卡的温度使用率等参数。用于监视和管理NVIDIA GPU设备的各种状态。
## 支持系统
Windows64;
Linux 64;
## NVML API
1. 初始化
   ```
   // First initialize NVML library
    result = nvmlInit();
    enum nvmlReturn_enum::NVML_SUCCESS = 0
   ```
2. 查询
   ```
    //GPU设备数量查询
    unsigned int device_count, i;
    result = nvmlDeviceGetCount(&device_count);

    //GPU设备名称查询（根据index）
    nvmlDevice_t device;
    char name[NVML_DEVICE_NAME_BUFFER_SIZE];
    result = nvmlDeviceGetHandleByIndex(i, &device);
    result = nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE);

    //GPU设备利用率查询（GPU利用率泛指GPU core的利用率）
    nvmlUtilization_t utilization;
    result = nvmlDeviceGetUtilizationRates(device, &utilization);
    //输出设备的利用率
    std::cout << "GPU 使用率： " << utilization.gpu << endl;

    //GPU设备显存信息查询
    nvmlMemory_t memory;
    result = nvmlDeviceGetMemoryInfo(device, &memory);
    //显存利用率
    std::cout << "显存使用率 " << utilization.memory << endl;
    //显存总量计算
    std::cout << "全部可用显存:" << (float)(memory.total)/1024.0f/1024.0f/1024.0f << "GB" << std::endl;
    //显存剩余量计算
    std::cout << "剩余可用显存:" << (float)(memory.free)/1024.0f/1024.0f/1024.0f << "GB" << std::endl;

   ```
3. 错误报告
   ```
    if (NVML_SUCCESS != result)
    {
        std::cout << "Failed to query device count: " << nvmlErrorString(result);
    }

    if (NVML_SUCCESS != result) {
                std::cout << "get device failed " << endl;
    }
    if (NVML_SUCCESS != result)
            {
                std::cout << "Failed to get memory info: " << nvmlErrorString(result);
                return -1;
            }
   ```
