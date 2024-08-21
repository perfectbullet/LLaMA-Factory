import psutil
import pynvml  # 导包

UNIT = 1024 * 1024 * 1024


def get_gpu_and_cpu_info():
    '''
    获取gpu使用情况
    '''
    usage_info = {}
    pynvml.nvmlInit()  # 初始化
    gpuDeriveInfo = pynvml.nvmlSystemGetDriverVersion()
    print("Drive版本: ", gpuDeriveInfo)  # 显示驱动信息
    usage_info['Drive版本'] = '{}'.format(gpuDeriveInfo)
    gpuDeviceCount = pynvml.nvmlDeviceGetCount()  # 获取Nvidia GPU块数
    print("GPU个数：", gpuDeviceCount)
    usage_info['GPU个数'] = '{}'.format(gpuDeviceCount)

    new_usage_info = {}
    usage_info['每个设备使用情况'] = new_usage_info
    for i in range(gpuDeviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)  # 获取GPU i的handle，后续通过handle来处理

        memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)  # 通过handle获取GPU i的信息

        gpuName = '{}'.format(pynvml.nvmlDeviceGetName(handle), encoding='utf-8')

        gpuTemperature = pynvml.nvmlDeviceGetTemperature(handle, 0)

        gpuFanSpeed = pynvml.nvmlDeviceGetFanSpeed(handle)

        gpuPowerState = pynvml.nvmlDeviceGetPowerState(handle)

        gpuUtilRate = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpuMemoryRate = pynvml.nvmlDeviceGetUtilizationRates(handle).memory
        one_gpu_info = {}
        print("第 %d 张卡：" % i, "-" * 30)
        new_usage_info["第 %d 张卡：" % i] = one_gpu_info
        print("显卡名：", gpuName)
        one_gpu_info['显卡名'] = gpuName
        print("内存总容量：", memoryInfo.total / UNIT, "GB")
        one_gpu_info['内存总容量'] = '{} {}'.format(memoryInfo.total / UNIT, "GB")
        print("使用容量：", memoryInfo.used / UNIT, "GB")
        one_gpu_info['使用容量'] = '{} {}'.format(memoryInfo.used / UNIT, "GB")
        print("剩余容量：", memoryInfo.free / UNIT, "GB")
        one_gpu_info['使用容量'] = '{} {}'.format(memoryInfo.free / UNIT, "GB")
        print("显存空闲率：", memoryInfo.free / memoryInfo.total)
        one_gpu_info['显存空闲率'] = memoryInfo.free / memoryInfo.total
        print("温度：", gpuTemperature, "摄氏度")
        one_gpu_info['温度'] = '{}{}'.format(gpuTemperature, "摄氏度")
        print("风扇速率：", gpuFanSpeed)
        one_gpu_info['风扇速率'] = gpuFanSpeed
        print("供电水平：", gpuPowerState)
        one_gpu_info['供电水平'] = gpuPowerState
        print("gpu计算核心满速使用率：", gpuUtilRate)
        one_gpu_info['gpu计算核心满速使用率'] = gpuUtilRate
        print("gpu内存读写满速使用率：", gpuMemoryRate)
        one_gpu_info['gpu内存读写满速使用率'] = gpuMemoryRate
        print("内存占用率：", memoryInfo.used / memoryInfo.total)
        one_gpu_info['内存占用率'] = memoryInfo.used / memoryInfo.total

        """
        # 设置显卡工作模式
        # 设置完显卡驱动模式后，需要重启才能生效
        # 0 为 WDDM模式，1为TCC 模式
        gpuMode = 0     # WDDM
        gpuMode = 1     # TCC
        pynvml.nvmlDeviceSetDriverModel(handle, gpuMode)
        # 很多显卡不支持设置模式，会报错
        # pynvml.nvml.NVMLError_NotSupported: Not Supported
        """

        # 对pid的gpu消耗进行统计
        pidAllInfo = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)  # 获取所有GPU上正在运行的进程信息
        pidUser_infoS = {}
        for pidInfo in pidAllInfo:
            pidUser_info = {}
            pidUser = psutil.Process(pidInfo.pid).username()
            print("进程pid：", pidInfo.pid, "用户名：", pidUser,
                  "显存占有：", pidInfo.usedGpuMemory / UNIT, "GB")  # 统计某pid使用的显存
            pidUser_info['进程pid'] = pidInfo.pid
            pidUser_info['用户名'] = pidUser
            pidUser_info['显存占有'] = '{} {}'.format(pidInfo.usedGpuMemory / UNIT, "GB")
            pidUser_infoS['进程pid{}'.format(pidInfo.pid)] = pidUser_info
        one_gpu_info['pid的gpu消耗'] = pidUser_infoS
    pynvml.nvmlShutdown()  # 最后关闭管理工具

    # ####################### 加入CPU 使用
    usage_info['CPU数量'] = psutil.cpu_count()
    # psutil.cpu_stats()获取CPU的统计信息
    usage_info['CPU的统计信息'] = psutil.cpu_freq()._asdict()
    usage_info['内存使用情况单位为字节'] = psutil.virtual_memory()._asdict()
    print(usage_info)
    return usage_info


if __name__ == '__main__':
    get_gpu_and_cpu_info()

