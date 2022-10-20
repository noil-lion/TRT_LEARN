# Linux下的TensorRT环境构建

## 环境及依赖
|软件|版本|说明|
|:--:|:--:|:--:|
|Ubuntu|20.04|OS|
|Nvidia GPU|A100-PCI|显卡|
|CUDA|11.6||
|CUDNN|8.2||
|CMAKE|3.22.1||
|TensorRT|8.2.4.2|.tar.gz[下载](https://developer.nvidia.com/nvidia-tensorrt-download)|
|TensorRT OSS|:|开源组件-[可选](https://github.com/nvidia/TensorRT)|
|pkg-config|:|[工具](https://blog.csdn.net/wxh0000mm/article/details/122322486)|
|z-lib|:|[工具](http://www.zlib.net/)|

## 安装编译
1. 下载OSS
    ```
    git clone -b master https://github.com/NVIDIA/TensorRT.git
    ```
1. 解压
    ```
    # 解压文件
    tar -xvzf TensorRT-8.2.4.2.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz
    ```

1. 设置环境变量
    ```
    export TRT_SOURCE=`pwd`
    export TRT_RELEASE=`pwd`/TensorRT-8.2.4.2
    export TENSORRT_LIBRARY_INFER=$TRT_RELEASE/targets/x86_64-linux-gnu/lib/libnvinfer.so.7
    export TENSORRT_LIBRARY_INFER_PLUGIN=$TRT_RELEASE/targets/x86_64-linux-gnu/lib/libnvinfer_plugin.so.7
    export TENSORRT_LIBRARY_MYELIN=$TRT_RELEASE/targets/x86_64-linux-gnu/lib/libmyelin.so
    ```

1. 开始编译
    ```
    $ cd TensorRT
    $ mkdir -p build && cd build
    $ cmake .. -DTRT_LIB_DIR=$TRT_RELEASE/lib 
    $ -DTRT_OUT_DIR=`pwd`/out
    $ make -j$(nproc)
    ```

1. 手动下载文件并修改cmake
* 有个文件很难下下来，先下载下载在放在指定位置:[protobuf下载](https://link.zhihu.com/?target=https%3A//github.com/google/protobuf/releases/download/v3.0.0/protobuf-cpp-3.0.0.tar.gz)
* 放置在TensorRT/build/third_party.protobuf/src/路径下
* 修改TensorRT/build/third_party.protobuf/src/third_party.protobuf-stamp/download-third_party.protobuf.cmake文件-{if(EXISTS)一直到该文件最后一行全部删掉}
    ```
    $ make -j$(nproc)
    ```
1. CmakeLists集成
    ```
    # 最低版本
    cmake_minimum_required(VERSION 2.9)
    # DEMO为项目名称
    project(demo)
    # C++版本
    add_definitions(-std=c++11)
    # 设置为TensorRT的根目录
    set(TRT_DIR "/work/wuzihao/TensorRT/TensorRT/TensorRT-8.2.4.2")
    set(TRT_SAMPLE "/work/wuzihao/TensorRT/demo")
    # 其他配置
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_BUILD_TYPE Debug)
    # 项目头文件路径
    include_directories(${PROJECT_SOURCE_DIR}/include)
    include_directories(${PROJECT_SOURCE_DIR}/samples/common)
    # CUDA
    # 关闭使用静态Runtime
    option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    # 在系统寻找CUDA包路径
    find_package(CUDA REQUIRED)
    # 标志文件路径
    include_directories(${CUDA_INCLUDE_DIRS})
    # arch、code根据实际显卡算力填写
    set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11;-g;-G;-gencode;arch=compute_75;code=sm_75)
    # 允许CUDA语法
    enable_language(CUDA)

    # TensorRT
    # 标志文件

    include_directories(${TRT_SAMPLE}/include)
    include_directories(${TRT_SAMPLE}/include/samples)

    # 动态链接库
    link_directories(${TRT_SAMPLE}/lib)  # TensorRT动态库
    link_directories(/usr/local/cuda/lib64)   # cuda动态库

    # 根据哪个文件生成可执行文件
    add_executable(main ${PROJECT_SOURCE_DIR}/src/main.cpp ${PROJECT_SOURCE_DIR}/src/logger.cpp)   # 源文件

    # 添加链接库
    target_link_libraries(main ${CUDA_LIBRARIES})
    target_link_libraries(main cudart)
    target_link_libraries(main nvinfer)
    target_link_libraries(main nvonnxparser)
    target_link_libraries(main cudnn)
    ```

1. DEMO
    ```
    $ mkdir build && cd build
    $ cmake ../
    $ make
    $ ./demo

    &&&& RUNNING TensorRT.sample_onnx_mnist [TensorRT v8200] # ./main
    [10/20/2022-14:49:11] [I] Building and running a GPU inference engine for Onnx MNIST
    [10/20/2022-14:49:13] [I] [TRT] [MemUsageChange] Init CUDA: CPU +329, GPU +0, now: CPU 336, GPU 33179 (MiB)
    [10/20/2022-14:49:16] [I] [TRT] [MemUsageChange] Init builder kernel library: CPU +414, GPU +543, now: CPU 769, GPU 33722 (MiB)
    [10/20/2022-14:49:16] [I] [TRT] ----------------------------------------------------------------
    [10/20/2022-14:49:16] [I] [TRT] Input filename:   ../data/mnist/mnist.onnx
    [10/20/2022-14:49:16] [I] [TRT] ONNX IR version:  0.0.3
    [10/20/2022-14:49:16] [I] [TRT] Opset version:    8
    [10/20/2022-14:49:16] [I] [TRT] Producer name:    CNTK
    [10/20/2022-14:49:16] [I] [TRT] Producer version: 2.5.1
    [10/20/2022-14:49:16] [I] [TRT] Domain:           ai.cntk
    [10/20/2022-14:49:16] [I] [TRT] Model version:    1
    [10/20/2022-14:49:16] [I] [TRT] Doc string:       
    [10/20/2022-14:49:16] [I] [TRT] ----------------------------------------------------------------
    [10/20/2022-14:49:16] [W] [TRT] parsers/onnx/onnx2trt_utils.cpp:364: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
    [10/20/2022-14:49:19] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +791, GPU +413, now: CPU 1560, GPU 33724 (MiB)
    [10/20/2022-14:49:19] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +128, GPU -13, now: CPU 1688, GPU 33711 (MiB)
    [10/20/2022-14:49:19] [I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
    [10/20/2022-14:49:24] [I] [TRT] Detected 1 inputs and 1 output network tensors.
    [10/20/2022-14:49:24] [I] [TRT] Total Host Persistent Memory: 6432
    [10/20/2022-14:49:24] [I] [TRT] Total Device Persistent Memory: 0
    [10/20/2022-14:49:24] [I] [TRT] Total Scratch Memory: 0
    [10/20/2022-14:49:24] [I] [TRT] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 0 MiB, GPU 0 MiB
    [10/20/2022-14:49:24] [I] [TRT] [BlockAssignment] Algorithm ShiftNTopDown took 0.038214ms to assign 3 blocks to 6 nodes requiring 31748 bytes.
    [10/20/2022-14:49:24] [I] [TRT] Total Activation Memory: 31748
    [10/20/2022-14:49:24] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
    [10/20/2022-14:49:24] [W] [TRT] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
    [10/20/2022-14:49:24] [W] [TRT] The getMaxBatchSize() function should not be used with an engine built from a network created with NetworkDefinitionCreationFlag::kEXPLICIT_BATCH flag. This function will always return 1.
    [10/20/2022-14:49:24] [I] [TRT] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 2109, GPU 33881 (MiB)
    [10/20/2022-14:49:24] [I] [TRT] Loaded engine size: 0 MiB
    [10/20/2022-14:49:24] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
    [10/20/2022-14:49:24] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 0 (MiB)
    [10/20/2022-14:49:24] [I] Input:
    [10/20/2022-14:49:24] [I] @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@.*@@@@@@@@@@
    @@@@@@@@@@@@@@@@.=@@@@@@@@@@
    @@@@@@@@@@@@+@@@.=@@@@@@@@@@
    @@@@@@@@@@@% #@@.=@@@@@@@@@@
    @@@@@@@@@@@% #@@.=@@@@@@@@@@
    @@@@@@@@@@@+ *@@:-@@@@@@@@@@
    @@@@@@@@@@@= *@@= @@@@@@@@@@
    @@@@@@@@@@@. #@@= @@@@@@@@@@
    @@@@@@@@@@=  =++.-@@@@@@@@@@
    @@@@@@@@@@       =@@@@@@@@@@
    @@@@@@@@@@  :*## =@@@@@@@@@@
    @@@@@@@@@@:*@@@% =@@@@@@@@@@
    @@@@@@@@@@@@@@@% =@@@@@@@@@@
    @@@@@@@@@@@@@@@# =@@@@@@@@@@
    @@@@@@@@@@@@@@@# =@@@@@@@@@@
    @@@@@@@@@@@@@@@* *@@@@@@@@@@
    @@@@@@@@@@@@@@@= #@@@@@@@@@@
    @@@@@@@@@@@@@@@= #@@@@@@@@@@
    @@@@@@@@@@@@@@@=.@@@@@@@@@@@
    @@@@@@@@@@@@@@@++@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@

    [10/20/2022-14:49:24] [I] Output:
    [10/20/2022-14:49:24] [I]  Prob 0  0.0000 Class 0: 
    [10/20/2022-14:49:24] [I]  Prob 1  0.0000 Class 1: 
    [10/20/2022-14:49:24] [I]  Prob 2  0.0000 Class 2: 
    [10/20/2022-14:49:24] [I]  Prob 3  0.0000 Class 3: 
    [10/20/2022-14:49:24] [I]  Prob 4  0.9911 Class 4: **********
    [10/20/2022-14:49:24] [I]  Prob 5  0.0001 Class 5: 
    [10/20/2022-14:49:24] [I]  Prob 6  0.0000 Class 6: 
    [10/20/2022-14:49:24] [I]  Prob 7  0.0000 Class 7: 
    [10/20/2022-14:49:24] [I]  Prob 8  0.0000 Class 8: 
    [10/20/2022-14:49:24] [I]  Prob 9  0.0088 Class 9: 
    [10/20/2022-14:49:24] [I] 
    ```

## 参考链接
[Link1](https://zhuanlan.zhihu.com/p/346307138)
[Link2](https://zhuanlan.zhihu.com/p/181274475)