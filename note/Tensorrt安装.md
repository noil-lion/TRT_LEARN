# TensorRT-Ubuntu20.04 安装流程
[安装参考](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
## TensorRT概述
NVIDIA ® TensorRT™ 的核心是一个 C++ 库，有助于在 NVIDIA 图形处理单元 (GPU) 上进行高性能推理。TensorRT基于一个已训练的网络（网络结构定义+训练后权重参数）构建出一个高度优化的推理引擎，执行推理任务。  
TensorRT核心组成包括模型解析器和运行时模块，分别对应其模型转换和模型推理部署的功能，提供C++和python API，所以TensorRT可通过网络定义表示深度学习模型，也可通过解析器进行加载已训练模型。
## 安装方式
Debian安装，从下载的本地安装包进行TensorRT安装，与容器安装不同。
## 环境及版本-安装要求
1. 查看Ubuntu版本。
```
cat /proc/version 
输出：
Linux version 5.15.0-43-generic (buildd@lcy02-amd64-026) (gcc (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0, GNU ld (GNU Binutils for Ubuntu) 2.34) #46~20.04.1-Ubuntu SMP Thu Jul 14 15:20:17 UTC 2022
# 能看到linux内核版本号、gcc版本、ubuntu版本及安装时间。

uname -a 

Linux 8d3d08540d36 5.15.0-43-generic #46~20.04.1-Ubuntu SMP Thu Jul 14 15:20:17 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
# linux内核版本号
```
2. CUDA 版本查看
```
nvcc -V
输出：
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Aug_15_21:14:11_PDT_2021
Cuda compilation tools, release 11.4, V11.4.120
Build cuda_11.4.r11.4/compiler.30300941_0

# 上面指令看不到的可以用
cat /usr/local/cuda/version.txt
# 还看不到的进目录/usr/local/cuda，vim ~/.bashrc 添加nvcc环境变量
```
3. cudnn版本查看
```
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```
## 主要流程
1. 下载ubuntu版本和对应CPU架构相匹配的TensorRt本地repo文件  
链接：https://developer.nvidia.com/nvidia-tensorrt-8x-download  
* 选择对应版本及文件类型，可选Debian 或 RPM 软件包。   
本例选择 nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.1.5-ga-20220604_1-1_amd64.deb   

2. 下载好本地包后，将文件上传至服务器，并cd到文件目录下。执行以下命令
```
os="ubuntuxx04"        # 改为当前ubuntu版本
tag="cudax.x-trt8.x.x.x-ga-yyyymmdd" # 改为当前cuda版本
sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb      # 执行安装软件包指令
sudo apt-key add /var/nv-tensorrt-repo-${os}-${tag}/*.pub     # - 把下载的key添加到本地trusted数据库中

sudo apt-get update                                           # 下载源更新
sudo apt-get install tensorrt                                 # 执行安装

# 输出：
The following additional packages will be installed:
  libcudnn8 libcudnn8-dev libnvinfer-bin libnvinfer-dev libnvinfer-plugin-dev libnvinfer-plugin8 libnvinfer-samples libnvinfer8 libnvonnxparsers-dev libnvonnxparsers8 libnvparsers-dev libnvparsers8
The following NEW packages will be installed:
  libcudnn8 libcudnn8-dev libnvinfer-bin libnvinfer-dev libnvinfer-plugin-dev libnvinfer-plugin8 libnvinfer-samples libnvinfer8 libnvonnxparsers-dev libnvonnxparsers8 libnvparsers-dev libnvparsers8 tensorrt
0 upgraded, 13 newly installed, 0 to remove and 138 not upgraded.
Need to get 1527 MB of archives.
After this operation, 3855 MB of additional disk space will be used.
Do you want to continue? [Y/n] Y

```

3. 如果要使用python 3.x推理
```
python3 -m pip install numpy        # python 安装numpy包
sudo apt-get install python3-libnvinfer-dev  # apt安装python推理工具包
```
4. 运行ONNX实例示例图，安装onnx-graphsurgeon工具包
```
python3 -m pip install numpy onnx
sudo apt-get install onnx-graphsurgeon
```
5. 环境变量配置
```
trtexec: command not found         # 添加trtexe的路径到环境变量文件中

vim ~/.bashrc                            # i 
export PATH=/usr/src/tensorrt/bin:$PATH   # esc :wq! 
source ~/.bashrc
```
## 验证安装
```
dpkg -l | grep TensorRT

输出：
ii  libnvinfer-bin                                              8.4.3-1+cuda11.6                  amd64        TensorRT binaries
ii  libnvinfer-dev                                              8.4.3-1+cuda11.6                  amd64        TensorRT development libraries and headers
ii  libnvinfer-plugin-dev                                       8.4.3-1+cuda11.6                  amd64        TensorRT plugin libraries
ii  libnvinfer-plugin8                                          8.4.3-1+cuda11.6                  amd64        TensorRT plugin libraries
ii  libnvinfer-samples                                          8.4.3-1+cuda11.6                  all          TensorRT samples
ii  libnvinfer8                                                 8.4.3-1+cuda11.6                  amd64        TensorRT runtime libraries
ii  libnvonnxparsers-dev                                        8.4.3-1+cuda11.6                  amd64        TensorRT ONNX libraries
ii  libnvonnxparsers8                                           8.4.3-1+cuda11.6                  amd64        TensorRT ONNX libraries
ii  libnvparsers-dev                                            8.4.3-1+cuda11.6                  amd64        TensorRT parsers libraries
ii  libnvparsers8                                               8.4.3-1+cuda11.6                  amd64        TensorRT parsers libraries
ii  onnx-graphsurgeon                                           8.4.3-1+cuda11.6                  amd64        ONNX GraphSurgeon for TensorRT package
ii  python3-libnvinfer                                          8.4.3-1+cuda11.6                  amd64        Python 3 bindings for TensorRT
ii  python3-libnvinfer-dev                                      8.4.3-1+cuda11.6                  amd64        Python 3 development package for TensorRT
ii  tensorrt                                                    8.4.3.1-1+cuda11.6                amd64        Meta package for TensorRT

```
