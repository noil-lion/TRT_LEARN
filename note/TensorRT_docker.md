# 基于docker的[TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/running.html#running)
TensorRT除了可以基于db安装包安装外，nvidia提供了不同版本环境下的TensorRT镜像文件，在tnensorRT模型转换及序列化，需要序列化器版本一致，最佳的方式为拉取相同版本的镜像文件，进行模型转换。
***
## 基于镜像安装
* Docker 引擎将对应image加载到运行软件的容器中。
* 可以使用的附加标志和设置来定义容器的运行时资源。
* 
1. 拉一个容器
安装好docker，并注册成为Nvidia DGX用户。
```
docker pull nvcr.io/nvidia/tensorrt:22.07-py3
```

2. 运行容器
```
docker run --gpus all -it --rm -v local_dir:container_dir nvcr.io/nvidia/tensorrt:<xx.xx>-py<x>
docker run --gpus '"device=4"' -it --rm -v /work:/work nvcr.io/nvidia/tensorrt:22.04-py3

# 执行模型转换
trtexec --onnx=resnet50_pytorch.onnx --saveEngine=resnet.engin
```

3. 添加其他包或扩展
* 基于此为基镜像，使用docker build 添加自定义，再重定义
* 修改源代码
