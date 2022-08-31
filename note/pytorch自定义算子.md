# Pytorch自定义算子
PyTorch是如何调用自定义的CUDA算子的。

## cuda算子文件实现
这里实现的功能是两个长度为的tensor相加，每个block有1024个线程，一共有个n/1024个block。
```
//add2.h
void launch_add2(float *c,
                 const float *a,
                 const float *b,
                 int n);

//add2.cu
// device端执行函数
__global__ void add2_kernel(float* c,
                            const float* a,
                            const float* b,
                            int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
            i < n; i += gridDim.x * blockDim.x) {
        c[i] = a[i] + b[i];
    }
}

// host端执行函数
void launch_add2(float* c,
                 const float* a,
                 const float* b,
                 int n) {
    dim3 grid((n + 1023) / 1024);
    dim3 block(1024);
    add2_kernel<<<grid, block>>>(c, a, b, n);
}
```

## Torch C++封装
这里是pytorch和C++的调用接口封装，将cuda算子封装成一个pytorch可调用接口
```
//add2.cpp
#include <torch/extension.h>  //torch的C++扩展库
#include "add2.h"             //cuda库文件

// torch_launch_add2函数传入的是C++版本的torch tensor，
//然后转换成C++指针数组，调用CUDA函数launch_add2来执行核函数。
void torch_launch_add2(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       int n) {
    launch_add2((float *)c.data_ptr(),
                (const float *)a.data_ptr(),
                (const float *)b.data_ptr(),
                n);
}

//将cuda执行函数再封装，用pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("torch_launch_add2",
          &torch_launch_add2,
          "add2 kernel warpper");
}

```
## 文件结构
```
├── include
│   └── add2.h # cuda算子的头文件
├── kernel
│   ├── add2_kernel.cu # cuda算子的具体实现
│   └── add2.cpp # cuda算子的cpp torch封装
├── CMakeLists.txt
├── LICENSE
├── README.md
├── setup.py
└── train.py # 使用cuda算子来训练模型
```
## pytoch编译调用

* 即时编译调用
```
# train.py
import torch
from torch.utils.cpp_extension import load

cuda_module = load(name="add2",
                   sources=["add2.cpp", "add2.cu"],
                   verbose=True)   //编译，sources参数，指定了需要编译的文件列表

 cuda_module.torch_launch_add2(cuda_c, a, b, n)  //调用
```

* Setuptools
```
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
setup(
    name="add2",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "add2",
            ["kernel/add2.cpp", "kernel/add2_kernel.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
```
接着执行：
> python3 setup.py install
这样可以生成动态链接库，同时将add2添加为python的模块了，可以直接import add2来调用。


* cmake
```
# CMakeLists.txt
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
# 修改为你自己的nvcc路径，或者删掉这行，如果能运行的话。
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(add2 LANGUAGES CXX CUDA)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
# 修改为你自己的python路径，或者删掉这行，如果能运行的话。
include_directories(/usr/include/python3.7)
include_directories(include)
set(SRCS kernel/add2.cpp kernel/add2_kernel.cu)
add_library(add2 SHARED ${SRCS})
target_link_libraries(add2 "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")
```
cpp端用的是TORCH_LIBRARY进行封装：
```
TORCH_LIBRARY(add2, m) {
    m.def("torch_launch_add2", torch_launch_add2);
}
```
编译指令
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ../
make
```

调用
```
import torch
torch.ops.load_library("build/libadd2.so")
torch.ops.add2.torch_launch_add2(c, a, b, n)
```

