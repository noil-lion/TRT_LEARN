# TorchScript : Torch jit tracer 实现解析

## TorchScript？

TorchScript是Pytorch官方提供的模型序列化格式，是Pytorch模型部署的工具之一，也可以实现图优化或与后端对接。  
生成方式有两种：1. 官方接口：trace记录数据流执行方式生成。 2. AST解析直接生成计算图的script方式。

## 生成模型的TorchScript
1. 使用jit组件
作为一种模型部署的惯用范式，通常需要先生成一个模型的中间表示（Intermediate Representation）,IR具有固定的图结构。
```
# 使用resnet18为例
import torch 
from torchvision.models import resnet18 
 
# 使用PyTorch model zoo中的resnet18作为例子 
model = resnet18() 
model.eval() 
 
# 通过trace的方法生成IR需要一个输入样例 
dummy_input = torch.rand(1, 3, 224, 224) 
 
# IR生成 
with torch.no_grad(): 
    jit_model = torch.jit.trace(model, dummy_input) 
```
这里使用trace模式来生成中间表示，所谓trace就是指进行一次模型推理，这次推理会记录所有经过的计算，这些计算记录会整合成计算图。  

