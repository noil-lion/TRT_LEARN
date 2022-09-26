import onnxruntime as ort
import torch
from torch import nn
import numpy as np

model = torch.jit.load("/work/wuzihao/ibotModelDeploy/ibotModel/pulmonary/0.1/heart/1/model.pt") # pytorch模型加载

ort_session = ort.InferenceSession('/work/wuzihao/ibotModelDeploy/modelTrans/result/heart/heart.onnx') # onnx模型加载

input = torch.rand(1, 1, 160, 176, 176)  # 输入初始化

torch_output = model(input).detach().numpy()   # torch模型输出计算

ort_output = ort_session.run(None, {ort_session.get_inputs()[0].name: input.numpy()})[0]  # onnx模型输出计算

print(torch_output.shape)
print(ort_output.shape)


print(np.allclose(torch_output, ort_output, rtol=0.0001,atol=0.0009))       # 对接近和相对接近。多个都接近，那么返回接近，绝对忍耐加上相对忍耐