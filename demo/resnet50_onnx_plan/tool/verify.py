from tokenize import Double
import onnxruntime as ort
import torch
from torch import nn
import numpy as np
	
ort_session = ort.InferenceSession('/work/wuzihao/ibotModelDeploy/modelTrans/result/heart/heart.onnx')
net_input = np.ones([1, 1, 160, 176, 176], dtype =np.float32)
outputs = ort_session.run(None, {ort_session.get_inputs()[0].name: net_input})
print(np.array(outputs[0]).shape)
net_input = torch.ones(1,1, 160, 176, 176)
net_input2 = torch.ones(1, 1, 160, 176, 176)
torch_model = torch.jit.load("/work/wuzihao/ibotModelDeploy/ibotModel/pulmonary/0.1/heart/1/model.pt") # pytorch模型加载
with torch.no_grad():
    net_input = net_input
    net_input2 = net_input2
    outputs_pt = torch_model(net_input)
    # outputs = torch_model(net_input2)
print(outputs_pt.shape)
loss_fn = nn.CrossEntropyLoss()
# test_loss = loss_fn(outputs_pt.cpu(), torch.from_numpy(np.array(outputs[0]))).item()
# 使用torch自带的二进制交叉熵计算
# loss_bce = torch.nn.BCELoss()(outputs_pt.cpu(), torch.from_numpy(np.array(outputs[0])))
# 1.softmax
s1 = torch.softmax(outputs_pt,dim=1)
s2 = torch.softmax(torch.from_numpy(np.array(outputs[0])),dim=1)
loss = nn.CrossEntropyLoss()(s1, s2)
print(outputs[0].shape)