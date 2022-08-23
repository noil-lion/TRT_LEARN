import torchvision.models as models
import torch
import torch.onnx


BATCH_SIZE = 32     # 模型batch
dummy_input=torch.randn(BATCH_SIZE, 3, 224, 224)  # 模型输入维度
# load the pretrained model
resnet50 = models.resnet50(pretrained=True, progress=False).eval()

# 从PyTorch保存ONNX文件
torch.onnx.export(resnet50, dummy_input, "resnet50_pytorch.onnx", verbose=False)

# 有两种方式将ONNX转为TensorRT引擎，一个是trtexe，还有一个是用TensorRT API构建训练网络，并保存为推理引擎
# 本例采用学习成本更低的trtexe进行ONNX转TENSORRT {trtexec --onnx=resnet50_onnx_model.onnx --saveEngine=resnet_engine.trt}

