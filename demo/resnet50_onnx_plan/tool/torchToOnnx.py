import torch
torch_model = torch.jit.load("/work/wuzihao/ibotModelDeploy/modelTrans/result/vessel/model.pt") # pytorch模型加载
batch_size = 1  #批处理大小
input_shape = (1, 1, 224, 224, 224)   #输入数据

jit_layer1 = torch_model
print(jit_layer1.graph) 
print(jit_layer1.code) 


"""
#在torch 1.6版本中重新加载一下网络参数
model = torch_model.to(device) #实例化模型并加载到cpu货GPU中
model.load_state_dict(torch.load("/work/wuzihao/ibotModelDeploy/ibotModel/pulmonary/0.1/heart/1/model.pt"))  #加载模型参数，model_cp为之前训练好的模型参数（zip格式）
#重新保存网络参数，此时注意改为非zip格式
torch.save(model.state_dict(), model_cp,_use_new_zipfile_serialization=False)

"""
# set the model to inference mode
torch_model.eval()

x = torch.randn(1, 1, 224, 224, 224)		# 生成张量
export_onnx_file = "../result/vessel/vessel.onnx"					# 目的ONNX文件名
with torch.no_grad(): 
    torch.onnx.export(torch_model, x,
                    export_onnx_file,
                    opset_version=9, 
                    input_names=["INPUT__0"],		# 输入名
                    output_names=["OUTPUT__0"],	    # 输出名
                    dynamic_axes={"INPUT__0":{0:"batch_size"},		# 批处理变量
                                    "OUTPUT__0":{0:"batch_size"}})
