# 记录Pytorch -> ONNX -> engine的错误及bug解决
1. 多节点输出网络的模型保存和onnx转换
   由于Unet网络结构为全卷积网络级联，为了避免中间层的特征丢失难以训练等问题，优化训练方式选为深监督训练，其具体实现为在训练阶段每个子网络会有一个辅助输出结果，赋予不同权重用来加强损失(深监督)，分辨率越高，赋予权重越大，这就导致网络在每一层都有不同维度的输出结果，但在推理阶段，我们只需要一个可用输出即可，可以通过设置do_ds为flase即可关闭辅助输出，以完整网络的输出为推理结果，也就不会出现多个输出节点的异常。
   nnUnet:将do_ds设置为false

2. while TensorRT does not natively support INT64. Attempting to cast down to INT32
    检查TensorRT版本对应ONNX支持版本：https://docs.nvidia.com/deeplearning/tensorrt/release-notes/tensorrt-8.html#tensorrt-8  
    我们转化后的ONNX模型的参数类型是INT64，然而：TensorRT本身不支持INT64，而对于INT32的精度，TensorRT是支持的，因此可以尝试把ONNX模型的精度改为INT32，然后再进行转换
   ```
    pip install onnx-simplifier
    python -m onnxsim your_model.onnx your_model_sim.onnx

   ```