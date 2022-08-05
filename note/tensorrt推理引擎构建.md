# 模型推理引擎构建之.pt->.engine方案
* 导出网络定义以及相关权重
* 解析网络定义以及相关权重
* 根据显卡算子构造出最优执行计划
* 将执行计划序列化存储
* 反序列化执行计划
* 进行推理
第三步表明tensorrt转换的模型是与硬件绑定的，当cuda和cudnn发生改变，那模型就得重新转换。

## 模型转换方式
1. trtexec
示例：
```
def torch2onnx(model_path,onnx_path):
    model = load_model(model_path)
    test_arr = torch.randn(1,3,32,448)
    input_names = ['input']
    output_names = ['output']
    tr_onnx.export(
        model,
        test_arr,
        onnx_path,
        verbose=False,
        opset_version=11,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"input":{3:"width"}}            #动态推理W纬度，若需其他动态纬度可以自行修改，不需要动态推理的话可以注释这行
    )
    print('->>模型转换成功！')
```
执行指令  
> ./trtexec --onnx=repvgg_a1.onnx --saveEngine=repvgg_a1.engine --workspace=1024  --fp16

# 模型推理引擎构建之tensorrt API搭建
使用Tensorrt API搭建目标网络结构，目前INetworkDefiniton类中已经将常见的、简单的网络层实现了（比如卷积层、池化层、激活函数层等），只要在搭建的时候使用就行了，有部分操作是不支持的，不过可以通过自定义或者使用Plugin的形式，将某些网路层的计算逻辑重写出来，并将其作为网络中的某一个层进行使用。

卷积实例使用及参数
```
// 添加卷积层函数
IConvolutionLayer* INetworkDefinition::addConvolutionNd	(
ITensor & input,  # 卷积层输入张量，数据类型为ITensor
int32_t nbOutputMaps,  # 输出特征图参数，
Dims kernelSize,   # 卷积核尺寸
Weights kernelWeights,  # 卷积核权重
Weights biasWeights  # 卷积偏置
)

//实例：给网络添加kernel size为3，stride为2，padding为1的卷积层

IConvolutionLayer* conv1 = network-> addConvlutionNd(input, middle, DImsHW{k, k}, weightMap[lname + ".weight"], weightMap[lname +  ".bias"]);  //添加卷积层实例
assert(conv1);  //
conv1->setStrideNd(DimsHW{2, 2});
conv1->setPaddingNd(DimsHW{1, 1});
```
