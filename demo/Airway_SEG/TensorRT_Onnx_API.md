# TensorRT 加载解析Onnx——Minist模型推理实例API解析
[C_API](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#c_topics)
## 基本流程
ONNX模型->创建并运行TensorRT引擎，演示如何使用TensorRT API将ONNX模型加载构建TensorRT引擎并进行推理。  
```mermaid 
flowchart LR
ONNX模型-->TensorRT模型-->推理引擎构建-->加载引擎进行推理
```
1. Pytorch导出模型ONNX格式  
1. 创建builder和网络实例  
1. 解析ONNX  
1. 构建engine  

## Build Phase-将onnx模型转为TensorRT模型
1. 头文件：NvInfer.h
   C++ API可以通过该头文件进行访问，记得使用nvinfer1作为命名空间。
   ```
   #include "NvInfer.h"
   using namespace nvinfer1;
   ```
   TensorRT C++ API接口类一般以前缀I开头，如ILogger、IBuilder等， CUDA context会在第一次调用CUDA时自动创建。

1. 实例化ILogger接口，该实例对象会捕获所有警告信息，忽略信息性消息。在创建构建器Builder时被需要。
   ```
   // 集成接口类ILogger，并重载log函数
   class Logger : public ILogger
   {
     void log(Severity severity, const char* msg) noexcept override
     {
        if (severity <= Severity::KWARNING):   //过滤info级别的消息，仅输出warning级的消息
            std::cout<< msg << std::endl;
     }
   }logger;  //实例化一个logger对象
   ```

1. 创建构建器builder实例指针
   ```
   nvinfer1::IBuilder* builder = std::unique_ptr(nvinfer1::createInferBuilder(logger.getTRTLogger()));  //关键API一，基于logger实例创建  builder实例
   ```
 

1. 创建网络的结构定义和配置
   创建好builder后，第一步是创建网络定义  
   ```
   nvinfer1::INetworkDefinition* network = nvinfer1::INetworkDefinition(builder->createNetwork());  //关键API二，创建网络实例指针

   nvinfer1::IbuilderConfig* cofig = nvinfer1::IBuilderConfig(builder->createBuilderConfig());  //关键API三，创建cofig实例指针

   nvonnxparser::IParser* parser = nvonnxparser::IParser(nvonnxparser::createParser(*network, Logger));  //关键API四，基于网络实例指针和Logger实例对象创建onnx解析器parser，后期parser解析好onnx文件后填充到network结构里，并用于最终的engine构建。

   constructNetwork函数用于构建网络所需的各项定义
   bool constructNetwork(builder, network, config, paser)
   {
    auto parsed = parser->parserFromFile(locateFile(params.onnxFilename, params.dataDirs).c_str())   //关键API五，从文件解析onnx网络,locateFile 函数根据传入参数的onnx文件路径，返回onnx文件路径以string格式,paserFromFileAPI从读取到的onnx模型进行结果和参数的解析。

    builder->setMaxBatchSize(params.batchSize);  //设置maxBatchSize接口
    config->setMaxWorkspaceSize(16MiB);   //设置转换模型配置的最大工作内存空间
    //还可以设置转换的推理精度根据传入参数params.fp16
    if(params.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);  //设置配置器为fp16推理精度，setFlag API
    }
    return true;  //完成网络构建前的一切初始化工作后，返回执行成功。
   }
   Engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->builderEngineWithCofig(*network, *config), samplesCommon::InferDeleter());  // builderEngineWithCofig函数用于根据cofig配置构建engine实例
   // 然后后面是以下网络network的参数设置输入输出个数、维度等

   ```

1. 完成序列化引擎构建
   ```
   // 根据前面准备好的network、cofig，构建执行plan实例
   IHostMemory(plan{builder->buildSerializedNetwork(*network, *cofig)});  

   
   ```


## Infer Phase-加载序列化引擎并进行推理API
1. 序列化并保存接口
   ```

   ```

1. 反序列化engine接口
   ```
   // 构建runtime运行时实例，与builder一样，需要logger记录器实例
   auto runtime = CreateInferRuntime(Logger);
    //构建engine引擎实例通过plan，engine也和序列化为文件保存在磁盘，后期通过反序列化进行推理引擎实例化并实现推理
   auto Engine = runtime->deseriazeCudaEngine(plan->data(), plan->size());
   ```
1. 创建RAII Buffer 管理对象实例
   common::BUfferManager buffer(Engine);  //以构建好的engine创建buffer管理者

1. 创建执行上下文对象context，优构建好的Engine实例对象创建，一个engine可以有多个执行上下文，用于多个重复推理任务。
   ```
   auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(Engine->createExecutionContext());
   ```

1. 推理缓冲区：用于给输入输出传递提供缓冲区，在GPU中申请两块显存，使用指针数组分别指向输入输出缓冲区。
   

## 头文件
