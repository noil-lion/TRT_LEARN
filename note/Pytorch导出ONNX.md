# 部署阶段的PYtorch和ONNX
所谓 PyTorch 转 ONNX，实际上就是把每个 PyTorch 的操作映射成了 ONNX 定义的算子。

## 当Pytorch转换模型到onnx模型出现算子不兼容
pytorch和onnx之间的模型转换存在兼容性问题，也就是pytorch中的一些运算和操作在onnx中不存在对应的算子进行一对一映射，这时如果进行模型转换，onnx会将改操作映射到自己支持的算子上，类似一种翻译，但经常会出现一对多的情况，或者存在无法翻译，或者导出 ONNX 模型时这个操作是无法被记录的（TraceWarning）。

##  操作自定义-PyTorch操作自定义
已经有ONNX算子和PyTorch操作映射，但有对Pytorch操作重定义的需求  
1. 自定义一个实现该操作的PyTorch算子，并将它映射到ONNX的期望算子上，其主要包括两个算子的两个函数的定义symbolic()和forward()函数。
   ```
   class NewOP(torch.autograd.Function): 
   ```
2. 理清缺失算子的输入输出维度及参数，还有该算子所需要实现的操作。
   ```
   # 如pytorch的插值方法interpolate。
   ```
3. 算子的推理行为的指向由算子的forward(ctx, inputs_1, input_2, para_1, ...)方法决定，首个参数必须为ctx，后面为输入数据，及可选参数等。
   ```
   @staticmethod 
    def forward(ctx, input, scales):           # 这里定义一个插值操作，映射到ONNX的resize算子，为实现可控的插值scale，将scales定位为输入参数。
        scales = scales.tolist()[-2:]          # 取输入参数
        return interpolate(input,              # 操作在Pytorch下的实际函数实现
                           scale_factor=scales, 
                           mode='bicubic', 
                           align_corners=False) 
   ```
4. 确定新算子要映射的ONNX算子，映射到ONNX的方法由一个算子的symbolic(g, inputs_1, input_2, para_1, ...)方法决定，首个参数必须为g，ONNX 算子的具体定义由 g.op 实现。g.op 的每个参数都可以映射到 ONNX 中的算子属性：
   ```
   @staticmethod 
    def symbolic(g, input, scales): 
        return g.op("Resize",              # 要映射的ONNX算子
                    input,                 # 输入数据
                    g.op("Constant",       # g.op ONNX算子构成定义
                         value_t=torch.tensor([], dtype=torch.float32)),   # 由ONNX中的Constant算子
                    scales,                                                # 输入参数scales数组
                    coordinate_transformation_mode_s="pytorch_half_pixel", # 算子的属性值设置
                    cubic_coeff_a_f=-0.75, 
                    mode_s='cubic', 
                    nearest_mode_s="floor") 
   ```
5. torch.onnx.export()模型导出
   ```
    x = torch.randn(1, 3, 256, 256) 
 
    with torch.no_grad(): 
        torch.onnx.export(model, (x, factor),   # 模型及输入参数
                        "srcnn3.onnx",          # 导出
                        opset_version=11,       # ONNX算子版本
                        input_names=['input', 'factor'],  # 输入数据name
                        output_names=['output'])          # 输出name
   ```

__以上实例是在已经明确Pytorch操作和ONNX有对应一对一算子时，对Pytorch的操作进行自定义微调，ONNX算子其实在这里没有变化，只是在更新完现有的Pytorch操作后，进行了一次重新绑定__   

__而实际部署中，还会遇到PyTorch操作没有对应的一对一的ONNX算子，常常翻译出一对多甚至是无法映射的情况，这时候，可以定义ONNX算子来拓展 ONNX 的表达能力。__

## 算子兼容-自定义ONNX新算子-https://zhuanlan.zhihu.com/p/513387413
ATen 是 PyTorch 内置的 C++ 张量计算库，PyTorch 算子在底层绝大多数计算都是用 ATen 实现的。
PyTorch 转 ONNX 时最容易出现的问题就是算子不兼容了。在转换普通的torch.nn.Module模型时，PyTorch一方面会用跟踪法-trace执行前向推理，将遇到的操作-算子整合成计算图，此外，Pytorch会把算子翻译成ONNX中定义的算子。这一过程可能会遇到的情况：
1. 一对一翻译
2. 一对多翻译，该算子没有直接对应的ONNX算子
3. 无法翻译，该算子没有定义翻译成ONNX的规则会报错

* 查看Pytorch和ONNX算子的对应情况
  ```
  # 先查看ONNX算子的定义情况
  ONNX算子定义查阅文档：https://github.com/onnx/onnx/blob/main/docs/Operators.md
  # 再查看PyTorch定义的算子的映射关系
  PyTorch与ONNX的算子映射支持文档：https://pytorch.org/docs/master/onnx_supported_aten_ops.html
  不同opset版本:https://github.com/pytorch/pytorch/blob/master/torch/onnx/symbolic_opset9.py
  PyTorch——ONNX：https://github.com/pytorch/pytorch/tree/master/torch/onnx
  ```

* 如何自定义ONNX算子-在Pytorch中支持更多ONNX算子  
情况1. 当ATen中已经实现了该算子，且ONNX中也有相关算子的定义，但相关算子的映射规则没写，这种情况只要为ATen算子补充描述映射规则的符号函数就行了。
```
# 符号函数：Pytorch算子类的一个静态方法，在把 PyTorch 模型转换成 ONNX 模型时，各个 PyTorch 算子的符号函数会被依次调用，以完成 PyTorch 算子到 ONNX 算子的转换。

def symbolic(g: torch._C.Graph, input_0: torch._C.Value, input_1: torch._C.Value, ...): 

# 其中参数包含计算图g，输入value，等参数。

# g有一个方法op，符号函数中调用此方法来为最终的计算图添加一个ONNX算子。
def op(name: str, input_0: torch._C.Value, input_1: torch._C.Value, ...) 

# 其中参数包含算子名称，如果该算子是普通的ONNX算子，只要把它再ONNX文档里的名称填进去就行了。情况复杂的情况可能要新建若干个ONNX算子。


```