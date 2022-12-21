# NVIDIA的GPU

## 显卡命名
* 命名规则 = 产品系列 + 显卡技术 + 级别 + 型号 + 后缀。例：NVIDIA GeForce GTX 1060 Ti

|name|value|说明|
|:--|:--|:--|
|芯片架构|Tesla(2006-2010)、Fermi(2010)、Pascal(2016)、Volta(2017)、Turing(2018)、Ampere(2020)|NVIDIA的GPU芯片架构用科学家名字来命名|
|产品系列|GeForce(面向消费者，通常是游戏卡)、Quadro(用于专业绘图设计)、Tesla(用于大规模的并联电脑运算)|每一类架构的芯片会生产不同的芯片，满足不同场景需要|
|显卡技术|RT：Ray Tracing，影响光线追踪的效率和吞吐 <br> Tensor： AI 和数据科学模型的训练吞吐量和效率 <br> CUDA：并行计算，影响浮点运算处理速度 <br> PCI-e：GPU到内存的接口技术 <br> NVLink：GPU间互联技术 ||
|级别|RTX(带光追等新技术的高端显卡); GTX(极致版); GTS(加强版、略逊与GTX); GT(频率提升版); GS(缩减版,略逊于GT); |性能：RTX-> GTX > GTS > GT > GS|
|型号|版本迭代数|区分是同系列显卡中的性能级别，越大越高端|
|后缀|SE(阉割版); TI(增强版); M(移动端专用); LE(限制版本)||

## 产品比对
|产品型号|芯片架构|CUDA Core 数量|功率|FP32算力|
|:--|:--|:--|:--|:--|
|RTX A2000|Ampere|3328|70W|8.0 TFLOPS|
|Geforce RTX 3080   |Ampere|8960|320W|29.12 TFLOPS|
|Geforce RTX 3080 Ti|Ampere|10240|350W|34.2 TFLOPS|
|Geforce RTX 3090|Ampere|10496|350W|35.6 TFLOP|
|Geforce RTX 3090 Ti|Ampere|10752|450|36.45 TFLOPS|
|NVIDIA A100 PCIe|Ampere|6912|300|19.5 TFLOPS|

|产品型号|芯片架构|CUDA Core 数量|FP32算力|显存带宽|ResNet推理实测|
|:--|:--|:--|:--|:--|:--|
|Geforce RTX 3090 |Ampere|10496|35.58 TFLOPS|935.8GB/s|1.1191s|
|NVIDIA A100 PCIe|Ampere|6912|19.5 TFLOPS|1935 GB/s|0.694s|