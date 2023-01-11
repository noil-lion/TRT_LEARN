# OpenCV parallel_for_ 并行化
在使用OpenCV进行图像处理时，其计算量是非常大的，在实际运行中如何高效的计算是很有必要的，现有的方法有很多，如 OpenMp, TBB, OpenCL 还有 Nvidia 的 CUDA。CUDA虽然能利用GPU进行并行加速，但不太适合毫秒级程序运行，主要是因为CPU和GPU间传输相对耗时（单一张图片使用opencv的cuda函数传输时间300ms），所以一般选择在CPU上进行高效计算，OpenCV 的并行计算函数 parallel_for_，它整合了上述的多个组件。


## OpenCV提供的并行框架
1. 英特尔线程构建块（第三方库，应显式启用），如TBB(Thread Building Blocks)。
2. OpenMP（集成到编译器，应该被显式启用）。
3. APPLE GCD（系统范围广，自动使用（仅限APPLE））
4. Windows RT并发（系统范围，自动使用（仅Windows RT））
5. Windows并发（运行时的一部分，自动使用（仅限Windows） - MSVC ++> = 10））
6. Pthreads
__OpenCV 库中可以使用多个并行框架，可以使用这些库来访问并行框架__

## Parallel_for_
Parallel_for_ 被介绍为parallel data processor，有两种使用方式。
1. 包含头文件
```
//首先需要包含<opencv2/core/utility.hpp>头文件
```
2. 使用方式一
```
void cv::parallel_for_ (const Range &range, const ParallelLoopBody &body, double nstripes=-1.)
```
1. 使用方式二
```
static void cv::parallel_for_ (const Range &range, std::function< void(const Range &)> functor, double nstripes=-1.)
```
## Parallel_for_的使用步骤（结合ParallelLoopBody使用）
1. 自定义一个类或结构体，使这个结构体或者是类继承自 ParallelLoopBody 类
```
class MyParallelClass : public ParallelLoopBody
{}
struct MyParallelStruct : public ParallelLoopBody
{}
```

2. 在自定义的类或者是结构体中，重写括号运算符（ ），这里只能接受一个 Range 类型的参数（这是与一般的重载不一样的地方），因为后面的parallel_for_需要使用。
```
void operator()(const Range& range)
{
   //在这里面进行“循环操作”
}
```

3. 使用 parallel_for_ 进行并行处理
```
parallel_for_(Range(start, end), MyParallelClass(构造函数列表));
//Range(start, end) 就是一个Range对象
//MyParallelClass(构造函数列表) 就是一个继承自ParallelLoopBody的类的对象

```

