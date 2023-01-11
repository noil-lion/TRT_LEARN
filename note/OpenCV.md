# Opencv_cuda
## OpenCV概述
* 图像处理：一般指数字图像处理，包括图像压缩，增强和复原，匹配、描述和识别3个部分。
* 计算机视觉：用摄影机和电脑代替人眼对目标进行识别、跟踪和测量等机器视觉，并进一步做图像处理。
* OpenCV：全称"Open Source Computer Vision Library"，直译就是"开源计算机视觉库"，是一个开源的跨平台计算机视觉和机器学习软件库，它实现了图像处理和计算机视觉方面的很多通用算法。支持运行在Linux、Windows、Mac OS、Android、iOS、Maemo、FreeBSD、OpenBSD等操作系统上，提供C++、python、java、MATLAB接口。

## OpenCV模块

|模块|	功能|
|:-|:-|
|opencv_core|定义了被所有其他模块和基本数据结构(包括重要的多维数组Mat)使用的基本函数.包含核心功能,尤其是底层数据结构和算法函数|
|opencv_imgproc|一些图像处理函数,包括滤波(线性的和非线性的),几何变换,颜色空间变换,直方图等|
|opencv_highgui|提供简单的用户接口功能.包含读写图像及视频的函数,以及操作图形用户界面函数|
|opencv_imgcodecs|一个用于读写图像的易用接口|
|opencv_feature2d|用于特征检测(角点对象和平面对象), 特征描述,特征匹配 等的一些函数.包含兴趣点检测子,描述子以及兴趣点匹配框架|
|opencv_calib3d|摄像机校准,包含相机标定,双目几何故事以及立体视觉函数|
|opencv_photo|包含计算摄影学, 涉及修复/去噪/高动态范围(HDR)图像等|
|opencv_stitching|用于图像拼接|
|opencv_videoio|对于视频捕获和视频编码器是一个易用的接口|
|opencv_videostab|视频稳定|
|opencv_video|提供了视频分析的功能(运动估计,背景提取以及对象跟踪)|
|opencv_objdetect|用于对象检测和预定义检测器实例(例如,人脸/眼睛/微笑/人/车等)的一些函数|
|pencv_ml|机器学习|
|opencv_flann|聚类和搜索,计算几何|
|opencv_shape	|形状距离和匹配|
|opencv_superres|超分辨率|
|opencv_contrib	|第三方代码|
|opencv_legacy	|废弃的代码|
|opencv_gpu	|GPU加速的代码|

__这些模块都有一个单独的头文件(位于include文件夹).典型的OpenCV C++代码将包含所需的模块,声明方式如下__

```
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
```

## Opencv 安装
[Link](https://blog.csdn.net/qq_43193873/article/details/126144636?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167293553116782425677778%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167293553116782425677778&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-126144636-null-null.142^v70^one_line,201^v4^add_ask&utm_term=opencv%E5%AE%89%E8%A3%85%20ubuntu&spm=1018.2226.3001.4187)
