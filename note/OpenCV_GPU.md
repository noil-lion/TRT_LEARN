## OpenCV_GPU
在大多数情况下，OpenCV都使用CPU执行计算，这并不总能保证所需的计算性能。在 NVIDIA 的支持下，2010年引入了GPU模块使用CUDA提供GPU加速。目前已经支持的模块：
* Core part 
* Background Segmentation
* Video Encoding/Decoding
* Feature Detection and Description
* Image Filtering
* Image Processing
* Legacy support：
* Object Detection
* Optical Flow
* Stereo Correspondence
* Image Warping
* Device layer

## GpuMat
GpuMat作为显存数据的主要容器，所有的GPU函数都将以GpuMat接收为输入和输出参数。通过引入一个新的类cv::gpu::GpuMat，可以减少在CPU和GPU间复制数据的开销。
```
#include <opencv2/cudaimgproc.hpp> 

cv::Mat img = cv::imread("image.png", IMREAD_GRAYSCALE); 
cv::cuda::GpuMat dst, src;
```

## CPU/GPU数据传递
数据传递与CUDA异构编程模型一致
1. CPU->GPU
   上传：将数据从主机内存（host）复制到设备显存（device）
   ```
   cv::cuda::GpuMat dst, src;
   src.upload(img);
   ```
2. GPU->CPU
   下载：将数据从设备显存（device）复制到主机内存（host）
   ```
   cv::Ptr<cv::cuda::CLAHE> ptr_clahe = cv::cuda::createCLAHE(5.0, cv::Size(8, 8)); 
   ptr_clahe->apply(src, dst); 
   cv::Mat result; 
   dst.download(result);
   ```

## 多GPU使用
默认情况下，OpenCV CUDA算法执行默认使用单个GPU，如果需要多个GPU，则必须手动分配设备，使用API。
```
cv::cuda::setDevice (cv2.cuda.SetDevice);
```
