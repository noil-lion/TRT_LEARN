

# 支气管算法分割测试代码实例
依赖  
    ITKcommon \ITK工具类  
    Logger \日志类  
    ElementAirwaySegmentation->ElementAlgorithmMachineLearningBase->ElementBase<execute()、loadInput()、run()、setOutput()> \算法类主线继承关系   
    PipelineContext<init()、 reset()、增删改查判别返回码map映射> \算法执行涉及的参数数据流结合体类
    每个算法有自己的返回码范围，返回码代表错误类型或成功等信息。
1. 参数初始化
   ```
   std::string inputFolder = argv[1];   //输入数据文件根目录
   std::string outputFolder = argv[2];  //输出目录
   std::string jsonFilePath = argv[3];  //配置文件载入路径
   std::string seriesID = argv[4];      //子文件夹编号
   std::string inputImagePath = inputFolder + "/" + seriesID + "/" + "image.nii.gz";
   std::string inputMaskPath = inputFolder + "/" + seriesID + "/" + "lung.nii.gz";
   std::string outputMaskPath = outputFolder + "/" + seriesID + "/" + "airway.nii.gz";
   ```

2. 参与对象实例化
   ```
   //Logger 实例化
   Logger::getInstance()->init("ElementAirwaySegmentationTest");
   //启动配置文件加载
   bool configValue = ConfigReader::getInstance().LoadConfig(jsonFilePath);  //配置文件实例化加载
   //执行流程实例化
   std::shared_ptr<PipelineContext> pipelineContext = std::make_shared<PipelineContext>();  //
   pipelineContext->set("inputImage", ItkUtil::readThreeDImage(inputImagePath));  //原始数据
   pipelineContext->set("lungMask", ItkUtil::readThreeDMask(inputMaskPath));      //肺部mask数据
   //算法执行元素ele实例化
   std::shared_ptr<ElementAirwaySegmentation> ele(new ElementAirwaySegmentation());
   ```

3. 算法执行
   ```
   //当前分割实例的算法执行实现
    ele->execute(pipelineContext);  //element依据数据内容流执行算法
  //算法执行实例ele继承自算法执行父类的基类Elementbase，其中囊括算法执行的基本步骤函数，其中run()是核心的执行推理函数，可以被各类算法所继承，这是一颗算法类树，其中具体到气管分割算法类已经到树的第三级了，在这一级中引入infer推理函数来具体实现气管分割算法的run()函数，下面看一下它的类函数定义
  void ElementAirwaySegmentation::run() {
  ibotcommon::Duration duration;
  //前处理
  ibotcommon::Duration preDuration;
  if (pipelineContext_->isSuccess()) preprocess();
  LOG_INFO_FMT("preprocess finish cost {}s.", preDuration.getDuration());
  //推理
  ibotcommon::Duration inferDuration;
  if (pipelineContext_->isSuccess()) infer();  //这里是对接triton或者其它推理后端的推理函数接口，其具体实现在CPP文件里有写
  LOG_INFO_FMT("infer finish cost {}s.", inferDuration.getDuration());
  //后处理
  ibotcommon::Duration postDuration;
  if (pipelineContext_->isSuccess()) postprocess();
  LOG_INFO_FMT("postprocess finish cost {}", postDuration.getDuration());

  LOG_INFO_FMT("run finish cost {}s.", duration.getDuration());

   ```

4. 推理函数接口实现
   ```
    void ElementAirwaySegmentation::infer() {
    std::vector<int> dimOrder = {1, 0, 2};
    std::shared_ptr<TritonInferRequest> inferRequest(new TritonInferRequest(option_, dimOrder));
    inferRequest->initialize();  //Triton infer client实例初始化
    inferRequest->inferPatchwise(normalizeImage_, patchIndexes_, "softmax"); //patch级推理
    outputMask_ = inferRequest->getOutput();
    }
   ```

5. 推理函数与Triton infer的对接函数inferPatchwise()
   ```
   // 这里的写法是每次发一个batch_size大小的patch数据给triton推理，
   // 3. send requests
  // Send requests of 'batch_size' images. If the number of images
  // isn't an exact multiple of 'batch_size' then just start over with
  // the first images until the batch is filled.
  //
  // Number of requests sent = ceil(number of images / batch_size)
  // {batch:{patch:{index_pair:{point:{}}}}}
  size_t patchIdx = 0;
  size_t sentCount = 0;
  bool lastRequest = false;
  size_t patchNum = patchIndexes.size();
  ThreeFlImageType::Pointer patchImage;
  std::vector<uint8_t> patchData;
  int lastNum = 0;

  while (!lastRequest) {
    int schedule = int(float(patchIdx + 1) / patchNum * 100);
    if (schedule % 25 == 0) {
      if (schedule > lastNum) {
        lastNum = schedule;
        LOG_INFO_FMT("infering: {}%", std::to_string(schedule));
      }
    };

    // Reset the input for new request.
    err = inputPtr->Reset();
    if (!err.IsOk()) { THROW_RETURNCODE(IBOT_RETURNCODE::MODEL_INFERENCE_ERROR, "failed resetting input : " + err.Message()); }

    // Set input to be the next 'batch_size' images (preprocessed).
    std::vector<std::vector<std::vector<int>>> batchPatchIndexes;
    for (int idx = 0; idx < option_.maxBatchSize; ++idx) {
      batchPatchIndexes.emplace_back(patchIndexes[patchIdx]);
      patchImage = ItkUtil::cropVolumeWithBoundingBox<ThreeFlImageType>(rawImage, patchIndexes[patchIdx]);
      bool boolValue = ItkUtil::patchImageToInputData(patchImage, dimOrder_, &patchData);
      if (!boolValue) { THROW_RETURNCODE(IBOT_RETURNCODE::MODEL_INFERENCE_ERROR, "patch image to input data error"); }
      if (option_.inputDatatype == "FP16") {
        std::vector<uint8_t> patchDataFP16(patchData.size() / 2, 0);
        for (int id = 0; id < int(patchData.size()) / 4; ++id) {
          float tmpFloat32;
          uint16_t tmpFloat16 = 0;
          memcpy(&tmpFloat32, &(patchData[id * 4]), sizeof(float));
          Float32toFloat16(&tmpFloat16, tmpFloat32);
          // if (id % 50000 == 0) std::cout << "Float32toFloat16: " << id << ", " << tmpFloat32 << std::endl;
          memcpy(&(patchDataFP16[id * 2]), &tmpFloat16, sizeof(uint16_t));
        }
        err = inputPtr->AppendRaw(patchDataFP16);
      } else {
        err = inputPtr->AppendRaw(patchData);
      }

      if (!err.IsOk()) { THROW_RETURNCODE(IBOT_RETURNCODE::MODEL_INFERENCE_ERROR, "failed setting input: " + err.Message()); }
      patchIdx = (patchIdx + 1) % patchNum;
      if (patchIdx == 0) { lastRequest = true; }
    }

    triton_options.request_id_ = std::to_string(sentCount);

    // Send request.
    triton::client::InferResult* result;
    err = grpcClient_->Infer(&result, triton_options, inputs, outputs, httpHeaders);
    if (!err.IsOk()) { THROW_RETURNCODE(IBOT_RETURNCODE::MODEL_INFERENCE_ERROR, "failed sending synchronous infer request: " + err.Message()); }
    std::unique_ptr<triton::client::InferResult> resPtr(result);
    bool boolValue = inferResultToMaskBuffer(std::move(resPtr), batchPatchIndexes, patchIndexes.size(), activation, sentCount, dimOrder_, gaussianMapsPtr,
                                             dataPtr, aggregatedGaussianPtr);
    if (!boolValue) { THROW_RETURNCODE(IBOT_RETURNCODE::MODEL_INFERENCE_ERROR, "failed infer result to result data"); }
    sentCount++;
  }

   ```