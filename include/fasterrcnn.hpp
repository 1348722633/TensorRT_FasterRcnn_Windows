#include <iostream>
#include <vector>
#include <string>
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using Severity = nvinfer1::ILogger::Severity;
#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cout << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)


class Logger : public nvinfer1::ILogger {
public:
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
	nvinfer1::ILogger& getTRTLogger() {
		return *this;
	}

};

class FasterRcnn {
 public:
   FasterRcnn(std::string prototxt_file, std::string model_file);
   void Initialize(std::string prototxt_file, std::string model_file);
   void Deinitialize();
   float* prepareImg(std::string img_path);
   void caffeToTRTModel(const std::string& deployFile,
	                    const std::string& modelFile,
	                    const std::vector<std::string> &outputs,
	                    unsigned int maxBatchSize,
	                    nvinfer1::IHostMemory** trtModelStream);
   void doInference(nvinfer1::IExecutionContext& context, float* inputData,
	                float* inputImInfo, std::vector<float>& outputBboxPred, 
	                std::vector<float>& outputClsProb, std::vector<float>& outputRois, 
	                int batchSize);
   void bboxTransformAndClip(std::vector<float>& rois, std::vector<float>& deltas, std::vector<float>& predBBoxes,
	                         float* imginfo, int batch_size, const int nmsMaxOut, const int numCls);
   std::vector<int> nms(std::vector<std::pair<float, int>> &scores_index, float* bbox, const int classNum,
	                    const int numClasses, const float nms_threshold);
   void Forward(std::string prototxt_path, std::string caffemodel_path, std::string img_path);
  private:
	nvinfer1::IHostMemory* trtModelStream_;
	nvinfer1::IRuntime* runtime_;
	nvinfer1::ICudaEngine* engine_;
	nvinfer1::IExecutionContext* context_;
	Logger logger_;
	std::vector<char*> outputs_;
	std::vector<std::string> outputs_caffe_;
	std::vector<char*> inputs_;
	int output_cls_;
	int output_pred_;
	int imginfo_;
	int batchsize_;
	std::string  *classes_;
	std::vector<int> input_shape_;
	int nms_maxout_;
	cv::Mat image_;
};