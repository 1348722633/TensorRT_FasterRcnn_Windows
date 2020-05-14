#include "fasterrcnn.hpp"
#include <cuda_runtime_api.h>
#include <cassert>
#include <iostream>
using namespace std;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace cv;
void FasterRcnn::Initialize(std::string prototxt_file, std::string model_file) {
	initLibNvInferPlugins(&logger_.getTRTLogger(), "");
	caffeToTRTModel(prototxt_file, model_file, outputs_caffe_, batchsize_, &trtModelStream_);
	runtime_ = createInferRuntime(logger_.getTRTLogger());
	engine_ = runtime_->deserializeCudaEngine(trtModelStream_->data(), trtModelStream_->size(), nullptr);
	trtModelStream_->destroy();
	context_ = engine_->createExecutionContext();
}

void FasterRcnn::Deinitialize() {
	runtime_->destroy();
	engine_->destroy();
	context_->destroy();
}

FasterRcnn::FasterRcnn(std::string prototxt_file, std::string model_file) {
	inputs_ = {"data", "im_info"};
	outputs_ = {"bbox_pred", "cls_prob", "rois"};
	outputs_caffe_ = { "bbox_pred", "cls_prob", "rois" };
	classes_ = new string[21]{ "background", "aeroplane", "bicycle", "bird", 
		                       "boat", "bottle", "bus", "car", "cat",
		                       "chair", "cow", "diningtable", "dog",
				               "horse", "motorbike", "person", "pottedplant", "sheep", 
		                       "sofa", "train", "tvmonitor" };
	const int INPUT_C = 3; const int INPUT_H = 375; const int INPUT_W = 500;
	input_shape_ = {INPUT_C, INPUT_H, INPUT_W};
	imginfo_ = 3; //(h, w, scale);
	output_cls_ = 21;
	nms_maxout_ = 300;
	batchsize_ = 1;
	output_pred_ = output_cls_ * 4;
	trtModelStream_ = nullptr;
	Initialize(prototxt_file, model_file);
}

float* FasterRcnn::prepareImg(std::string img_path) {
	cv::Mat input;
	cv::Mat image = cv::imread(img_path);
	float pixelMean[3]{ 102.9801f, 115.9465f, 122.7717f };
	cv::resize(image, input, cv::Size(input_shape_[2], input_shape_[1]));
	image_ = input;
	int kImageHeight = input.rows; int kImagewidth = input.cols; int kImageChannels = input.channels();
//	cout << "kImageHeight" << kImageHeight << endl;
	/* 减去均值*/
	input.convertTo(input, CV_32FC3);
	for (int row = 0; row < kImageHeight; row++) {
		for (int col = 0; col < kImagewidth; col++) {
			input.at<cv::Vec3f>(row, col)[0] -= pixelMean[0];
			input.at<cv::Vec3f>(row, col)[1] -= pixelMean[1];
			input.at<cv::Vec3f>(row, col)[2] -= pixelMean[2];
		}
	}
	vector<cv::Mat> input_channels(kImageChannels);
	cv::split(input, input_channels);
	//cv::subtract(input_channels[0], pixelMean[0]);
	float* result = new float[kImageHeight * kImagewidth * kImageChannels];
	float* data = result;
	int channellength = kImageHeight * kImagewidth;
	for (int i = 0; i < kImageChannels; i++) {
		memcpy(data, input_channels[i].data, channellength * sizeof(float));
		data += channellength;
	}
	return result;
}

void FasterRcnn::caffeToTRTModel(const std::string& deployFile,
	                             const std::string& modelFile,
	                             const std::vector<std::string> &outputs,
	                             unsigned int maxbatchsize,
	                             nvinfer1::IHostMemory** trtModelStream) {
	IBuilder *builder = createInferBuilder(logger_.getTRTLogger());
	assert(build != nullptr);
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser * parser = createCaffeParser();
	cout << "Begin parsing model..." << std::endl;
	const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(),
		                      modelFile.c_str(), *network, nvinfer1::DataType::kFLOAT);
	cout << "End parsing model..." << std::endl;
	for (auto& s : outputs) {
		cout <<"output blob"<<s.c_str() << endl;
		network->markOutput(*blobNameToTensor->find(s.c_str()));
	}
	builder->setMaxBatchSize(maxbatchsize);
	builder->setMaxWorkspaceSize(10 << 20);
	//builder->setFp16Mode(true);
	std::cout << "build cuda engine" << endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);
	std::cout << "end build cuda engine" << endl;
	network->destroy();
	parser->destroy();
	(*trtModelStream) = engine->serialize();
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void FasterRcnn::doInference(nvinfer1::IExecutionContext& context, float* inputData,
	                         float* inputImInfo, std::vector<float>& outputBboxPred,
	                         std::vector<float>& outputClsProb, std::vector<float>& outputRois,
	                         int batchsize_) {
	const ICudaEngine& engine = context.getEngine();
	assert(engine.getNbBindings() == 5);
	int inputindex0 = engine.getBindingIndex(inputs_[0]);
	int inputindex1 = engine.getBindingIndex(inputs_[1]);
	int outputindex0 = engine.getBindingIndex(outputs_[0]);
	int outputindex1 = engine.getBindingIndex(outputs_[1]);
	int outputindex2 = engine.getBindingIndex(outputs_[2]);
	//cout << "inputindex0 is" << inputindex0 << "outputindex0 is" << outputindex2;
	//cout << "inputdata" << inputData[1000] << endl;
	const int dataSize = batchsize_ * input_shape_[0] * input_shape_[1] * input_shape_[2];
	const int imgInfoSize = batchsize_ * imginfo_;
	const int bboxPredSize = batchsize_ * nms_maxout_ * output_pred_;
	const int clsProbSize = batchsize_ * nms_maxout_ * output_cls_;
	const int roiSize = batchsize_ * nms_maxout_ * 4;
	void* buffers[5];
    CHECK(cudaMalloc(&buffers[inputindex0], dataSize * sizeof(float)));
	CHECK(cudaMalloc(&buffers[inputindex1], imgInfoSize * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputindex0], bboxPredSize * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputindex1], clsProbSize * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputindex2], roiSize * sizeof(float)));
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));
	/*设置输入*/
	CHECK(cudaMemcpyAsync(buffers[inputindex0], inputData, dataSize*sizeof(float), 
		                  cudaMemcpyHostToDevice, stream));
	CHECK(cudaMemcpyAsync(buffers[inputindex1], inputImInfo, imgInfoSize*sizeof(float),
		                  cudaMemcpyHostToDevice, stream));
	context.enqueue(batchsize_, buffers, stream, nullptr);
	/*输出*/
	CHECK(cudaMemcpyAsync(outputBboxPred.data(), buffers[outputindex0], bboxPredSize*sizeof(float),
		                  cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(outputClsProb.data(),  buffers[outputindex1], clsProbSize*sizeof(float),
		                  cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(outputRois.data(), buffers[outputindex2], roiSize * sizeof(float),
		                  cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream); /*等待流执行完成*/

	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputindex0]));
	CHECK(cudaFree(buffers[inputindex1]));
	CHECK(cudaFree(buffers[outputindex0]));
	CHECK(cudaFree(buffers[outputindex1]));
	CHECK(cudaFree(buffers[outputindex2]));
	cout << "inferrence ok" << endl;
} 

void FasterRcnn::bboxTransformAndClip(std::vector<float>& rois, std::vector<float>& deltas, 
	                                  std::vector<float>& predBBoxes,float* imginfo, int batch_size, 
	                                  const int nmsMaxOut, const int numCls) {
	for (int i = 0; i < batch_size * nmsMaxOut; i++) {
		float width = rois[i * 4 + 2] - rois[i * 4] + 1;
		float height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
		float ctr_x = rois[i * 4] + 0.5 * width;
		float ctr_y = rois[i * 4 + 1] + 0.5 * height;
		float* image_offset = imginfo + i / nmsMaxOut * 3;
		for (int j = 0; j < numCls; j++) {
			float dx = deltas[i * numCls * 4 + j * 4];
			float dy = deltas[i * numCls * 4 + j * 4 + 1];
			float dw = deltas[i * numCls * 4 + j * 4 + 2];
			float dh = deltas[i * numCls * 4 + j * 4 + 3];
			float pred_ctr_x = dx * width + ctr_x;
			float pred_ctr_y = dy * height + ctr_y;
			float pred_w = exp(dw) * width;
			float pred_h = exp(dh) * height;
			predBBoxes[i * numCls * 4 + j * 4] = std::max(std::min(pred_ctr_x - 0.5f * pred_w, 
				                                 image_offset[1] - 1.f), 0.f);
			predBBoxes[i * numCls * 4 + j * 4 + 1] = std::max(std::min(pred_ctr_y - 0.5f * pred_h, 
				                                 image_offset[0] - 1.f), 0.f);
			predBBoxes[i * numCls * 4 + j * 4 + 2] = std::max(std::min(pred_ctr_x + 0.5f * pred_w, 
				                                 image_offset[1] - 1.f), 0.f);
			predBBoxes[i * numCls * 4 + j * 4 + 3] = std::max(std::min(pred_ctr_y + 0.5f * pred_h, 
				                                 image_offset[0] - 1.f), 0.f);
		}
	}
}
std::vector<int>FasterRcnn::nms(std::vector<std::pair<float, int>> &scores_index, float* bbox, const int classNum,
	const int numClasses, const float nms_threshold) {
	auto overlap1D = [](float x1min, float x1max, float x2min, float x2max)->float {
		if (x1min > x2min) {
			swap(x1min, x2min);
			swap(x1max, x2max);
		}
		return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
	};
	auto computeIoU = [&overlap1D](float* bbox1, float* bbox2)->float {
		float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
		float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
		float overlap2D = overlapX * overlapY;
		float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
		float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] -  bbox2[1]);
		float u = area1 + area2 - overlap2D;
		return u == 0 ? 0 : overlap2D / u;
	};
	/*do nms*/
	std::vector<int> indices;
	for (auto score_index : scores_index) {
		const int index = score_index.second;
		bool keep = true;
		for (int k = 0; k < indices.size(); k++) {
			if (keep) {
				int idx = indices[k];
				float iou = computeIoU(&bbox[(index * numClasses + classNum) * 4], 
					&bbox[(idx * numClasses + classNum) * 4]);
				keep = (iou < nms_threshold);
			}
			else break;
		}
		if (keep) indices.push_back(index);
	}
	return indices;
}
void FasterRcnn::Forward(string prototxt_path, string caffemodel_path, string img_path) {
	float *data = prepareImg(img_path);
	float imginfo[3]{input_shape_[1], input_shape_[2], 1};
	vector<float> rois;
	vector<float> bboxPreds;
	vector<float> clsProb;
	vector<float> predBBoxes;
	rois.assign(nms_maxout_ * 4, 0);
	bboxPreds.assign(nms_maxout_ * output_cls_ * 4, 0);
	clsProb.assign(nms_maxout_ * output_cls_, 0);
	predBBoxes.assign(nms_maxout_ * output_cls_ * 4, 0);
	doInference(*context_, data, imginfo, bboxPreds, clsProb, rois, batchsize_);
	for (int i = 0; i < batchsize_; i++) {
		for (int j = 0; j < nms_maxout_ * 4; j++) {
			rois[i * nms_maxout_ * 4 + j] /= imginfo[i * 3 + 2];
		}
	}
	/*decode the boxes*/
	bboxTransformAndClip(rois, bboxPreds, predBBoxes, imginfo, batchsize_, nms_maxout_, output_cls_);

	const float nms_threshold = 0.3f;
	const float score_threshold = 0.5f;
	for (int i = 0; i < batchsize_; i++) {
		float* bbox = predBBoxes.data() + i * nms_maxout_ * output_cls_ * 4;
		float* scores = clsProb.data() + i * nms_maxout_ * output_cls_;
		for (int c = 1; c < output_cls_; c++) {
			std::vector<std::pair<float, int>> scores_index;
			for (int r = 0; r < nms_maxout_; r++) {
				if (scores[r * output_cls_ + c] > score_threshold) {
				//	cout << "scores is" << scores[r * output_cls_ + c]<<"c is"<<c;
					scores_index.push_back(std::make_pair(scores[r * output_cls_ + c], r));
					std::stable_sort(scores_index.begin(), scores_index.end(), 
						             [](const std::pair<float,int>& pair1, const std::pair<float,int>& pair2) {
						              return pair1.first > pair2.first;});
				}
			}
			/*nms*/
			std::vector<int> indices = nms(scores_index, bbox, c, output_cls_, nms_threshold);
			cout << "indices size is" << indices.size() << endl;
			/*draw the bbox*/
			string classname = classes_[c];
			for (int k = 0; k < indices.size(); k++) {
				int index = indices[k];
				cout << "draw the box" << endl;
				cv::rectangle(image_,cv::Point(bbox[index * output_cls_ * 4 + c * 4], 
					bbox[index * output_cls_ * 4 + c * 4 + 1]),cv::Point(bbox[index * output_cls_ * 4 + c * 4 + 2],
					bbox[index * output_cls_ * 4 + c * 4 +3]),cv::Scalar(255,0,0),1);
				int center_x = 0.5 * (bbox[index * output_cls_ * 4 + c * 4] + 
					                  bbox[index * output_cls_ * 4 + c * 4 + 2]);
				int center_y = 0.5 * (bbox[index * output_cls_ * 4 + c * 4 + 1] +
					bbox[index * output_cls_ * 4 + c * 4 + 3]);
			    cv::putText(image_, classname, cv::Point(center_x, center_y), 
					         cv::FONT_HERSHEY_COMPLEX, 2, 1, 8, 0);
			}
		}
	}
	cv::imwrite("./result.jpg", image_);
}