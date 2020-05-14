#include <cuda_runtime_api.h>
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "fasterrcnn.hpp"
#include<iostream>
#include<vector>
#include<string>
#include<chrono>
using namespace std;
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
//包含纯虚函数的类不允许被实例化

int main() {
	
	string image_path = "D:/code/TRT_FasterRcnn/image/2.jpg";
	string prototxt_path = "D:/code/TRT_FasterRcnn/data/faster_rcnn_test_iplugin.prototxt";
	string model_path = "D:/code/TRT_FasterRcnn/data/VGG16_faster_rcnn_final.caffemodel";
	FasterRcnn faster(prototxt_path, model_path);
	string img_path = "D:/code/TRT_FasterRcnn/image/3.jpg";
	auto starttime = std::chrono::steady_clock::now();
	faster.Forward(prototxt_path, model_path, img_path);
	auto endtime = std::chrono::steady_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (endtime - starttime);
	cout << "推理时间为" << duration.count() << endl;
	faster.Deinitialize();
	//float *result = faster.prepareImg(image_path);
	//cout << result[1000] << endl;
	////cout << data << endl;
	//cout << sizeof(double) << endl;
	return 0;
}