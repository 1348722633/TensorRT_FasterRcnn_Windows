﻿# TensorRT_FasterRcnn_Windows
Download TensorRT-5.1.5.0(cuda 10) for windows
```
git clone git clone https://github.com/1348722633/TensorRT_FasterRcnn_Windows.git
cd TensorRT_FasterRcnn_Windows 
mkdir TRT
# copy the lib and include dir of TensorRT to TRT or you can change the CMakeLists.txt to change the link path.
# change the the CUDA lib path and include path in CMakeLists.txt
mkdir build
cd build
cmake .. -G"Visual Studio 14 2015 Win64"(the version of visual studio depends on you)
```



