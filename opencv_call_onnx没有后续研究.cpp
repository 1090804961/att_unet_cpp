//
// Created by ZY on 2020-05-02.
//
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

using namespace std;
using namespace cv;
using namespace torch;

int main(){
    try {
        auto model = cv::dnn::readNetFromONNX(R"(C:\Users\ZY\CLionProjects\att_unet_cpp\conv.onnx)");

        auto data = torch::ones({1,1,20,20});

        Mat d = Mat::ones(Size(1,1),CV_8SC3);
//        model.forward(d);
//
//        cout<<d<<endl;
//        cv::dnn::readNetFromTorch(R"(C:\Users\ZY\CLionProjects\att_unet_cpp\resnet18.pt)",false);
    }catch(exception e) {
        cout<<e.what()<<endl;
    }
    return 0;

}