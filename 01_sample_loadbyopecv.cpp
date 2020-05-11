//
// Created by ZY on 2020-05-01.
//
#include <iostream>

#include <opencv2/opencv.hpp>

#include <torch/script.h>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <torch/nn/functional.h>

using namespace torch;
using namespace nn;
using namespace functional;
using namespace std;
using namespace cv;

const int IMAGE_SIZE = 768;
const int IMAGE_CHANNEL = 3;
int main(){

    try {
        cv::dnn::Net net = cv::dnn::readNetFromTorch(R"(C:\Users\ZY\CLionProjects\att_unet_cpp\attunet.pt)"); //这个有些层不支持，所以不好用

        }catch(Exception e) {
        cout<<e.what()<<endl;
        }
//    catch(const c10::Error& e) {
//        cout<<e.what()<<endl;
//    }
    return 0;
}
