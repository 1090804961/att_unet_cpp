//
// Created by ZY on 2020-05-03.
//
#include <iostream>

#include <opencv2/opencv.hpp>

#include <torch/script.h>
#include <torch/torch.h>
#include <torch/cuda.h>

using namespace torch;
using namespace nn;
using namespace std;

int main() {

//c++ 中的切片，一次智能切一个dim
    auto tt = torch::rand({1,8,2});
    tt = torch::slice(tt,1,0,5);
    cout<<tt<<endl;
    return 0;
}