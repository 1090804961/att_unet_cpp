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

char drive[_MAX_DRIVE] = {0};
char dir[_MAX_DIR] = {0};
char fname[_MAX_FNAME] = {0};
char ext[_MAX_EXT] = {0};

double startTime,endTime;

const int IMAGE_SIZE = 1120;
const int IMAGE_CHANNEL = 3;

Mat outImage , showimg;

cv::Mat resultImg(IMAGE_SIZE, IMAGE_SIZE, CV_8UC3); //save result image

at::Tensor intputdata,outputdata;


void cvImage2Tensor(Mat &outImage,at::Tensor & inputdata);
void Tensor2Image(string savePath,at::Tensor inputdata,cv::Mat &resultImg,string fname);

int main(){

    try {
        torch::jit::script::Module module = torch::jit::load(R"(C:\Users\ZY\CLionProjects\att_unet_cpp\attunet_half.pt)",at::kCUDA);
        module.eval();
        cout<<".........loadmodel success........."<<endl;

        //find all imagesname in floder
        string imgRoot = "F:\\model\\cvimg\\*.bmp"; // 获取bmp格式的图片
        string savePath = "F:\\model\\cvtest";

        vector<String> filenames;
        cv::glob(imgRoot,filenames, false); //recursive为false，则只获取中文件夹的图片，不管子文件夹

        double totalTime =(double)cv::getTickCount();

        for(int i=0;i<100;i++){


//            cout<<filenames[i]<<endl;
            outImage = cv::imread(filenames[0],-1); // read  image . BGR or RGB as input : 0.03s

            cvImage2Tensor(outImage,intputdata); //out to ouputdata :CUDAHalfType : 0.015s

//            cout<<".........forward........."<<endl;
            outputdata = module.forward( {intputdata} ).toTensor();
            outputdata = torch::gt(torch::sigmoid(outputdata),0.5).squeeze(); //激活到0-1之间，网络模型中没有，所以在这里加 : 0.01s

            startTime = (double)cv::getTickCount();
            //sequeeze trans tensor shape from 1*C*H*W to C*H*W
            //permute C*H*W to H*W*C
            //------------save image-----------------
            intputdata = at::add(at::mul(intputdata,0.5*255),0.5*255);//归一化到0-255间

            //输出[c,h,w]
            intputdata = intputdata.squeeze(0);
//            outputdata = outputdata.toType(torch::kBool).squeeze();

            //[c,h,w]
            intputdata[0] = intputdata[0].masked_fill(outputdata,255); //mask操作 这里三个通道都被赋值了
            intputdata = intputdata.permute({1, 2, 0}).detach().clamp(0, 255).to(torch::kU8).to(at::kCPU);
            endTime = ((double)cv::getTickCount() - startTime) / cv::getTickFrequency();
            cout << "本次：到保存图片为止的 cost time： " <<  endTime << endl;
            //savemask
//            outputdata = outputdata.squeeze(0).detach().permute({1, 2, 0}); //踩坑：不加轴，导致前两维都被删除
//            outputdata = outputdata.mul(255).clamp(0, 255).to(torch::kU8).to(torch::kCPU);

            _splitpath_s( filenames[i].c_str(), drive, dir, fname, ext );
//            cout<<".......forward finally........"<<outputdata.sizes()<<outputdata.type()<<endl;
            Tensor2Image(savePath,intputdata,resultImg,string(fname)); // ouputdata is input

        }
        endTime = ((double)cv::getTickCount() - totalTime) / cv::getTickFrequency();
        cout << "总计耗时 cost time： " <<  endTime << endl;

    }catch(Exception e) {
        cout<<e.what()<<endl;
    }
    return 0;
}

void cvImage2Tensor(Mat &outImage,at::Tensor & inputdata){


    resize(outImage,outImage,Size(IMAGE_SIZE,IMAGE_SIZE));

    cvtColor(outImage,outImage,COLOR_BGR2RGB); //BGR to RGB

    outImage.convertTo(outImage,CV_32FC3, 1.0f/255.0f); // 转换为32 float ，alpha 实现归一化

    inputdata = at::from_blob(outImage.data,{1,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNEL}); //Mat to Tensor
    inputdata = inputdata.permute({0,3,1,2}).cuda();
    inputdata = at::div(at::sub(inputdata,0.5),0.5).toType(torch::kFloat16);
}

void Tensor2Image(string savePath,at::Tensor inputdata, cv::Mat &resultImg,string fname){

    //copy the data from out_tensor to resultImg
    std::memcpy((void *) resultImg.data, inputdata.data_ptr(), sizeof(torch::kU8) * inputdata.numel());

    // save image
    savePath = savePath+"/"+fname+".jpg";
//    cout<<savePath<<endl;
    cv::cvtColor(resultImg,resultImg,COLOR_RGB2BGR);
    cv::imwrite(savePath,resultImg);
}