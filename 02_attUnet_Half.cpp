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

const int IMAGE_SIZE = 1120;
const int IMAGE_CHANNEL = 3;
int main(){

    try {
        torch::jit::script::Module module = torch::jit::load(R"(C:\Users\ZY\CLionProjects\att_unet_cpp\attunet_half.pt)",at::kCUDA);

//        module.to(torch::kCUDA);

//        cout<<inputdata<<endl;
        cout<<".........1........."<<endl;
        // Create a vector of inputs.
//        std::vector<torch::jit::IValue> inputs;
//        inputs.emplace_back(torch::ones({1, 3, 768, 768}));

        //加载图片
        Mat inputImg , showimg;
        string imgRoot = "F:\\model\\_images\\不可接受碳化物 (1).bmp";
        at::Tensor data;

        double t = (double)cv::getTickCount();
        {
            inputImg = cv::imread(imgRoot,-1);
            resize(inputImg,inputImg,Size(IMAGE_SIZE,IMAGE_SIZE));
            resize(inputImg,showimg,Size(768,768));
            imshow("input",showimg);
            cout<<".........inputimage........."<<endl;
            cvtColor(inputImg,inputImg,COLOR_BGR2RGB);

            inputImg.convertTo(inputImg,CV_32FC3, 1.0f/255.0f); // 转换为32 float ，alpha 实现归一化



            data = at::from_blob(inputImg.data,{1,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNEL});
            data = data.permute({0,3,1,2}).cuda();
            data = at::div(at::sub(data,0.5),0.5).toType(torch::kFloat16);
        }
        cout<<".........forward........."<<endl;
//        at::Tensor out_tensor = module.forward( {torch::ones({1, 3, 768, 768}).to(at::kCUDA)} ).toTensor();
        at::Tensor out_tensor = module.forward( {data} ).toTensor();
        cout<<"........2.........."<<endl;
        out_tensor = torch::gt(torch::sigmoid(out_tensor),0.5).toType(torch::kFloat16); //激活到0-1之间，网络模型中没有，所以在这里加
        cout<<"网络输出成功"<<endl;

        // 输出转为 opencv 格式
        {
            //            tips:
            //            1.squeeze只用于batchsize为1的场景
            //            2.permute 是将存储格式从pytorch形式转成opencv格式
            //            3.因为在处理前对cvmat中的值做了归一化，所以现在要*255恢复，同时对于不在0-255范围内的数据，需要做限制
            //            4.因为cvmat的数据格式是8UC3，所以torch tensor要提前转换成kU8
            //sequeeze trans tensor shape from 1*C*H*W to C*H*W
            //permute C*H*W to H*W*C
            out_tensor = out_tensor.squeeze(0).detach().permute({1, 2, 0}); //踩坑：不加轴，导致前两维都被删除
            cout<<"........3.........."<<endl;
            //see tip3，tip4
            out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);
            cout<<"........4.........."<<endl;
            out_tensor = out_tensor.to(torch::kCPU);
            cout<<"........5.........."<<endl;

            cv::Mat resultImg(IMAGE_SIZE, IMAGE_SIZE, CV_8UC1); //(行数，列数，类型)
            cout<<"........6.........."<<endl;
            //copy the data from out_tensor to resultImg
            std::memcpy((void *) resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8) * out_tensor.numel());

            cout<<"显示结果图"<<endl;
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            cout << "从读取图片到网络输出结果 cost time： " <<  t << endl;

            //            Mat img = cv::imread(R"()",IMREAD_GRAYSCALE);
            resize(resultImg,resultImg,Size(768,768));
            cv::imshow("result_img",resultImg);
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
    }catch(Exception e) {
        cout<<e.what()<<endl;
    }
//    catch(const c10::Error& e) {
//        cout<<e.what()<<endl;
//    }
    return 0;
}
