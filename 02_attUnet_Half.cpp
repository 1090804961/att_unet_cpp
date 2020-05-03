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

        //����ͼƬ
        Mat inputImg , showimg;
        string imgRoot = "F:\\model\\_images\\���ɽ���̼���� (1).bmp";
        at::Tensor data;

        double t = (double)cv::getTickCount();
        {
            inputImg = cv::imread(imgRoot,-1);
            resize(inputImg,inputImg,Size(IMAGE_SIZE,IMAGE_SIZE));
            resize(inputImg,showimg,Size(768,768));
            imshow("input",showimg);
            cout<<".........inputimage........."<<endl;
            cvtColor(inputImg,inputImg,COLOR_BGR2RGB);

            inputImg.convertTo(inputImg,CV_32FC3, 1.0f/255.0f); // ת��Ϊ32 float ��alpha ʵ�ֹ�һ��



            data = at::from_blob(inputImg.data,{1,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNEL});
            data = data.permute({0,3,1,2}).cuda();
            data = at::div(at::sub(data,0.5),0.5).toType(torch::kFloat16);
        }
        cout<<".........forward........."<<endl;
//        at::Tensor out_tensor = module.forward( {torch::ones({1, 3, 768, 768}).to(at::kCUDA)} ).toTensor();
        at::Tensor out_tensor = module.forward( {data} ).toTensor();
        cout<<"........2.........."<<endl;
        out_tensor = torch::gt(torch::sigmoid(out_tensor),0.5).toType(torch::kFloat16); //���0-1֮�䣬����ģ����û�У������������
        cout<<"��������ɹ�"<<endl;

        // ���תΪ opencv ��ʽ
        {
            //            tips:
            //            1.squeezeֻ����batchsizeΪ1�ĳ���
            //            2.permute �ǽ��洢��ʽ��pytorch��ʽת��opencv��ʽ
            //            3.��Ϊ�ڴ���ǰ��cvmat�е�ֵ���˹�һ������������Ҫ*255�ָ���ͬʱ���ڲ���0-255��Χ�ڵ����ݣ���Ҫ������
            //            4.��Ϊcvmat�����ݸ�ʽ��8UC3������torch tensorҪ��ǰת����kU8
            //sequeeze trans tensor shape from 1*C*H*W to C*H*W
            //permute C*H*W to H*W*C
            out_tensor = out_tensor.squeeze(0).detach().permute({1, 2, 0}); //�ȿӣ������ᣬ����ǰ��ά����ɾ��
            cout<<"........3.........."<<endl;
            //see tip3��tip4
            out_tensor = out_tensor.mul(255).clamp(0, 255).to(torch::kU8);
            cout<<"........4.........."<<endl;
            out_tensor = out_tensor.to(torch::kCPU);
            cout<<"........5.........."<<endl;

            cv::Mat resultImg(IMAGE_SIZE, IMAGE_SIZE, CV_8UC1); //(����������������)
            cout<<"........6.........."<<endl;
            //copy the data from out_tensor to resultImg
            std::memcpy((void *) resultImg.data, out_tensor.data_ptr(), sizeof(torch::kU8) * out_tensor.numel());

            cout<<"��ʾ���ͼ"<<endl;
            t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
            cout << "�Ӷ�ȡͼƬ������������ cost time�� " <<  t << endl;

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
