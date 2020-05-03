#include <iostream>

#include <opencv2/opencv.hpp>

#include <torch/script.h>
#include <torch/torch.h>
#include <torch/cuda.h>

using namespace torch;
using namespace nn;
using namespace std;
int main() {
    try {

        torch::jit::script::Module module = torch::jit::load("../attunet_half.pt");//读取半精度 的模型
        module.to(at::kCUDA);

        torch::Tensor inputdata = torch::ones({1, 3, 768, 768}).toType(torch::kFloat16).to(at::kCUDA);//转为 16 float 精度的 输入

        vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::jit::IValue(inputdata));

        at::Tensor output = module.forward(inputs).toTensor();
        std::cout << "output" <<std::endl;

    }catch(exception e) {
        cout<<e.what()<<endl;
    }

    return 0;
}
