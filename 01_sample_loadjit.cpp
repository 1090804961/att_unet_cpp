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
//      加载模型
        torch::jit::script::Module module = torch::jit::load("../attunet.pt");
        module.to(at::kCUDA);

//        torch::Tensor inputdata = torch::ones({1, 3, 768, 768}).cuda();
        torch::Tensor inputdata = torch::ones({1, 3, 768, 768}).to(at::kCUDA);

//      转换 tensor 类型为torch::jit::IValue  并 添加入 vector 中
        vector<torch::jit::IValue> inputs;
        inputs.push_back(torch::jit::IValue(inputdata));
//      前向运算
        at::Tensor output = module.forward(inputs).toTensor();
        std::cout << "output" <<std::endl;

    }catch(exception e) {
        cout<<e.what()<<endl;
    }

    return 0;
}
