#include "../include/Relu_dense.h"

Relu_dense::Relu_dense(int size_in, int neuron_count)
        : Dense(size_in, neuron_count)
        {}

void Relu_dense::activate() {
    linearTransform();
    for (int i = 0; i < neuron_count; i++) {
        h_[i] = z_[i] > 0 ? z_[i] : 0;
    }
}
//求出d(a)/d(z_)
double Relu_dense::deriv_fun(int i) {
    if (i >= neuron_count){
        std::cout << "不存在该神经元";
        exit(1);
    }
    if(z_[i] > 0 )
        return 1;
    else
        return 0;
}

void Relu_dense::delta_calc(bool isOutputLayer, const Frame& frame) {
    std::cout << "ReLU function cannot be used as the activation function "
                 "in the last layer because its delta cannot be computed.";
    exit(1);
}
