#include "../include/Softmax_dense.h"
#include "../include/Frame.h"
//前向声明不管用


Softmax_dense::Softmax_dense(int size_in, int neuron_count)
        : Dense(size_in, neuron_count) {
    sum_exp = 0;
}

void Softmax_dense::calcDenominator() {
    sum_exp = 0;
    for (int i = 0; i < neuron_count; i++) {
        sum_exp += exp(z_[i]);
    }
}

void Softmax_dense::activate() {
    linearTransform();
    calcDenominator();
    double max = 0;
    for (int i = 0; i < neuron_count; i++) {
        h_[i] = (exp(z_[i]) - max) / (sum_exp - max * neuron_count);
    }
}
//放在里面的问题在于既要frame里面的数据，又要知道后面一层的神经元个数
void Softmax_dense::delta_calc(bool isOutputLayer, const Frame& frame) {
    if (isOutputLayer) {
        for (int i = 0; i < neuron_count; i++) {
            delta_[0][i] = h_[i] - frame.sample.output[i];
        }
    }
    //softmax层不作为中间层,暂时不考虑作为delta传递的问题
    else{
        std::cout << "softmax层不作为中间层,暂时不考虑作为delta传递的问题";
        exit(1);
    }
}
