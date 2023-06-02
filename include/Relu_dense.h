#pragma once

#include "Dense.h"

class Frame;

class Relu_dense : public Dense {
public:
    Relu_dense() = delete;

    Relu_dense(int size_in, int neuron_count) ;

    ~Relu_dense() override = default ;

    void activate() override;

    //relu不会作为最后一层，不实现

    double deriv_fun(int i);

    //误差计算放外面的框架
    void delta_calc(bool isOutputLayer, const  Frame&frame);

private:


};



