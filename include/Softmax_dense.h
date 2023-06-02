#pragma once
#include "Dense.h"
#include <cmath>
class Frame;

class Softmax_dense : public Dense {
public:
    Softmax_dense() = delete;

    Softmax_dense(int size_in, int neuron_count) ;

    ~Softmax_dense() override = default;

    void calcDenominator();

    void activate() override;

//    void fix_weight() override;

//    double deriv_fun(int k , int l);//不作为中间层，暂时不考虑二位还是一维

    void delta_calc(bool isOutputLayer,  const  Frame&frame);

private:

    double sum_exp;

    std::vector<std::vector<double>> deriv_fun_matrix;

};

