#pragma once

#include "../include/read_csv.h"
#include "../include/Linearalgebra.h"

class Frame;

class Dense {
public:
    Dense() = delete;

    virtual ~Dense() = default;

    int get_size_in() const;

    int get_neuron_count() const;

    Matrix get_w() const;

    std::vector<double> get_z();

    std::vector<double> get_x();

    Matrix get_delta() const;

    std::vector<double> get_b() const;

    std::vector<double> get_output() const;

    void set_size_in(int size_in);

    void set_neuron_count(int neuron_count);

    void set_input(std::vector<double> x);

    void set_w(Matrix x);

    void set_b(std::vector<double> b);

    void set_delta(Matrix delta);

    void linearTransform();

    virtual void activate() = 0;

//    virtual void fix_weight() = 0;
//    virtual void fix_bias() = 0;

//    virtual double deriv_fun(int ) = 0;

    virtual void delta_calc(bool isOutputLayer, const Frame &frame) = 0;

protected:
    //并不希望用户直接调用构造函数，所以将其设置为protected
    Dense(int size_in, int neuron_count);

    int size_in;

    int neuron_count;

    std::vector<double> x_;

    Matrix w_;

    std::vector<double> b_;

    std::vector<double> z_;

    std::vector<double> h_;

    Matrix delta_;


private:


};


