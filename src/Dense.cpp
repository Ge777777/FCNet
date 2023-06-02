#include <utility>

#include "../include/Dense.h"

Dense::Dense(int size_in, int neuron_count)
        : x_(size_in),
          w_(neuron_count, std::vector<double>(size_in)),
          b_(neuron_count),
          z_(neuron_count),
          h_(neuron_count),
          delta_(1, std::vector<double>(neuron_count)) {
    this->size_in = size_in;
    this->neuron_count = neuron_count;
}

void Dense::linearTransform() {
    for (int i = 0; i < neuron_count; i++) {
        z_[i] = 0;
        for (int j = 0; j < size_in; j++) {
                z_[i] += x_[j] * w_[i][j];
        }
        z_[i] += b_[i];
    }
}
Matrix Dense::get_w() const {
    return this->w_;
}

std::vector<double> Dense::get_z(){
    return this->z_;
}

std::vector<double> Dense:: get_x(){
    return this->x_;
}

Matrix Dense::get_delta() const {
    return this->delta_;
}

std::vector<double> Dense::get_b() const {
    return this->b_;
}


std::vector<double> Dense::get_output() const{
    return this->h_;
}

int Dense::get_size_in() const {
    return this->size_in;
}

int Dense::get_neuron_count() const {
    return this->neuron_count;
}


void Dense::set_size_in(int size_in) {
    this->size_in = size_in;
    x_ = std::vector<double>(size_in);


}

void Dense::set_input(std::vector<double> x) {
    this->x_ = std::move(x);
}

void Dense::set_w(Matrix w){
    this->w_ = std::move(w);
}

void Dense::set_b(std::vector<double> b) {
    this->b_ = std::move(b);
}

void Dense::set_neuron_count(int neuron_count) {
    this->neuron_count = neuron_count;
    w_ = std::vector<std::vector<double>>(neuron_count, std::vector<double>(size_in));
    b_ = std::vector<double>(neuron_count);
    z_ = std::vector<double>(neuron_count);

}

void Dense::set_delta(Matrix delta) {
    this->delta_ = std::move(delta);
}