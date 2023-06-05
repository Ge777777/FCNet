#include <utility>

#include "../include/Frame.h"

Frame::Frame() : sample(0, 0), test_sample(0, 0) {}

void Frame::forward() {
    layers[0]->set_input(sample.input);
    int i;
    for (i = 0; i < layers.size() - 1; i++) {
        layers[i]->activate();
        layers[i + 1]->set_input(layers[i]->get_output());
    }
    layers[i]->activate();
    output = layers[i]->get_output();
}

void Frame::predict() {

    layers[0]->set_input(test_sample.input);
    int i;
    for (i = 0; i < layers.size() - 1; i++) {
        layers[i]->activate();
        layers[i + 1]->set_input(layers[i]->get_output());
    }
    layers[i]->activate();
    output = layers[i]->get_output();
    double max = 0;
    int index = 0;
    for (int i = 0; i < output.size(); i++) {
        index = output[i] > max ? i : index;
        max = output[i] > max ? output[i] : max;
    }
    for (int i = 0; i < output.size(); i++) {
        output[i] = 0;
    }
    output[index] = 1;


    if (output == test_sample.output) {
        correct_count++;
    }
}

void Frame::backward(const double rate) {
    //最后一层梯度
    layers[layers.size() - 1]->delta_calc(true, *this);
    for (int i = layers.size() - 2; i >= 0; i--) {
        layers[i]->set_delta(algebra::multiply(layers[i + 1]->get_delta(), layers[i + 1]->get_w()));
        // relu的导数就是1；
    }
    for (int i = layers.size() - 1; i >= 0; i--) {
        for (int j = 0; j < layers[i]->get_neuron_count(); j++) {
            for (int k = 0; k < layers[i]->get_size_in(); k++) {
                w[i][j][k] -= rate * layers[i]->get_delta()[0][j] * layers[i]->get_x()[k];
                //输出因子
            }
            b[i][j] -= rate1 * layers[i]->get_delta()[0][j];
        }
        layers[i]->set_w(w[i]);
        layers[i]->set_b(b[i]);
    }
}


void Frame::add_layer(Dense *layer) {
    layer->set_w(random(layer->get_neuron_count(), layer->get_size_in()));
    layer->set_b(random(layer->get_neuron_count()));
    if (layers.empty()) {
        layers.push_back(layer);
        layer->set_size_in(this->IO_[0].input.size());
    } else {
        int size = layers[layers.size() - 1]->get_neuron_count();
        layers.push_back(layer);
        layer->set_size_in(size);
    }
    this->w.push_back(layer->get_w());
    this->b.push_back(layer->get_b());

}


Frame::~Frame() {
    //没有实现深拷贝，不删除
//    for (auto i : layers){
//        delete i;
//    }
}

Matrix Frame::random(int n, int m) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(lower_bound, upper_bound);

    Matrix matrix(n, std::vector<double>(m));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            matrix[i][j] = dis(gen);
        }
    }
    return matrix;
}

std::vector<double> Frame::random(int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(lower_bound, upper_bound);


    std::vector<double> vec(n);
    for (int i = 0; i < n; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}


void Frame::set_IO(std::vector<IOData> IO) {
    this->IO_ = std::move(IO);
    if (!layers.empty()) {
        layers[0]->set_size_in(IO_[0].input.size());
        layers[0]->set_input(IO_[0].input);
    }
}

void Frame::set_test(std::vector<IOData> test) {
    this->test_ = std::move(test);
    total_count = test_.size();
    correct_count = 0;
}

double Frame::get_accuracy() const {
    return static_cast<double>(correct_count) / total_count;
}

std::vector<Dense *> Frame::get_layers() const {
    return layers;
}
