#include <iostream>
#include "../include/Frame.h"
int main() {
    Frame frame;

    frame.set_IO(read_csv());
    frame.set_test(read_csv("../dataset/mnist_test.csv"));

    Relu_dense relu1(784,100);
    frame.add_layer(&relu1);

    Relu_dense relu2(100,50);
    frame.add_layer(&relu2);

    Softmax_dense softmax(50,10);
    frame.add_layer(&softmax);

    std::cout << "start training" << std::endl;
    for (int i = 0; i < 30000; i++) {
        if (i % 1000 == 0) {
            printf("sample: %d\n", i);
        }
        frame.sample = frame.IO_[i];
        frame.forward();
        frame.backward(rate1);
    }
    for (int i = 30000; i < 60000; i++) {
        if (i % 1000 == 0) {
            printf("sample: %d\n", i);
        }
        frame.sample = frame.IO_[i];
        frame.forward();
        frame.backward(rate2);

    }
    std::cout << "finished training" << std::endl;
    std::cout << "start testing" << std::endl;
    for (int i = 0; i < frame.test_.size(); i++) {
        frame.test_sample = frame.test_[i];
        frame.predict();
    }
    std::cout << "finished testing" << std::endl;
    std::cout << "accuracy: " << frame.get_accuracy() << std::endl;

    return 0;
}
