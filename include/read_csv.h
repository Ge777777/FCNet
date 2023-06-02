#pragma once

#include<vector>
#include<cstring>
#include<fstream>
#include<iostream>
#include<sstream>

static const int header = 1;

static const int size_in = 784;

static const int size_out = 10;

class IOData {
public:
    std::vector<double> input, output;

    IOData(int in, int out) : input(in), output(out) {};

    IOData(const IOData &sample) {
        this->input = sample.input;
        this->output = sample.output;
    }

    ~IOData() = default;
};

std::vector<IOData> read_csv(std::string path = "../dataset/mnist_train.csv");