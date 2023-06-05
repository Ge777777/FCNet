#include "Relu_dense.h"
#include "Softmax_dense.h"
#include "Linearalgebra.h"
#include <random>
#include <ctime>

const std::string Relu = "Relu";

const std::string Softmax = "Softmax";

const double rate1 = 0.001;

const double rate2 = 0.0005;

const double upper_bound = 0.1;

const double lower_bound = -0.1;

class Frame {

public:

    Frame();

    ~Frame();

    void forward();

    void predict();

    void backward(double rate);

    void add_layer(Dense *layer);

    void set_IO(std::vector<IOData> IO);

    void set_test(std::vector<IOData> test);

    double get_accuracy() const;

    //给layers变量设置输出端口
    std::vector<Dense *> get_layers() const;

    static Matrix random(int n, int m);

    static std::vector<double> random(int i);

    std::vector<IOData> IO_;
    IOData sample;//表示本次训练所用到的训练集

    std::vector<IOData> test_;//表示本次训练所用到的测试集
    IOData test_sample;//表示本次训练所用到的测试集


private:

    friend void Softmax_dense::delta_calc(bool isOutputLayer, const Frame &frame);
    //需要访问该类的私有成员 sample，所以要声明为友元

    int correct_count = 0;

    int total_count = 0;

    double accuracy = 0;

    std::vector<Dense *> layers;

    std::vector<double> output;

    std::vector<Matrix> w;

    Matrix b;


};


