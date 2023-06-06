#                         FCNet

## 项目描述

实现了一个全连接层训练框架，它可以根据需求去实现自己的目的。

在测试中利用全连接层实现了数字识别，测试集选用的mnist，准确率九十。

## 设计思路

在全连接层中，神经网络由一层层的神经元组成，虽然每一层的激活函数不尽相同，但是每一层都有很多相似之处，所以基类 Dense 就是实现神经层的相同部分，而Relu_dense 和 Softmax 则是由基类 Dense 根据不同的激活函数派生出来的派生类。

类Frame负责把构建一个神经网络、前向传播和反向传播，以及预测等操作封装起来，这样不仅更安全，而且更简单，更易操作。

## 类和函数的设计





### Dense类

~~~c++
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
~~~

#### 变量

实现了是全连接层的相同部分，变量w，x，b，z，h分别表示权重，输入，偏置，线性和，输出；delta是每层的误差，是在神经网络中一直传播的变量。

#### 函数

##### 构造函数和析构函数

对于构造函数

首先，不允许调用默认的构造函数，其次，把实现的构造函数放在protected里面，不允许外部调用，允许派生类中的构造函数显式的调用。

对于析构函数

声明为虚析构函数，由于并没有使用动态分配，所以采用默认的析构函数

##### get 和 set 函数

为保证安全，Dense类里面的函数都不允许被访问，所以设置了两个端口，一个是get，一个是set。

##### linearTransform函数

计算 z = wx + b

##### 虚函数部分

对于activate()

作为正向传播和反向传播的正要部分，由本层的输入计算本层的输出，具体实现于激活函数有关

对于delta_calc（）

只有作为最后一层输出层的时候才会被调用，且与激活函数和损失函数有关



### Relu_dense 类



~~~c++
class Relu_dense : public Dense {
public:
    Relu_dense() = delete;

    Relu_dense(int size_in, int neuron_count) ;

    ~Relu_dense() override = default ;

    void activate() override;

    double deriv_fun(int i);

    void delta_calc(bool isOutputLayer, const  Frame&frame);

private:


};
~~~



#### 变量

所有的变量都在dense声明过了

#### 函数

##### 构造函数和析构函数

对于构造函数

~~~c++
Relu_dense::Relu_dense(int size_in, int neuron_count)
        : Dense(size_in, neuron_count)
        {}
~~~



这里只需要显示的调用基类的构造函数

对于析构函数

该类没有变量，跟没有使用动态分配，只需要默认的析构函数就好了

##### 虚函数的覆写

- activate()

​	实现Relu函数即 



​                                    $$ f(x) = \begin{cases}    x, & \text{if } x \geq 0 \\    0, & \text{otherwise} \end{cases} $$



-  deriv_fun(int i)

  计算函数的导数即

  

  ​                              $$  f(x) = \begin{cases}  1, & \text{if } x \geq 0 \\ 0, & \text{outherwise} \end{cases}$$











- void delta_calc(bool isOutputLayer, const  Frame&frame);

  

\[![image-20230605150353259](C:\Users\小邰的77\AppData\Roaming\Typora\typora-user-images\image-20230605150353259.png)]

该函数是实现本层作为输出层时，求$$ \dfrac {\alpha L \theta} {\alpha w}$$

ReLu函数作为最后一层并没有太大的意义，所以这里并不具体实现其功能，而是采用斜面的方式

~~~c++
void Relu_dense::delta_calc(bool isOutputLayer, const Frame& frame) {
    std::cout << "ReLU function cannot be used as the activation function "
                 "in the last layer because its delta cannot be computed.";
    exit(1);
}
~~~



### Softmax_dense类

~~~c++
class Frame;

class Softmax_dense : public Dense {
public:
    Softmax_dense() = delete;

    Softmax_dense(int size_in, int neuron_count) ;

    ~Softmax_dense() override = default;

    void activate() override;

//    double deriv_fun(int k , int l);//不作为中间层，暂时不考虑二位还是一维

    void delta_calc(bool isOutputLayer,  const  Frame&frame);

private:

    double sum_exp;
};
~~~

#### 变量

sum_exp表示softmax函数的分母

#### 函数

##### 构造函数和析构函数

这里和Relu类的基本一致

##### 虚函数 

- activate() 

  实现函数的正向传播，公式如下

  

  

  ​                    $$ f(z_i) = \dfrac {e^{z_i}} {\Sigma e^{z_l}}$$    l = 1.2.3.......

  

- delta_calc(bool isOutputLayer,  const  Frame&frame)

  计算最后一层的梯度，公式如下

  

  ​              $$ \dfrac {\alpha L \theta} {w_i} = h_i - 1\ or\ 0$$        1和0区别于onehot编码对应的值

  其中这个onehot编码存储在Frame类里面,所以需要声明为Frame的友元函数



- deriv_fun(int k , int l)

  softmax同样不常用于隐含层。

### Frame类

~~~ c++
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
~~~

#### 变量

correct_count 和 total_count 和 accuracy来统计准确度



output来统计一次正向传播后的输出结果,采用onehot编码格式



layers来存储神经网络的层



Frame的w和b比Dense里面的维度高一维,它的目的是存储梯度下降后的w

和b(运用梯度下降算法,需要把整个网络遍历一遍后,再去更新值)



#### 函数

##### 构造函数和析构函数

对于构造函数

~~~c++
Frame::Frame() : sample(0, 0), test_sample(0, 0) {}
~~~



不需要形参,但是samplej和test_sample都是类IOData的实例,所以需要在初始化列表时就地构造.



对于析构函数

虽然用到了Dense指针,但是这里既没有深拷贝也没有动态分配,所以析构函数也不需要实现,为空

##### set和get函数

设置端口

##### 操作函数

forward()和backwrd()分别实现正向传播和反向传播

~~~c++
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
~~~



~~~c++
void Frame::backward(const double rate) {
    //最后一层求梯度
    layers[layers.size() - 1]->delta_calc(true, *this);
    for (int i = layers.size() - 2; i >= 0; i--) {
        layers[i]->set_delta(algebra::multiply(layers[i + 1]->get_delta(), layers[i + 1]->get_w()));
        //relu的导数就是1
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
~~~



predict()



拿训练好的模型的进行预测,并记录答案的个数

~~~c++
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
~~~





对于 add_layers函数

形参是类的实例的指针,在push_back元素之外,把前后层的大小连接适配起来

### Read_csv文件

实现了读取csv文件,把mnist的数据读取到IOData里面

~~~ c++
std::vector<IOData> read_csv(std::string path) {
    std::vector<IOData> dataset;
    IOData row(size_in, size_out);
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "File not found!" << std::endl;
        return dataset;
    } else {
        std::cout << "Success" << std::endl;
    }
    std::string line;
    for (int i = 0; i < header; i++) {
        std::getline(file, line);
    }
    while (std::getline(file, line)) {
        std::string cell;
        std::stringstream ss(line);
        getline(ss, cell, ',');//每行的第一个数据为 outputVector
        for (int i = 0; i < 10; i++) {
            row.output[i] = 0;
        }
        row.output[std::stoi(cell)] = 1;

        for (int i = 0; i < 28 * 28 - 1; ++i) {
            std::getline(ss, cell, ',');
            row.input[i] = std::stod(cell) / 255;
        }
        std::getline(ss, cell);
        row.input[28 * 28 - 1] = std::stod(cell) / 255;
        dataset.push_back(row);
        line.clear();
    }
    file.close();
    std::cout << "Done" << std::endl;
    return dataset;
}
~~~

读取的过程并不复杂



## RGBtoGrayConverter

该文件主要是调用了opencv的库，实现了把一个三色图片转成28*28的灰色图片，将数据转化为可以输入的数据

~~~c++
class RGBtoGrayConverter
 {
 public:
     RGBtoGrayConverter(std::string path);
     cv::Mat get_img();
     cv::Mat get_gray_img();
     void show_img();
     void show_gray_img();
     void save_gray_img(std::string path);
     void set_img(cv::Mat img);
     void set_path(std::string path);
     void set_lable(int lable);
     std::string get_path();
     std::vector<IOData> load_data();
 private:
     std::string path;
     cv::Mat img;
     cv::Mat gray_img;
     IOData data;

 };

~~~

这里提供了一些端口，可以查看中间过程，以及对private变量的赋值，函数的具体实现都很简单，这里不再一一赘述。

## 实现过程

- 采用MinGW编译器
- 使用stl库和opencv的库
- 主要算法就是梯度下降算法

## 功能展示

minst集预测的准确度大概在90

"C:\Users\小邰的77\Desktop\Code\DAYI 2\cnn’\cmake-build-release\cnn.exe"
 accuracy: 0.9075

进程已结束,退出代码0



![image-20230605210307360](C:\Users\小邰的77\AppData\Roaming\Typora\typora-user-images\image-20230605210307360.png)



## 总结

### 优点

- 良好的封装和抽象。该类封装了表示一个神经网络层的基本属性和功能,但比较抽象,具体的激活函数和参数更新函数留给子类实现。这增加了程序的灵活性和扩展性。
- 项目实现了个自己的训练框架，可以根据需求构造神经网络，满足不同的任务需求，灵活多变。
-  遵循DRY原则。Dense类遵循了不重复自己(Don't Repeat Yourself)的原则,提取出多个层的共性功能,避免在不同的层中写重复的代码。这使得代码更加简洁高效。
- 高内聚低耦合。Dense类的各个函数功能相对独立,但又紧密相关,符合高内聚的原则。而与具体的激活层子类耦合度较低,只通过虚函数调用,符合低耦合原则。这使程序结构清晰,容易维护和理解。
- 多态的使用,重点函数采用虚函数实现。如activate()、delta_calc()等函数采用虚函数实现,这样在子类的实现中可以按需重写,实现不同的功能。这是实现抽象类的常用手段。
- 输入参数采用引用传递。如delta_calc()的frame参数采用const Frame &形式。因为Frame可能是一个较大对象,如果按值传递,效率会较低。采用引用传递可以避免传递较大对象的开销。
- 封装和信息隐藏。隐藏实现细节,提供统一接口。私有属性和受保护构造函数。size_in、neuron_count、w_、b_、z_、h_和delta_这些属性采用私有权限,不允许外部直接访问。构造函数也采用protected权限,只允许子类调用。这可以隐藏该类的实现细节,对外提供统一的接口,符合良好的封装习惯。

### 不足

- 运行速度过慢,矩阵乘法的实现还需要优化,可以调用Eigen库,运行成本过高
- 准确度还需要提升,可以从下面几个方面进行考虑
  1. 训练数据只有6万,训练远远不够
  2. 目前rate学习率采用的是前三万数据大步幅 rate = 0.001 后三万数据采用0.0005,更好的办法是采用动态的学习率
  3. 采取更好的梯度下降算法
