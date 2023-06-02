#include "../include/read_csv.h"

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