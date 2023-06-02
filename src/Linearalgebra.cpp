#include "../include/linearalgebra.h"
using Matrix = std::vector<std::vector<double>>;

namespace algebra {
    using Matrix = std::vector<std::vector<double>>;
    Matrix zeros(size_t n, size_t m) {
        Matrix zeros(n, std::vector<double>(m));
        return zeros;
    }

    Matrix ones(size_t n, size_t m) {
        Matrix ones(n, std::vector<double>(m, 1));
        return ones;
    }

    Matrix Unit(size_t n) {
        Matrix unit(n, std::vector<double>(n));
        for (int i = 0; i < unit.size(); i++) {
            unit[i][i] = 1;
        }
        return unit;
    }

    Matrix random(size_t n, size_t m, double min, double max) {
        if (max > min) {
            std::random_device rd;//创建随机数引擎
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(min, max);
            Matrix random(n, std::vector<double>(m));
            for (int i = 0; i < n; i++)
                for (int j = 0; j < m; j++)
                    random[i][j] = dis(gen);
            return random;
        } else {
            throw std::logic_error("min >= max");

        }
    }

    void show(const Matrix &matrix) {
        for (int i = 0; i < matrix.size(); i++) {
            std::cout << "[";
            for (int j = 0; j < matrix[0].size(); j++) {
                std::cout <<std::fixed<< std::left << std::setw(10) << std::setprecision(3) << matrix[i][j];
            }
            std::cout << "]" << std::endl;
        }
    }

    Matrix multiply(const Matrix &matrix, double c) {
        Matrix matrix_answer = matrix;
        for (int i = 0; i < matrix.size(); i++)

            for (int j = 0; j < matrix[i].size(); j++)
                matrix_answer[i][j] *= c;
        return matrix_answer;
    }

    Matrix multiply(const Matrix &matrix1, const Matrix &matrix2) {
        //矩阵可成的前提是matrix1的列 == matrix2的行数

        if (matrix1.empty() || matrix2.empty()) {
            std::cout << "Matrix is empty" << std::endl;
            return zeros(0, 0);
        }
        int m1 = matrix1.size(), n1 = matrix1[0].size();
        int m2 = matrix2.size(), n2 = matrix2[0].size();
        if (n1 == m2) {
            int m = m1, n = n2;
            Matrix matrix_answer(m, std::vector<double>(n));
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    for (int k = 0; k < m2; k++) {
                        matrix_answer[i][j] += matrix1[i][k] * matrix2[k][j];
                    }
                }
            }
            return matrix_answer;
        } else {
            throw std::logic_error("Matrix type does not match, not multiplyable");

        }

    }

    Matrix sum(const Matrix &matrix, double c) {
        Matrix matrix_answer = matrix;
        for (int i = 0; i < matrix.size(); i++)

            for (int j = 0; j < matrix[i].size(); j++)
                matrix_answer[i][j] += c;
        return matrix_answer;
    }

    Matrix sum(const Matrix &matrix1, const Matrix &matrix2) {
        if (matrix1.empty() && matrix2.empty()) {
            return zeros(0, 0);
        } else if (matrix1.empty() && !matrix2.empty() || !matrix1.empty() && matrix2.empty()) {
            throw std::logic_error("Matrix type does not match");
        }
        int m1 = matrix1.size(), n1 = matrix1[0].size();
        int m2 = matrix2.size(), n2 = matrix2[0].size();
        if (m1 == m2 && n1 == n2) {
            Matrix matrix_answer(m1, std::vector<double>(n1));
            for (int i = 0; i < m1; i++) {
                for (int j = 0; j < n1; j++) {
                    matrix_answer[i][j] = matrix1[i][j] + matrix2[i][j];
                }
            }
            return matrix_answer;
        } else {
            throw std::logic_error("Matrix type does not match");
        }
    }

    Matrix transpose(const Matrix &matrix) {
        if (matrix.empty())
            return zeros(0, 0);
        else {
            Matrix transpose(matrix[0].size(), std::vector<double>(matrix.size()));
            for (int i = 0; i < transpose.size(); i++)
                for (int j = 0; j < transpose[i].size(); j++) {
                    transpose[i][j] = matrix[j][i];
                }
            return transpose;
        }
    }

    Matrix minor(const Matrix &matrix, size_t n, size_t m) {
        if (n < matrix.size() && m < matrix[0].size()) {
            Matrix minor = matrix;
            minor.erase(minor.begin() + n);
            for (auto &i: minor)
                i.erase(i.begin() + m);
            return minor;
        } else {
            std::cout << "Out of range" << std::endl;
            return zeros(0, 0);
        }
    }

    double determinant(const Matrix &matrix) {
        //判断是否为空矩阵
        if (matrix.size() == 0 || matrix[0].size() == 0)
            return 1;
            //判断是否是方阵
        else if (matrix.size() != matrix[0].size()) {
            throw std::logic_error("Matrix is not square");
        } else {
            //一阶矩阵返回
            if (matrix.size() == 1) {
                return matrix[0][0];
            } else  {
                double answer = 0;
                for (int i = 0; i < matrix.size(); i++)
                    answer += pow(-1, i + 1 + 1) * matrix[i][0] * determinant(minor(matrix, i, 0));
                return answer;
            }
        }
    }

    //交换两行
    Matrix ero_swap(const Matrix &matrix, size_t r1, size_t r2) {
        if (r1 < matrix.size() && r2 < matrix.size() && r1 >= 0 && r2 >= 0) {
            Matrix answer = matrix;
            std::swap(answer[r1], answer[r2]);
            return answer;
        } else {

            throw std::logic_error("Out of range");
        }
    }

    //倍增
    Matrix ero_multiply(const Matrix &matrix, size_t r, double c) {
        if (r < matrix.size()) {
            Matrix answer = matrix;
            for (int i = 0; i < matrix[i].size(); i++) {
                answer[r][i] *= c;
            }
            return answer;
        } else {
            std::cout << "Out of range" << std::endl;
            return zeros(0, 0);
        }
    }

    //倍加
    Matrix ero_sum(const Matrix &matrix, size_t r1, double c, size_t r2) {
        Matrix answer = matrix;
        if (r1 < matrix.size() && r2 < matrix.size() && r1 >= 0 && r2 >= 0) {
            for (int i = 0; i < matrix[0].size(); i++) {
                answer[r2][i] += answer[r1][i] * c;
            }
        }
        return answer;
    }

    //上三角
    Matrix upper_triangular(const Matrix &matrix) {
        if (0 == matrix.size()) {
            return zeros(0, 0);
        } else if (matrix.size() == matrix[0].size()) {
            Matrix answer = matrix;
            for (int i = 0; i < matrix[0].size() - 1; i++) {
                if(answer[i][i] == 0)
                {
                    for (int j = i + 1; j < matrix.size(); j++) {
                        if (answer[j][i] != 0) {
                            answer = ero_swap(answer, i, j);
                            break;
                        }
                    }
                }
                for (int j = i + 1; j < matrix.size(); j++) {
                    answer = ero_sum(answer, i, -answer[j][i] / answer[i][i], j);
                }
            }
            return answer;
        } else {
            throw std::logic_error("Matrix is not square");
        }
    }

    //矩阵求逆
    Matrix inverse(const Matrix &matrix) {
        //行列式存在且不等于0,并且不是空矩阵
        if (matrix.size() == 0 || matrix[0].size() == 0) {
            std::cout << "Empty matrix" << std::endl;
            return zeros(0, 0);
        } else  if(determinant(matrix) == 0) {
            throw std::logic_error("non_singular_matrix");
        } else {
            //求出上三角矩阵
            Matrix unit_to_inverse = Unit(matrix.size());
            Matrix matrix_to_unit = matrix;
            for (int i = 0; i < matrix[0].size() - 1; i++) {
                if(matrix_to_unit[i][i] == 0)
                {
                    for (int j = i + 1; j < matrix.size(); j++) {
                        if (matrix_to_unit[j][i] != 0) {
                            matrix_to_unit = ero_swap(matrix_to_unit, i, j);
                            unit_to_inverse = ero_swap(unit_to_inverse, i, j);
                            break;
                        }
                    }
                }
                for (int j = i + 1; j < matrix.size(); j++) {
                    unit_to_inverse = ero_sum(unit_to_inverse, i, -matrix_to_unit[j][i] / matrix_to_unit[i][i], j);
                    matrix_to_unit = ero_sum(matrix_to_unit, i, -matrix_to_unit[j][i] / matrix_to_unit[i][i], j);
                }
            }
            for (int i = 1; i < matrix_to_unit.size(); i++) {
                if(matrix_to_unit[i][i] == 0)
                {
                    for (int j = i + 1; j < matrix.size(); j++) {
                        if (matrix_to_unit[j][i] != 0) {
                            matrix_to_unit = ero_swap(matrix_to_unit, i, j);
                            unit_to_inverse = ero_swap(unit_to_inverse, i, j);
                            break;
                        }
                    }
                }
                for (int j = 0; j < i; j++) {
                    unit_to_inverse = ero_sum(unit_to_inverse, i, -matrix_to_unit[j][i] / matrix_to_unit[i][i], j);
                    matrix_to_unit = ero_sum(matrix_to_unit, i, -matrix_to_unit[j][i] / matrix_to_unit[i][i], j);
                }
            }
            for (int i = 0; i < matrix_to_unit.size(); i++) {
                for(int j = 0; j < matrix_to_unit[0].size(); j++){
                    unit_to_inverse[i][j] /= matrix_to_unit[i][i];
                    matrix_to_unit[i][j] /= matrix_to_unit[i][i];
                }
            }
            return unit_to_inverse;
        }
    }
    //矩阵结合
    Matrix concatenate(const Matrix &matrix1, const Matrix &matrix2, int axis) {
        if (!axis && matrix1[0].size() == matrix2[0].size()) {
            Matrix concatenate(matrix1.size() + matrix2.size(), std::vector<double>(matrix2[0].size()));
            for (int i = 0; i < matrix1.size() + matrix2.size(); i++) {
                concatenate[i] = i < matrix1.size() ? matrix1[i] : matrix2[i - matrix1.size()];
            }
            return concatenate;
        } else if (!axis && matrix1[0].size() != matrix2[0].size()) {
            throw std::logic_error("Matrix size not match");
        } else if (axis && matrix1.size() == matrix2.size()) {
            Matrix concatenate(matrix1.size(), std::vector<double>(matrix1[0].size() + matrix2[0].size()));
            for (int j = 0; j < matrix1[0].size() + matrix2[0].size(); j++)
                for (int i = 0; i < matrix1.size(); i++) {
                    concatenate[i][j] = j < matrix1[0].size() ? matrix1[i][j] : matrix2[i][j - matrix1[0].size()];
                }
            return concatenate;
        } else {
            throw std::logic_error("Matrix size not match");
        }
    }
}