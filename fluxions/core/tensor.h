#ifndef FLUXIONS_REPO_TENSOR_H
#define FLUXIONS_REPO_TENSOR_H

#include <Eigen/Dense>
#include <string>

namespace fluxions {

using Scalar = float;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using RowVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

class Tensor {
private:
    Matrix data_;

public:
    Tensor();
    explicit Tensor(Matrix data);
    Tensor(int rows, int cols);

    const Matrix& data() const;

    int rows() const;
    int cols() const;
    int size() const;
    std::string dtype() const;
    int ndim() const;
    void shape() const;

    Tensor operator+(const Tensor& a) const;
    Tensor operator-(const Tensor& a) const;
    Tensor operator*(const Tensor& a) const;
    Tensor mul(const Tensor& a) const;
    Tensor div(const Tensor& a) const;

    bool _validate_matmul_shapes(const Tensor& a) const;
    float sum() const;
    float mean() const;
    float max() const;
    Tensor reshape(int r, int c) const;
    Tensor transpose() const;
};

} // namespace fluxions

#endif // FLUXIONS_REPO_TENSOR_H
