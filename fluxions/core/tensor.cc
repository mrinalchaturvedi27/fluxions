//
// Created by Mrinal Chaturvedi on 9/06/26.
//

#include "core/tensor.h"

#include <cassert>
#include <iostream>
#include <utility>

namespace fluxions {

Tensor::Tensor() = default;

Tensor::Tensor(Matrix data)
    : data_(std::move(data)) {}

Tensor::Tensor(int row, int col)
    : data_(row, col) {
    data_.setZero();
}

const Matrix& Tensor::data() const {
    return data_;
}

int Tensor::rows() const {
    return static_cast<int>(data_.rows());
}

int Tensor::cols() const {
    return static_cast<int>(data_.cols());
}

int Tensor::size() const {
    return static_cast<int>(data_.size());
}

std::string Tensor::dtype() const {
    return "float";
}

int Tensor::ndim() const {
    return 2;
}

void Tensor::shape() const {
    std::cout << '(' << data_.rows() << ',' << data_.cols() << ')' << '\n';
}

Tensor Tensor::operator+(const Tensor& a) const {
    assert(rows() == a.rows());
    assert(cols() == a.cols());
    return Tensor(data_ + a.data());
}

Tensor Tensor::operator-(const Tensor& a) const {
    assert(rows() == a.rows());
    assert(cols() == a.cols());
    return Tensor(data_ - a.data());
}

Tensor Tensor::operator*(const Tensor& a) const {
    assert(_validate_matmul_shapes(a));
    return Tensor(data_ * a.data());
}

Tensor Tensor::mul(const Tensor& a) const {
    assert(rows() == a.rows());
    assert(cols() == a.cols());
    return Tensor((data_.array() * a.data().array()).matrix());
}

Tensor Tensor::div(const Tensor& a) const {
    assert(rows() == a.rows());
    assert(cols() == a.cols());
    return Tensor((data_.array() / a.data().array()).matrix());
}

bool Tensor::_validate_matmul_shapes(const Tensor& a) const {
    return cols() == a.rows();
}

float Tensor::sum() const {
    return data_.sum();
}

float Tensor::mean() const {
    return data_.mean();
}

float Tensor::max() const {
    return data_.maxCoeff();
}

Tensor Tensor::reshape(int rows, int cols) const {
    assert(rows * cols == size());
    Eigen::Map<const Matrix> mapped(data_.data(), rows, cols);
    return Tensor(Matrix(mapped));
}

Tensor Tensor::transpose() const {
    return Tensor(data_.transpose());
}

} // namespace fluxions
