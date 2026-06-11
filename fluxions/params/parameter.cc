//
// Created by Mrinal Chaturvedi on 11/06/26.
//

#include "params/parameter.h"

#include <cassert>
#include <utility>

namespace fluxions {

ParameterStorage::ParameterStorage(const Dim& shape, std::string name)
    : value_(static_cast<int>(shape.rows()), static_cast<int>(shape.cols())),
      grad_(static_cast<int>(shape.rows()), static_cast<int>(shape.cols())),
      name_(std::move(name)) {
   
    zero_grad();
}

const std::string& ParameterStorage::name() const {
    return name_;
}

Dim ParameterStorage::shape() const {
    return Dim(static_cast<unsigned int>(value_.rows()),
               static_cast<unsigned int>(value_.cols()));
}

Tensor& ParameterStorage::value() {
    return value_;
}

const Tensor& ParameterStorage::value() const {
    return value_;
}

Tensor& ParameterStorage::grad() {
    return grad_;
}

const Tensor& ParameterStorage::grad() const {
    return grad_;
}

bool ParameterStorage::trainable() const {
    return trainable_;
}

void ParameterStorage::set_trainable(bool trainable) {
    trainable_ = trainable;
}

int ParameterStorage::numel() const {
    return static_cast<int>(value_.size());
}

bool ParameterStorage::has_grad() const {
    return has_grad_;
}

void ParameterStorage::zero_value() {
    value_ = Tensor(value_.rows(), value_.cols());
}

void ParameterStorage::zero_grad() {
    grad_ = Tensor(value_.rows(), value_.cols());
    has_grad_ = false;
}

void ParameterStorage::accumulate_grad(const Tensor& g) {
    
    // - check same shape
    // - grad_ += g
    // - mark has_grad_
    assert(g.rows() == grad_.rows());
    assert(g.cols() == grad_.cols());

    grad_ = grad_ + g;
    has_grad_ = true;
}


Parameter::Parameter(std::shared_ptr<ParameterStorage> storage)
    : storage_(std::move(storage)) {}

bool Parameter::valid() const {
    return static_cast<bool>(storage_);
}

const std::string& Parameter::name() const {
    assert(valid());
    return storage_->name();
}

Dim Parameter::shape() const {
    assert(valid());
    return storage_->shape();
}

Tensor& Parameter::value() {
    assert(valid());
    return storage_->value();
}

const Tensor& Parameter::value() const {
    assert(valid());
    return storage_->value();
}

Tensor& Parameter::grad() {
    assert(valid());
    return storage_->grad();
}

const Tensor& Parameter::grad() const {
    assert(valid());
    return storage_->grad();
}

bool Parameter::trainable() const {
    assert(valid());
    return storage_->trainable();
}

void Parameter::set_trainable(bool trainable) {
    assert(valid());
    storage_->set_trainable(trainable);
}

int Parameter::numel() const {
    assert(valid());
    return storage_->numel();
}

void Parameter::zero_grad() {
    assert(valid());
    storage_->zero_grad();
}

void Parameter::accumulate_grad(const Tensor& g) {
    assert(valid());
    storage_->accumulate_grad(g);
}

// Owns all parameter storages for a model.
Parameter ParameterCollection::add_parameter(const Dim& shape,
                                             const std::string& name) {
    // TODO:
    // - create shared ParameterStorage
    // - store it in params_
    // - return Parameter handle
    auto storage = std::make_shared<ParameterStorage>(shape, name);
    params_.push_back(storage);
    return Parameter(storage);
}

const std::vector<std::shared_ptr<ParameterStorage>>&
ParameterCollection::storages() const {
    return params_;
}

std::vector<Parameter> ParameterCollection::parameters() const {
    std::vector<Parameter> result;
    result.reserve(params_.size());

    for (const auto& storage : params_) {
        result.emplace_back(storage);
    }

    return result;
}

void ParameterCollection::zero_grad() {
    for (const auto& storage : params_) {
        storage->zero_grad();
    }
}

int ParameterCollection::parameter_count() const {
    int total = 0;

    for (const auto& storage : params_) {
        if (storage->trainable()) {
            total += storage->numel();
        }
    }

    return total;
}

int ParameterCollection::size() const {
    return static_cast<int>(params_.size());
}

} // namespace fluxions
