//
// Created by Mrinal Chaturvedi on 11/06/26.
//

#ifndef FLUXIONS_REPO_PARAMETER_H
#define FLUXIONS_REPO_PARAMETER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/dim.h"
#include "core/tensor.h"

namespace fluxions {

// Owns the real trainable data.
// survives across computation graphs
class ParameterStorage {
private:
    Tensor value_;
    Tensor grad_;

    std::string name_;
    bool trainable_ = true;
    bool has_grad_ = false;

public:
    ParameterStorage() = default;
    ParameterStorage(const Dim& shape, std::string name);

    const std::string& name() const;
    Dim shape() const;

    Tensor& value();
    const Tensor& value() const;

    Tensor& grad();
    const Tensor& grad() const;

    bool trainable() const;
    void set_trainable(bool trainable);

    int numel() const;
    bool has_grad() const;

    void zero_value();
    void zero_grad();
    void accumulate_grad(const Tensor& g);
};


// Lightweight handle, Copying this should NOT copy the actual parameter values.
class Parameter {
private:
    std::shared_ptr<ParameterStorage> storage_;
public:
    Parameter() = default;
    explicit Parameter(std::shared_ptr<ParameterStorage> storage);

    bool valid() const;

    const std::string& name() const;
    Dim shape() const;

    Tensor& value();
    const Tensor& value() const;

    Tensor& grad();
    const Tensor& grad() const;

    bool trainable() const;
    void set_trainable(bool trainable);

    int numel() const;
    void zero_grad();
    void accumulate_grad(const Tensor& g);

};


class ParameterCollection {
private:
    std::vector<std::shared_ptr<ParameterStorage>> params_;
public:
    ParameterCollection() = default;

    Parameter add_parameter(const Dim& shape, const std::string& name = "");
    const std::vector<std::shared_ptr<ParameterStorage>>& storages() const;

    std::vector<Parameter> parameters() const;
    void zero_grad();

    int parameter_count() const;
    int size() const;


};

} // namespace fluxions

#endif // FLUXIONS_REPO_PARAMETER_H
