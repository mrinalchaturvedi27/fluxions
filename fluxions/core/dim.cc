//
// Created by Mrinal Chaturvedi on 11/06/26.
//


#include "core/dim.h"
#include <ostream>


namespace fluxions {
    Dim::Dim()
        : rows_(0), cols_(0), batch_(1) {}

    Dim::Dim(unsigned int rows, unsigned int cols)
        : rows_(rows), cols_(cols), batch_(1) {}

    Dim::Dim(unsigned int rows, unsigned int cols, unsigned int batch)
        : rows_(rows), cols_(cols), batch_(batch) {}

    unsigned int Dim::rows() const {
        return rows_;
    }

    unsigned int Dim::cols() const {
        return cols_;
    }

    unsigned int Dim::batch_elems() const {
        return batch_;
    }

    unsigned int Dim::batch_size() const {
        return rows_ * cols_;
    }

    unsigned int Dim::size() const {
        return batch_size() * batch_elems();
    }

    unsigned int Dim::ndims() const {
        return 2;
    }

    Dim Dim::transpose() const {
        return Dim(cols_, rows_, batch_);
    }

    Dim Dim::single_batch() const {
        return Dim(rows_, cols_);
    }

    bool Dim::operator==(const Dim& other) const {
        return rows_ == other.rows()
            && cols_ == other.cols()
            && batch_ == other.batch_elems();
    }

    bool Dim::operator!=(const Dim& other) const {
        return !(*this == other);
    }

    std::ostream& operator<<(std::ostream& os, const Dim& dim) {
        os << '{' << dim.rows() << ',' << dim.cols();

        if (dim.batch_elems() != 1) {
            os << 'X' << dim.batch_elems();
        }

        os << '}';
        return os;
    }

} // namespace fluxions