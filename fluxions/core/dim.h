//
// Created by Mrinal Chaturvedi on 11/06/26.
//

#ifndef FLUXIONS_REPO_DIM_H
#define FLUXIONS_REPO_DIM_H
#include <iosfwd>

namespace fluxions {

    class Dim {
    private:
        unsigned int rows_;
        unsigned int cols_;
        unsigned int batch_;

    public:
        Dim();
        Dim(unsigned int rows, unsigned int cols);
        Dim(unsigned int rows, unsigned int cols, unsigned int batch);

        unsigned int rows() const;
        unsigned int cols() const;
        unsigned int batch_elems() const;

        unsigned int batch_size() const;
        unsigned int size() const;
        unsigned int ndims() const;

        Dim transpose() const;
        Dim single_batch() const;

        bool operator==(const Dim& other) const;
        bool operator!=(const Dim& other) const;
    };

    std::ostream& operator<<(std::ostream& os, const Dim& dim);

} // namespace fluxions

#endif // FLUXIONS_DIM_H