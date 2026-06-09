# Eigen Syntax Reference for Fluxions

This file is a practical Eigen reference for building a CPU-first neural-network
library with a DyNet-style computation graph. It focuses on syntax you are
likely to use for tensors, parameters, forward operations, gradients, and small
linear algebra utilities.

Official docs used while writing this:

- Eigen 5.0 topics index: <https://libeigen.gitlab.io/eigen/docs-5.0/topics.html>
- Eigen quick reference: <https://libeigen.gitlab.io/eigen/docs-nightly/group__QuickRefPage.html>
- Matrix class tutorial: <https://libeigen.gitlab.io/eigen/docs-nightly/group__TutorialMatrixClass.html>
- Matrix arithmetic tutorial: <https://libeigen.gitlab.io/eigen/docs-5.0/group__TutorialMatrixArithmetic.html>
- Array/coefficient-wise operations: <https://libeigen.gitlab.io/eigen/docs-nightly/group__TutorialArrayClass.html>
- Block operations: <https://libeigen.gitlab.io/eigen/docs-nightly/group__TutorialBlockOperations.html>
- Slicing and indexing: <https://libeigen.gitlab.io/eigen/docs-5.0/group__TutorialSlicingIndexing.html>
- Reductions, visitors, broadcasting: <https://libeigen.gitlab.io/eigen/docs-nightly/group__TutorialReductionsVisitorsBroadcasting.html>
- Reshape: <https://libeigen.gitlab.io/eigen/docs-5.0/group__TutorialReshape.html>
- STL iterators and algorithms: <https://libeigen.gitlab.io/eigen/docs-5.0/group__TutorialSTL.html>
- `Ref` class: <https://libeigen.gitlab.io/eigen/docs-nightly/classEigen_1_1Ref.html>
- Lazy evaluation and aliasing: <https://libeigen.gitlab.io/eigen/docs-nightly/TopicLazyEvaluation.html>
- Aliasing details: <https://libeigen.gitlab.io/eigen/docs-nightly/group__TopicAliasing.html>
- Storage orders: <https://libeigen.gitlab.io/eigen/docs-3.0/TopicStorageOrders.html>
- Coefficient-wise math catalog: <https://libeigen.gitlab.io/eigen/docs-5.0/group__CoeffwiseMathFunctions.html>
- Structures with Eigen members: <https://libeigen.gitlab.io/eigen/docs-nightly/group__TopicStructHavingEigenMembers.html>
- STL containers with Eigen: <https://libeigen.gitlab.io/eigen/docs-nightly/group__TopicStlContainers.html>
- Passing Eigen objects by value: <https://libeigen.gitlab.io/eigen/docs-3.3/group__TopicPassingByValue.html>

## 1. Include and Namespace

Use `Eigen/Dense` for dense matrices, vectors, and arrays.

```cpp
#include <Eigen/Dense>

int main() {
    Eigen::MatrixXf x(2, 3);
    x.setZero();
}
```

You can shorten names inside implementation files:

```cpp
#include <Eigen/Dense>

using Eigen::MatrixXf;
using Eigen::VectorXf;

int main() {
    MatrixXf W = MatrixXf::Random(3, 2);
    VectorXf b = VectorXf::Zero(3);
}
```

For a neural-network library, prefer project aliases:

```cpp
#include <Eigen/Dense>

namespace fluxions {
using Scalar = float;
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using RowVector = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;
} // namespace fluxions
```

CMake integration usually looks like this:

```cmake
find_package(Eigen3 REQUIRED)

target_link_libraries(your_target
    PUBLIC
        Eigen3::Eigen
)
```

Eigen is header-only for the dense features in this reference. Linking the
`Eigen3::Eigen` CMake target mainly gives your compiler the right include paths
and compile definitions.

## 2. Mental Model: Matrix vs Array

Eigen has two dense object families:

- `Matrix`: linear algebra meaning. `A * B` means matrix multiplication.
- `Array`: coefficient-wise meaning. `a * b` means element-by-element multiply.

For NN code, use `MatrixXf` for tensors and switch to `.array()` only when you
need element-wise operations.

```cpp
Eigen::MatrixXf A(2, 2);
Eigen::MatrixXf B(2, 2);
A << 1, 2,
     3, 4;
B << 10, 20,
     30, 40;

Eigen::MatrixXf matmul = A * B;                    // linear algebra product
Eigen::MatrixXf hadamard = (A.array() * B.array()); // element-wise product
```

Convert between the two expression modes:

```cpp
Eigen::MatrixXf X = Eigen::MatrixXf::Random(2, 2);

Eigen::MatrixXf Y = X.array().exp().matrix(); // Matrix -> Array -> Matrix
Eigen::ArrayXXf Z = X.array();                // Matrix expression assigned to Array
Eigen::MatrixXf W = Z.matrix();               // Array expression assigned to Matrix
```

## 3. Common Types

Dynamic-size float matrix:

```cpp
Eigen::MatrixXf X(4, 3); // 4 rows, 3 columns, float
```

Dynamic-size double matrix:

```cpp
Eigen::MatrixXd X(4, 3); // double
```

Dynamic-size float column vector:

```cpp
Eigen::VectorXf x(5); // 5 x 1
```

Dynamic-size float row vector:

```cpp
Eigen::RowVectorXf x(5); // 1 x 5
```

Fixed-size matrix:

```cpp
Eigen::Matrix<float, 2, 3> W; // 2 rows, 3 columns, fixed at compile time
```

Fixed-size vector:

```cpp
Eigen::Vector3f v; // same as Matrix<float, 3, 1>
```

Generic template form:

```cpp
Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> X;
Eigen::Matrix<float, Eigen::Dynamic, 1> v;
Eigen::Matrix<float, 1, Eigen::Dynamic> row;
```

Useful typedef naming:

```cpp
Eigen::Matrix2f A;      // 2 x 2 float
Eigen::Matrix3d B;      // 3 x 3 double
Eigen::MatrixXi C;      // dynamic x dynamic int
Eigen::Vector4f v;      // 4 x 1 float
Eigen::RowVector3d r;   // 1 x 3 double
Eigen::ArrayXXf arr;    // dynamic x dynamic float array
Eigen::ArrayXf vec_arr; // dynamic 1D float array
```

Use `Eigen::Index` for rows, columns, and sizes:

```cpp
Eigen::MatrixXf X(10, 20);
Eigen::Index rows = X.rows();
Eigen::Index cols = X.cols();
Eigen::Index total = X.size();
```

## 4. Construction and Initialization

Default construction creates an object with no useful initialized values.

```cpp
Eigen::MatrixXf X;       // size 0 x 0
Eigen::MatrixXf Y(2, 3); // allocated, but coefficients are uninitialized
```

Initialize with comma syntax:

```cpp
Eigen::MatrixXf X(2, 3);
X << 1, 2, 3,
     4, 5, 6;
```

Initialize fixed-size vectors directly:

```cpp
Eigen::Vector3f x(1.0f, 2.0f, 3.0f);
```

Initialize with nested initializer lists:

```cpp
Eigen::MatrixXi X {
    {1, 2, 3},
    {4, 5, 6}
};
```

Zeros:

```cpp
Eigen::MatrixXf X = Eigen::MatrixXf::Zero(3, 4);
Eigen::VectorXf b = Eigen::VectorXf::Zero(3);
```

Ones:

```cpp
Eigen::MatrixXf X = Eigen::MatrixXf::Ones(2, 3);
```

Constant:

```cpp
Eigen::MatrixXf X = Eigen::MatrixXf::Constant(2, 3, 0.01f);
```

Identity:

```cpp
Eigen::MatrixXf I = Eigen::MatrixXf::Identity(4, 4);
```

Random values:

```cpp
Eigen::MatrixXf W = Eigen::MatrixXf::Random(3, 2); // roughly in [-1, 1]
```

Scale random values, useful for simple parameter init:

```cpp
float scale = 0.1f;
Eigen::MatrixXf W = scale * Eigen::MatrixXf::Random(3, 2);
```

Linearly spaced vector:

```cpp
Eigen::VectorXf x = Eigen::VectorXf::LinSpaced(5, 0.0f, 1.0f);
// x = [0, 0.25, 0.5, 0.75, 1]
```

Set an existing object:

```cpp
Eigen::MatrixXf X(2, 3);
X.setZero();
X.setOnes();
X.setConstant(2.0f);
X.setRandom();
```

Set identity on an existing square matrix:

```cpp
Eigen::MatrixXf I(4, 4);
I.setIdentity();
```

## 5. Shape, Size, and Resizing

Read dimensions:

```cpp
Eigen::MatrixXf X(2, 3);
int rows = static_cast<int>(X.rows());
int cols = static_cast<int>(X.cols());
int size = static_cast<int>(X.size());
```

Resize dynamic objects:

```cpp
Eigen::MatrixXf X(2, 3);
X.resize(4, 5); // old values may be destroyed
```

Conservative resize keeps old coefficients where possible:

```cpp
Eigen::VectorXf v(3);
v << 1, 2, 3;
v.conservativeResize(5);
v.tail(2).setZero();
```

Assignment can resize the left-hand side if it is dynamic:

```cpp
Eigen::MatrixXf A(2, 2);
Eigen::MatrixXf B(5, 4);
A = B; // A becomes 5 x 4
```

Check shape manually in your own NN ops:

```cpp
Eigen::MatrixXf matmul_checked(const Eigen::MatrixXf& A,
                               const Eigen::MatrixXf& B) {
    assert(A.cols() == B.rows());
    return A * B;
}
```

## 5.1. Reshape Views

`reshaped()` returns a view expression with a different shape. It does not copy
unless you assign it into a concrete matrix.

Reshape a matrix expression:

```cpp
Eigen::MatrixXf X(2, 3);
X << 1, 2, 3,
     4, 5, 6;

Eigen::MatrixXf Y = X.reshaped(3, 2);
```

Flatten as a column-vector expression:

```cpp
Eigen::VectorXf flat = X.reshaped();
```

Flatten in row-major order:

```cpp
Eigen::VectorXf flat_row_major = X.reshaped<Eigen::RowMajor>();
```

Use `reshaped()` to initialize a fixed matrix from a vector:

```cpp
Eigen::VectorXf theta = Eigen::VectorXf::LinSpaced(6, 1.0f, 6.0f);
Eigen::MatrixXf W = theta.reshaped(2, 3);
```

In-place reshape for dynamic matrices is `resize()`, not `reshaped()`:

```cpp
Eigen::MatrixXf X(2, 3);
X.setRandom();
X.resize(3, 2); // same storage, new dynamic shape
```

For NN tensors, be explicit about flatten order. Eigen's default reshape view
interprets coefficients in column-major order.

## 6. Coefficient Access

Matrix element access is row first, column second. Indices start at 0.

```cpp
Eigen::MatrixXf X(2, 3);
X.setZero();
X(0, 0) = 1.0f;
X(1, 2) = 5.0f;
```

Vector element access:

```cpp
Eigen::VectorXf x(3);
x(0) = 10.0f;
x(1) = 20.0f;
x(2) = 30.0f;
```

`operator[]` is allowed for vectors:

```cpp
Eigen::VectorXf x(2);
x[0] = 1.0f;
x[1] = 2.0f;
```

Raw memory pointer:

```cpp
Eigen::MatrixXf X(2, 3);
float* ptr = X.data();
ptr[0] = 1.0f;
```

Important: default dense matrices are column-major, so `data()` is laid out by
columns unless you explicitly request row-major storage.

## 7. Basic Matrix Arithmetic

Addition and subtraction:

```cpp
Eigen::MatrixXf A = Eigen::MatrixXf::Ones(2, 2);
Eigen::MatrixXf B = Eigen::MatrixXf::Constant(2, 2, 3.0f);

Eigen::MatrixXf C = A + B;
Eigen::MatrixXf D = B - A;
```

Scalar multiply and divide:

```cpp
Eigen::MatrixXf X = Eigen::MatrixXf::Ones(2, 2);

Eigen::MatrixXf A = 2.0f * X;
Eigen::MatrixXf B = X / 4.0f;
```

Matrix multiplication:

```cpp
Eigen::MatrixXf A(2, 3);
Eigen::MatrixXf B(3, 4);
A.setRandom();
B.setRandom();

Eigen::MatrixXf C = A * B; // 2 x 4
```

Matrix-vector multiplication:

```cpp
Eigen::MatrixXf W(3, 2);
Eigen::VectorXf x(2);
W.setRandom();
x.setRandom();

Eigen::VectorXf y = W * x; // 3 x 1
```

Compound assignment:

```cpp
Eigen::MatrixXf X = Eigen::MatrixXf::Ones(2, 2);
X += Eigen::MatrixXf::Identity(2, 2);
X *= 0.5f;
```

Transpose:

```cpp
Eigen::MatrixXf X(2, 3);
X.setRandom();

Eigen::MatrixXf Xt = X.transpose(); // 3 x 2
```

In-place transpose:

```cpp
Eigen::MatrixXf X(2, 3);
X.setRandom();
X.transposeInPlace(); // X becomes 3 x 2
```

Dot product:

```cpp
Eigen::VectorXf a(3), b(3);
a << 1, 2, 3;
b << 4, 5, 6;

float d = a.dot(b); // 32
```

Norms:

```cpp
Eigen::VectorXf g(3);
g << 3, 4, 0;

float l2 = g.norm();        // 5
float l2_sq = g.squaredNorm();
float l1 = g.lpNorm<1>();
float linf = g.lpNorm<Eigen::Infinity>();
```

Trace:

```cpp
Eigen::MatrixXf X = Eigen::MatrixXf::Identity(3, 3);
float tr = X.trace(); // 3
```

Avoid explicit inverse for solving systems:

```cpp
Eigen::MatrixXf A(3, 3);
Eigen::VectorXf b(3);
A.setRandom();
b.setRandom();

Eigen::VectorXf x = A.colPivHouseholderQr().solve(b);
```

## 8. Coefficient-Wise Array Operations

Use `.array()` for element-wise neural-network math.

Element-wise multiply:

```cpp
Eigen::MatrixXf A(2, 2), B(2, 2);
A.setRandom();
B.setRandom();

Eigen::MatrixXf C = (A.array() * B.array()).matrix();
```

Element-wise divide:

```cpp
Eigen::MatrixXf Y = (A.array() / B.array()).matrix();
```

Add/subtract scalar to every coefficient:

```cpp
Eigen::MatrixXf X = Eigen::MatrixXf::Ones(2, 2);
Eigen::MatrixXf Y = (X.array() + 1.0f).matrix();
Eigen::MatrixXf Z = (X.array() - 0.5f).matrix();
```

Power:

```cpp
Eigen::MatrixXf sq = X.array().square().matrix();
Eigen::MatrixXf cube = X.array().pow(3.0f).matrix();
```

Square root:

```cpp
Eigen::MatrixXf x = Eigen::MatrixXf::Constant(2, 2, 4.0f);
Eigen::MatrixXf y = x.array().sqrt().matrix();
```

Exponential:

```cpp
Eigen::MatrixXf logits(2, 2);
logits.setRandom();

Eigen::MatrixXf e = logits.array().exp().matrix();
```

Log:

```cpp
Eigen::MatrixXf probs = Eigen::MatrixXf::Constant(2, 2, 0.5f);
Eigen::MatrixXf logp = probs.array().log().matrix();
```

Tanh:

```cpp
Eigen::MatrixXf h = X.array().tanh().matrix();
```

Absolute value:

```cpp
Eigen::MatrixXf abs_x = X.array().abs().matrix();
```

Coefficient-wise min and max:

```cpp
Eigen::MatrixXf clipped = X.array().max(-1.0f).min(1.0f).matrix();
```

ReLU:

```cpp
Eigen::MatrixXf relu(const Eigen::MatrixXf& x) {
    return x.array().max(0.0f).matrix();
}
```

Sigmoid:

```cpp
Eigen::MatrixXf sigmoid(const Eigen::MatrixXf& x) {
    return (1.0f / (1.0f + (-x.array()).exp())).matrix();
}
```

Derivative of sigmoid if you already have sigmoid output:

```cpp
Eigen::MatrixXf sigmoid_backward_from_y(const Eigen::MatrixXf& y) {
    return (y.array() * (1.0f - y.array())).matrix();
}
```

Derivative of tanh if you already have tanh output:

```cpp
Eigen::MatrixXf tanh_backward_from_y(const Eigen::MatrixXf& y) {
    return (1.0f - y.array().square()).matrix();
}
```

Comparisons produce boolean array expressions:

```cpp
Eigen::MatrixXf X(2, 2);
X << -1, 2,
      3, -4;

bool all_positive = (X.array() > 0.0f).all();
bool any_positive = (X.array() > 0.0f).any();
int count_positive = static_cast<int>((X.array() > 0.0f).count());
```

`select` is Eigen's vectorized ternary:

```cpp
Eigen::MatrixXf X(2, 2);
X << -1, 2,
      3, -4;

Eigen::MatrixXf relu_x = (X.array() > 0.0f).select(X.array(), 0.0f).matrix();
```

Leaky ReLU:

```cpp
Eigen::MatrixXf leaky_relu(const Eigen::MatrixXf& x, float alpha) {
    return (x.array() > 0.0f)
        .select(x.array(), alpha * x.array())
        .matrix();
}
```

Useful coefficient-wise math catalog:

```cpp
Eigen::MatrixXf X = Eigen::MatrixXf::Random(2, 2);

Eigen::MatrixXf abs_x = X.array().abs().matrix();
Eigen::MatrixXf inv_x = X.array().inverse().matrix(); // 1 / x
Eigen::MatrixXf sq_x = X.array().square().matrix();
Eigen::MatrixXf cube_x = X.array().cube().matrix();
Eigen::MatrixXf sqrt_x = X.array().abs().sqrt().matrix();
Eigen::MatrixXf rsqrt_x = X.array().abs().rsqrt().matrix();
Eigen::MatrixXf exp_x = X.array().exp().matrix();
Eigen::MatrixXf log_x = X.array().abs().log().matrix();
Eigen::MatrixXf log1p_x = X.array().abs().log1p().matrix();
```

Trig and hyperbolic functions:

```cpp
Eigen::MatrixXf X = Eigen::MatrixXf::Random(2, 2);

Eigen::MatrixXf sin_x = X.array().sin().matrix();
Eigen::MatrixXf cos_x = X.array().cos().matrix();
Eigen::MatrixXf tan_x = X.array().tan().matrix();
Eigen::MatrixXf sinh_x = X.array().sinh().matrix();
Eigen::MatrixXf cosh_x = X.array().cosh().matrix();
Eigen::MatrixXf tanh_x = X.array().tanh().matrix();
```

NaN and infinity checks:

```cpp
Eigen::MatrixXf X(2, 2);
X << 1.0f, std::numeric_limits<float>::infinity(),
     3.0f, std::numeric_limits<float>::quiet_NaN();

Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> finite_mask = X.array().isFinite();
Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> nan_mask = X.array().isNaN();
Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> inf_mask = X.array().isInf();
```

Matrix-side `cwise*` methods are alternatives to `.array()`:

```cpp
Eigen::MatrixXf X = Eigen::MatrixXf::Random(2, 2);
Eigen::MatrixXf Y = Eigen::MatrixXf::Random(2, 2);

Eigen::MatrixXf abs_x = X.cwiseAbs();
Eigen::MatrixXf inv_x = X.cwiseInverse();
Eigen::MatrixXf min_xy = X.cwiseMin(Y);
Eigen::MatrixXf max_xy = X.cwiseMax(Y);
Eigen::MatrixXf prod_xy = X.cwiseProduct(Y);
Eigen::MatrixXf quot_xy = X.cwiseQuotient(Y);
```

## 9. Reductions

Whole-matrix reductions:

```cpp
Eigen::MatrixXf X(2, 3);
X << 1, 2, 3,
     4, 5, 6;

float s = X.sum();       // 21
float p = X.prod();      // 720
float mean = X.mean();   // 3.5
float minv = X.minCoeff();
float maxv = X.maxCoeff();
```

Column-wise reductions:

```cpp
Eigen::MatrixXf X(2, 3);
X << 1, 2, 3,
     4, 5, 6;

Eigen::RowVectorXf col_sums = X.colwise().sum(); // [5, 7, 9]
Eigen::RowVectorXf col_max = X.colwise().maxCoeff();
```

Row-wise reductions:

```cpp
Eigen::VectorXf row_sums = X.rowwise().sum(); // [6, 15]
Eigen::VectorXf row_max = X.rowwise().maxCoeff();
```

Find max value and index in a vector:

```cpp
Eigen::VectorXf scores(4);
scores << 0.1f, 0.7f, -0.2f, 0.4f;

Eigen::Index argmax;
float best = scores.maxCoeff(&argmax);
```

Find max value and matrix position:

```cpp
Eigen::MatrixXf X(2, 2);
X << 1, 5,
     3, 2;

Eigen::Index row, col;
float best = X.maxCoeff(&row, &col);
```

Boolean reductions:

```cpp
Eigen::MatrixXf X(2, 2);
X << 1, 2,
     3, 4;

bool ok = (X.array() > 0.0f).all();
bool has_large = (X.array() > 3.0f).any();
int num_large = static_cast<int>((X.array() > 3.0f).count());
```

## 10. Broadcasting

Eigen broadcasting is usually done with `rowwise()` or `colwise()`.

Add a bias vector to every column. This is common if each column is one batch
example.

```cpp
Eigen::MatrixXf Z(3, 4); // 3 features, 4 batch items
Eigen::VectorXf b(3);    // one bias per feature
Z.setRandom();
b << 0.1f, 0.2f, 0.3f;

Z.colwise() += b;
```

Non-mutating column broadcast:

```cpp
Eigen::MatrixXf Y = Z.colwise() + b;
```

Add a row vector to every row:

```cpp
Eigen::MatrixXf X(3, 4);
Eigen::RowVectorXf r(4);
X.setRandom();
r << 1, 2, 3, 4;

X.rowwise() += r;
```

If you have a column vector but need row broadcasting, transpose it:

```cpp
Eigen::VectorXf v(4);
v << 1, 2, 3, 4;

Eigen::MatrixXf X(3, 4);
X.setZero();
X.rowwise() += v.transpose();
```

Replicate explicitly:

```cpp
Eigen::VectorXf b(3);
b << 1, 2, 3;

Eigen::MatrixXf B = b.replicate(1, 4); // 3 rows, 4 repeated columns
```

Softmax per column with broadcasting:

```cpp
Eigen::MatrixXf softmax_columns(const Eigen::MatrixXf& logits) {
    Eigen::MatrixXf shifted =
        logits.rowwise() - logits.colwise().maxCoeff();
    Eigen::MatrixXf exps = shifted.array().exp().matrix();
    Eigen::RowVectorXf denom = exps.colwise().sum();
    return (exps.array().rowwise() / denom.array()).matrix();
}
```

For a batch layout where columns are examples:

```cpp
// logits: classes x batch
// output: classes x batch, each column sums to 1
Eigen::MatrixXf probs = softmax_columns(logits);
```

## 11. Blocks, Rows, Columns, and Slices

General block:

```cpp
Eigen::MatrixXf X(4, 5);
X.setRandom();

Eigen::MatrixXf sub = X.block(1, 2, 2, 3);
// start row 1, start col 2, take 2 rows and 3 cols
```

Write into a block:

```cpp
Eigen::MatrixXf X = Eigen::MatrixXf::Zero(4, 4);
X.block(1, 1, 2, 2).setOnes();
```

Fixed-size block syntax:

```cpp
Eigen::MatrixXf X(4, 4);
X.setRandom();

Eigen::Matrix2f top_left = X.block<2, 2>(0, 0);
```

Single row:

```cpp
Eigen::MatrixXf X(3, 4);
X.setRandom();

Eigen::RowVectorXf row = X.row(1);
X.row(2).setZero();
```

Single column:

```cpp
Eigen::VectorXf col = X.col(0);
X.col(1).array() *= 2.0f;
```

Top/bottom rows:

```cpp
Eigen::MatrixXf top = X.topRows(2);
Eigen::MatrixXf bottom = X.bottomRows(1);
```

Left/right columns:

```cpp
Eigen::MatrixXf left = X.leftCols(2);
Eigen::MatrixXf right = X.rightCols(2);
```

Corners:

```cpp
Eigen::MatrixXf tl = X.topLeftCorner(2, 2);
Eigen::MatrixXf br = X.bottomRightCorner(2, 2);
```

Vector head/tail/segment:

```cpp
Eigen::VectorXf v(6);
v << 1, 2, 3, 4, 5, 6;

Eigen::VectorXf h = v.head(3);       // 1, 2, 3
Eigen::VectorXf t = v.tail(2);       // 5, 6
Eigen::VectorXf s = v.segment(2, 3); // 3, 4, 5
```

Modify a vector segment:

```cpp
v.segment(1, 3).setZero();
```

Use blocks for minibatches:

```cpp
// X: features x total_examples
Eigen::MatrixXf batch = X.middleCols(32, 16); // 16 examples starting at col 32
```

Use blocks for parameter packing:

```cpp
Eigen::VectorXf theta(12);
theta.setRandom();

Eigen::Map<Eigen::MatrixXf> W(theta.data(), 3, 4);
```

## 11.1. Slicing and Indexing API

Eigen 3.4+ also supports Python-like slicing through `operator()`. Include the
placeholders you use most often:

```cpp
using Eigen::all;
using Eigen::last;
using Eigen::lastN;
using Eigen::seq;
using Eigen::seqN;
```

Take all rows and one column:

```cpp
Eigen::MatrixXf X(3, 5);
X.setRandom();

Eigen::VectorXf c = X(all, 2); // same idea as X.col(2)
```

Take one row and all columns:

```cpp
Eigen::RowVectorXf r = X(1, all); // same idea as X.row(1)
```

Take a rectangular slice with inclusive end indices:

```cpp
Eigen::MatrixXf sub = X(seq(0, 1), seq(2, 4));
// rows 0..1, cols 2..4
```

Take a block with start and size:

```cpp
Eigen::MatrixXf sub = X(seqN(1, 2), seqN(0, 3));
// start row 1, take 2 rows; start col 0, take 3 cols
```

Use `last` for the final row/column:

```cpp
Eigen::VectorXf last_col = X(all, last);
Eigen::RowVectorXf last_row = X(last, all);
```

Use `lastN` for the final rows/columns:

```cpp
Eigen::MatrixXf tail_cols = X(all, lastN(2));
Eigen::MatrixXf tail_rows = X(lastN(2), all);
```

Use strides:

```cpp
Eigen::MatrixXf even_cols = X(all, seq(0, last, 2));
```

Reverse with negative stride:

```cpp
Eigen::MatrixXf reversed_cols = X(all, seq(last, 0, -1));
```

Gather arbitrary columns by index list:

```cpp
std::vector<int> ids{4, 0, 2};
Eigen::MatrixXf gathered = X(all, ids);
```

Gather with an Eigen integer array:

```cpp
Eigen::ArrayXi ids(3);
ids << 2, 2, 0;

Eigen::MatrixXf gathered = X(all, ids);
```

Use slicing for minibatches:

```cpp
// X: features x total_examples
int start = 64;
int batch = 32;

Eigen::MatrixXf minibatch = X(all, seqN(start, batch));
```

Assign into a slice:

```cpp
Eigen::MatrixXf X = Eigen::MatrixXf::Zero(4, 5);
X(seqN(1, 2), all).setOnes(); // rows 1 and 2 become ones
```

## 12. `Map`: View External Memory as Eigen

`Eigen::Map` creates a non-owning Eigen view over existing memory. This is useful
for tensor storage, arena allocation, memory pools, or flattened parameters.

Map contiguous vector memory:

```cpp
float data[] = {1, 2, 3};
Eigen::Map<Eigen::Vector3f> v(data);

v(0) = 10.0f; // writes into data[0]
```

Map dynamic vector memory:

```cpp
float* raw = new float[5]{1, 2, 3, 4, 5};
Eigen::Map<Eigen::VectorXf> v(raw, 5);

float s = v.sum();
delete[] raw;
```

Map matrix memory:

```cpp
float data[] = {1, 2, 3, 4, 5, 6};

// Default Eigen MatrixXf is column-major:
Eigen::Map<Eigen::MatrixXf> X(data, 2, 3);
```

Map row-major memory:

```cpp
using RowMajorMatrixXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

float data[] = {1, 2, 3, 4, 5, 6};
Eigen::Map<RowMajorMatrixXf> X(data, 2, 3);
```

Map const memory:

```cpp
void print_sum(const float* data, int n) {
    Eigen::Map<const Eigen::VectorXf> v(data, n);
    std::cout << v.sum() << "\n";
}
```

Map with stride:

```cpp
float data[] = {1, 99, 2, 99, 3, 99};

Eigen::Map<Eigen::VectorXf, 0, Eigen::InnerStride<2>> v(data, 3);
// v sees [1, 2, 3]
```

## 13. `Ref`: Function Parameters Without Unnecessary Copies

`Eigen::Ref` lets a non-template function accept many compatible Eigen objects.

Read-only matrix parameter:

```cpp
float sum_all(const Eigen::Ref<const Eigen::MatrixXf>& X) {
    return X.sum();
}
```

Mutable matrix parameter:

```cpp
void zero_in_place(Eigen::Ref<Eigen::MatrixXf> X) {
    X.setZero();
}
```

Read-only vector parameter:

```cpp
float l2_norm(const Eigen::Ref<const Eigen::VectorXf>& x) {
    return x.norm();
}
```

Use `Ref` with blocks:

```cpp
void add_one(Eigen::Ref<Eigen::MatrixXf> X) {
    X.array() += 1.0f;
}

Eigen::MatrixXf A = Eigen::MatrixXf::Zero(4, 4);
add_one(A.block(1, 1, 2, 2));
```

When you want maximum flexibility and are okay with templates:

```cpp
template <typename Derived>
float sum_template(const Eigen::MatrixBase<Derived>& X) {
    return X.sum();
}
```

For most Fluxions operation APIs, this is a good default:

```cpp
using MatRef = Eigen::Ref<const Eigen::MatrixXf>;
using MutMatRef = Eigen::Ref<Eigen::MatrixXf>;
using VecRef = Eigen::Ref<const Eigen::VectorXf>;

void affine_forward(MatRef W, MatRef X, VecRef b, MutMatRef Y) {
    Y.noalias() = W * X;
    Y.colwise() += b;
}
```

## 13.1. STL Iterators

Dense 1D Eigen expressions support STL-style iteration.

Range-for over a vector:

```cpp
Eigen::VectorXf x(3);
x << 1, 2, 3;

for (float value : x) {
    std::cout << value << "\n";
}
```

Mutate vector values:

```cpp
for (float& value : x) {
    value *= 2.0f;
}
```

Use STL algorithms on 1D arrays/vectors:

```cpp
#include <algorithm>

Eigen::ArrayXf x(5);
x << 3, 1, 4, 2, 5;

std::sort(x.begin(), x.end());
```

For 2D matrices, iterate over a flattened reshaped view:

```cpp
Eigen::MatrixXf X(2, 3);
X << 1, 2, 3,
     4, 5, 6;

for (float value : X.reshaped()) {
    std::cout << value << "\n";
}
```

Iterate row-wise:

```cpp
Eigen::ArrayXXf X = Eigen::ArrayXXf::Random(3, 4);

for (auto row : X.rowwise()) {
    row = row.max(0.0f); // in-place ReLU per row
}
```

Iterate column-wise:

```cpp
Eigen::ArrayXXf X = Eigen::ArrayXXf::Random(3, 4);

for (auto col : X.colwise()) {
    col -= col.mean();
}
```

## 14. Storage Order

Eigen defaults to column-major storage for matrices.

```cpp
Eigen::MatrixXf X(2, 3); // column-major by default
```

Explicit row-major matrix:

```cpp
using RowMajorMatrixXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

RowMajorMatrixXf X(2, 3);
```

Explicit column-major matrix:

```cpp
using ColMajorMatrixXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
```

Storage order matters when:

- you use `data()`
- you map external memory with `Map`
- you interface with C APIs
- you care about cache access patterns

Example:

```cpp
using RowMajorMatrixXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

RowMajorMatrixXf X(2, 3);
X << 1, 2, 3,
     4, 5, 6;

float first_raw = X.data()[0]; // 1
float second_raw = X.data()[1]; // 2 in row-major
```

For a first NN backend, pick one storage order and keep it consistent. Eigen's
default column-major is fine if you store minibatch examples as columns.

## 15. Lazy Evaluation, `eval()`, and `noalias()`

Eigen uses expression templates. Most expressions are not computed until they
are assigned.

```cpp
Eigen::MatrixXf A, B, C;
A = Eigen::MatrixXf::Random(2, 2);
B = Eigen::MatrixXf::Random(2, 2);
C = Eigen::MatrixXf::Random(2, 2);

Eigen::MatrixXf Y = A + B + C; // usually one fused evaluation loop
```

Force evaluation:

```cpp
Eigen::MatrixXf Y = (A + B).eval();
```

Use `eval()` to avoid aliasing problems with overlapping blocks:

```cpp
Eigen::MatrixXf X(3, 3);
X << 1, 2, 3,
     4, 5, 6,
     7, 8, 9;

X.bottomRightCorner(2, 2) = X.topLeftCorner(2, 2).eval();
```

Use `noalias()` when assigning a matrix product into a destination that you know
does not overlap with the inputs:

```cpp
Eigen::MatrixXf A(2, 3), B(3, 4), C(2, 4);
A.setRandom();
B.setRandom();

C.noalias() = A * B;
```

Good affine layer pattern:

```cpp
Eigen::MatrixXf Y(W.rows(), X.cols());
Y.noalias() = W * X;
Y.colwise() += b;
```

Do not use `noalias()` if the output is also an input:

```cpp
// Bad if A appears on both sides:
// A.noalias() = A * B;

// Safe:
A = (A * B).eval();
```

Common alias-safe in-place operations:

```cpp
X.transposeInPlace();
X.reverseInPlace();
```

## 16. The `auto` Rule

Eigen expressions can be lazy proxy objects. `auto` can accidentally store a
lazy expression instead of a concrete matrix.

Usually safe:

```cpp
auto rows = X.rows();
auto cols = X.cols();
```

Potentially surprising:

```cpp
auto expr = A + B; // expr is not necessarily a MatrixXf
```

If you want actual computed data, spell the type or call `.eval()`:

```cpp
Eigen::MatrixXf Y = A + B;
auto Z = (A + B).eval();
```

For graph nodes, it is usually better to store concrete values:

```cpp
struct Value {
    Eigen::MatrixXf data;
    Eigen::MatrixXf grad;
};
```

## 17. Casting and Scalar Types

Cast float matrix to double:

```cpp
Eigen::MatrixXf Xf = Eigen::MatrixXf::Random(2, 2);
Eigen::MatrixXd Xd = Xf.cast<double>();
```

Cast int labels to float:

```cpp
Eigen::VectorXi labels(3);
labels << 0, 1, 2;

Eigen::VectorXf labels_f = labels.cast<float>();
```

Complex matrices exist, but you probably do not need them for the first NN
backend:

```cpp
Eigen::MatrixXcf C(2, 2); // complex<float>
```

## 18. Diagonal Operations

Create diagonal matrix from vector:

```cpp
Eigen::VectorXf d(3);
d << 1, 2, 3;

Eigen::MatrixXf D = d.asDiagonal();
```

Multiply by diagonal without explicitly forming a dense matrix:

```cpp
Eigen::MatrixXf X(3, 4);
Eigen::VectorXf scale(3);
X.setRandom();
scale << 1, 2, 3;

Eigen::MatrixXf Y = scale.asDiagonal() * X;
```

Access diagonal:

```cpp
Eigen::MatrixXf X = Eigen::MatrixXf::Identity(4, 4);
Eigen::VectorXf d = X.diagonal();
X.diagonal().array() += 1.0f;
```

## 19. Triangular and Symmetric Views

Useful mostly for linear algebra utilities, not everyday NN code.

Triangular solve:

```cpp
Eigen::MatrixXf L = Eigen::MatrixXf::Identity(3, 3);
Eigen::VectorXf b(3);
b << 1, 2, 3;

Eigen::VectorXf x = L.triangularView<Eigen::Lower>().solve(b);
```

Use only upper triangle:

```cpp
Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 3);
Eigen::MatrixXf U = A.triangularView<Eigen::Upper>();
```

Symmetric/self-adjoint matrix-vector product:

```cpp
Eigen::MatrixXf A = Eigen::MatrixXf::Random(3, 3);
Eigen::VectorXf x = Eigen::VectorXf::Random(3);

Eigen::VectorXf y = A.selfadjointView<Eigen::Upper>() * x;
```

## 20. Sparse Basics

For the first Fluxions backend, dense matrices are enough. Sparse is useful
later for very large sparse features or graph-style data.

Include sparse module:

```cpp
#include <Eigen/Sparse>
```

Sparse matrix type:

```cpp
Eigen::SparseMatrix<float> S(1000, 1000);
```

Insert triplets:

```cpp
std::vector<Eigen::Triplet<float>> triplets;
triplets.emplace_back(0, 1, 2.0f);
triplets.emplace_back(3, 4, 5.0f);

Eigen::SparseMatrix<float> S(10, 10);
S.setFromTriplets(triplets.begin(), triplets.end());
```

Sparse-dense multiply:

```cpp
Eigen::VectorXf x = Eigen::VectorXf::Random(10);
Eigen::VectorXf y = S * x;
```

## 21. Common NN Snippets

Affine layer, columns are batch items:

```cpp
// W: out_dim x in_dim
// X: in_dim x batch
// b: out_dim
// Y: out_dim x batch
Eigen::MatrixXf affine(const Eigen::MatrixXf& W,
                       const Eigen::MatrixXf& X,
                       const Eigen::VectorXf& b) {
    Eigen::MatrixXf Y(W.rows(), X.cols());
    Y.noalias() = W * X;
    Y.colwise() += b;
    return Y;
}
```

Affine backward:

```cpp
struct AffineGrads {
    Eigen::MatrixXf dW;
    Eigen::MatrixXf dX;
    Eigen::VectorXf db;
};

AffineGrads affine_backward(const Eigen::MatrixXf& W,
                            const Eigen::MatrixXf& X,
                            const Eigen::MatrixXf& dY) {
    AffineGrads g;
    g.dW.resize(dY.rows(), X.rows());
    g.dX.resize(W.cols(), dY.cols());

    g.dW.noalias() = dY * X.transpose();
    g.dX.noalias() = W.transpose() * dY;
    g.db = dY.rowwise().sum();
    return g;
}
```

Mean squared error:

```cpp
float mse(const Eigen::MatrixXf& pred, const Eigen::MatrixXf& target) {
    Eigen::MatrixXf diff = pred - target;
    return diff.array().square().mean();
}
```

MSE gradient:

```cpp
Eigen::MatrixXf mse_grad(const Eigen::MatrixXf& pred,
                         const Eigen::MatrixXf& target) {
    return (2.0f / static_cast<float>(pred.size())) * (pred - target);
}
```

Softmax cross entropy for one-hot labels, columns are batch items:

```cpp
float softmax_cross_entropy(const Eigen::MatrixXf& logits,
                            const Eigen::MatrixXf& one_hot) {
    Eigen::MatrixXf shifted =
        logits.rowwise() - logits.colwise().maxCoeff();
    Eigen::RowVectorXf logsumexp =
        shifted.array().exp().matrix().colwise().sum().array().log().matrix();
    Eigen::MatrixXf log_probs = shifted.rowwise() - logsumexp;
    return -(one_hot.array() * log_probs.array()).sum()
        / static_cast<float>(logits.cols());
}
```

Softmax cross entropy gradient:

```cpp
Eigen::MatrixXf softmax_cross_entropy_grad(const Eigen::MatrixXf& logits,
                                           const Eigen::MatrixXf& one_hot) {
    Eigen::MatrixXf probs = softmax_columns(logits);
    return (probs - one_hot) / static_cast<float>(logits.cols());
}
```

SGD update:

```cpp
void sgd_update(Eigen::MatrixXf& param,
                const Eigen::MatrixXf& grad,
                float lr) {
    param -= lr * grad;
}
```

Gradient clipping by global norm for one matrix:

```cpp
void clip_grad(Eigen::MatrixXf& grad, float max_norm) {
    float n = grad.norm();
    if (n > max_norm) {
        grad *= max_norm / (n + 1e-6f);
    }
}
```

Xavier/Glorot uniform init:

```cpp
Eigen::MatrixXf glorot_uniform(int fan_in, int fan_out) {
    float limit = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    return limit * Eigen::MatrixXf::Random(fan_out, fan_in);
}
```

Flatten matrix to vector view:

```cpp
Eigen::MatrixXf W(3, 4);
W.setRandom();

Eigen::Map<Eigen::VectorXf> flat(W.data(), W.size());
```

Use `Map` for a tensor wrapper:

```cpp
struct TensorView {
    float* data;
    int rows;
    int cols;

    Eigen::Map<Eigen::MatrixXf> matrix() {
        return Eigen::Map<Eigen::MatrixXf>(data, rows, cols);
    }
};
```

Use `const Map` for read-only tensor data:

```cpp
struct ConstTensorView {
    const float* data;
    int rows;
    int cols;

Eigen::Map<const Eigen::MatrixXf> matrix() const {
        return Eigen::Map<const Eigen::MatrixXf>(data, rows, cols);
    }
};
```

## 21.1. Passing and Storing Eigen Objects Safely

Do not pass Eigen matrices by value in performance-sensitive code:

```cpp
// Avoid:
void bad_forward(Eigen::MatrixXf X);

// Prefer:
void good_forward(const Eigen::Ref<const Eigen::MatrixXf>& X);
```

Return by value is fine for computed results:

```cpp
Eigen::MatrixXf make_zero(int rows, int cols) {
    return Eigen::MatrixXf::Zero(rows, cols);
}
```

Dynamic-size members such as `MatrixXf` are normal graph-node storage:

```cpp
struct NodeValue {
    Eigen::MatrixXf value;
    Eigen::MatrixXf grad;
};
```

If a class stores fixed-size vectorizable Eigen members, add Eigen's aligned
operator macro:

```cpp
struct SmallState {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector4f h;
    Eigen::Matrix4f J;
};
```

For STL containers of fixed-size vectorizable Eigen types, use Eigen's aligned
allocator if you hit alignment assertions or target older compiler modes:

```cpp
std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>> states;
states.emplace_back(Eigen::Vector4f::Zero());
```

For your first Fluxions graph, prefer dynamic `MatrixXf`/`VectorXf` in node
storage and avoid storing many tiny fixed-size Eigen objects unless there is a
clear performance reason.

## 22. Computation Graph Design Notes

Store actual matrices in graph values:

```cpp
struct NodeValue {
    Eigen::MatrixXf value;
    Eigen::MatrixXf grad;
};
```

Forward op example:

```cpp
struct MatMulOp {
    static Eigen::MatrixXf forward(const Eigen::MatrixXf& A,
                                  const Eigen::MatrixXf& B) {
        Eigen::MatrixXf Y(A.rows(), B.cols());
        Y.noalias() = A * B;
        return Y;
    }
};
```

Backward op example:

```cpp
struct MatMulBackward {
    static void backward(const Eigen::MatrixXf& A,
                         const Eigen::MatrixXf& B,
                         const Eigen::MatrixXf& dY,
                         Eigen::MatrixXf& dA,
                         Eigen::MatrixXf& dB) {
        dA.noalias() += dY * B.transpose();
        dB.noalias() += A.transpose() * dY;
    }
};
```

Element-wise op forward/backward:

```cpp
struct ReluOp {
    static Eigen::MatrixXf forward(const Eigen::MatrixXf& X) {
        return X.array().max(0.0f).matrix();
    }

    static Eigen::MatrixXf backward(const Eigen::MatrixXf& X,
                                    const Eigen::MatrixXf& dY) {
        return (X.array() > 0.0f).select(dY.array(), 0.0f).matrix();
    }
};
```

Accumulate gradients, do not overwrite them:

```cpp
node.grad += incoming_grad;
```

Initialize gradients with shape:

```cpp
node.grad = Eigen::MatrixXf::Zero(node.value.rows(), node.value.cols());
```

## 23. Common Mistakes

Using `*` when you wanted element-wise multiply:

```cpp
// Matrix multiply:
Eigen::MatrixXf C = A * B;

// Element-wise multiply:
Eigen::MatrixXf H = (A.array() * B.array()).matrix();
```

Mixing matrix and array expressions directly:

```cpp
// Bad:
// Eigen::MatrixXf Y = X + X.array();

// Good:
Eigen::MatrixXf Y = X + X.array().matrix();
```

Forgetting bias shape:

```cpp
// b must be out_dim x 1 if using colwise() += b.
Y.colwise() += b;
```

Wrong softmax axis:

```cpp
// If columns are batch examples, normalize each column.
Eigen::MatrixXf probs = softmax_columns(logits);
```

Dangerous aliasing:

```cpp
// Risky:
// X = X.transpose();

// Good:
X.transposeInPlace();
```

Using `inverse()` unnecessarily:

```cpp
// Avoid:
// x = A.inverse() * b;

// Prefer:
x = A.colPivHouseholderQr().solve(b);
```

Assuming `resize()` preserves values:

```cpp
// resize may destroy old coefficients.
X.resize(new_rows, new_cols);

// use conservativeResize when preservation matters.
X.conservativeResize(new_rows, new_cols);
```

Assuming `MatrixXf Y(2, 3)` is zero-initialized:

```cpp
Eigen::MatrixXf Y(2, 3); // uninitialized
Y.setZero();             // now zero
```

## 24. Recommended Fluxions Defaults

Start simple:

```cpp
namespace fluxions {
using Scalar = float;
using Tensor = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
using TensorRef = Eigen::Ref<const Tensor>;
using MutTensorRef = Eigen::Ref<Tensor>;
} // namespace fluxions
```

Use a consistent minibatch convention:

```cpp
// Recommended:
// Tensor shape = features x batch
// W shape      = output_features x input_features
// b shape      = output_features x 1
// Y = W * X + b
```

Use column-major first:

```cpp
using Tensor = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;
```

Switch to row-major only if your tensor storage and access patterns clearly need
it:

```cpp
using RowMajorTensor =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
```

Prefer concrete matrices inside graph storage and `Ref` at API boundaries:

```cpp
struct TensorValue {
    fluxions::Tensor data;
    fluxions::Tensor grad;
};

void op_forward(fluxions::TensorRef x, fluxions::MutTensorRef y);
```

## 25. Tiny End-to-End Example

This example is a single affine + ReLU forward/backward pass with Eigen syntax
you will use constantly.

```cpp
#include <cassert>
#include <cmath>
#include <Eigen/Dense>

using Tensor = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

Tensor relu(const Tensor& x) {
    return x.array().max(0.0f).matrix();
}

Tensor relu_backward(const Tensor& x, const Tensor& dY) {
    return (x.array() > 0.0f).select(dY.array(), 0.0f).matrix();
}

int main() {
    const int in_dim = 4;
    const int out_dim = 3;
    const int batch = 2;

    Tensor X = Tensor::Random(in_dim, batch);
    Tensor W = 0.1f * Tensor::Random(out_dim, in_dim);
    Vector b = Vector::Zero(out_dim);

    Tensor Z(out_dim, batch);
    Z.noalias() = W * X;
    Z.colwise() += b;

    Tensor Y = relu(Z);

    Tensor dY = Tensor::Ones(out_dim, batch);
    Tensor dZ = relu_backward(Z, dY);

    Tensor dW(out_dim, in_dim);
    Tensor dX(in_dim, batch);
    Vector db(out_dim);

    dW.noalias() = dZ * X.transpose();
    dX.noalias() = W.transpose() * dZ;
    db = dZ.rowwise().sum();

    float lr = 0.01f;
    W -= lr * dW;
    b -= lr * db;
}
```
