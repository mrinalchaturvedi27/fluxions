# Fluxions

Fluxions is a C++ neural-network library inspired by DyNet-style
computation graphs.

## Build

```bash
cmake -S . -B cmake-build-debug
cmake --build cmake-build-debug
ctest --test-dir cmake-build-debug --output-on-failure
```

# Phase-A Foundations
## Tensor Operations

Our Tensor class will support all fundamental operations that neural networks
need:

```text
Operation Types:
┌─────────────────┬─────────────────┬─────────────────┐
│ Element-wise    │ Matrix Ops      │ Shape Ops       │
├─────────────────┼─────────────────┼─────────────────┤
│ + Addition      │ @ Matrix Mult   │ .reshape()      │
│ - Subtraction   │ .transpose()    │ .sum()          │
│ * Multiplication│                 │ .mean()         │
│ / Division      │                 │ .max()          │
└─────────────────┴─────────────────┴─────────────────┘
```

In C++, Fluxions uses `operator*` for matrix multiplication for now, because C++
does not have Python's `@` operator. Element-wise multiplication is exposed as
`mul()`.

## Tensor Class Architecture

```text
Tensor Class Structure:
┌─────────────────────────────────┐
│ Core Attributes:                │
│ • data_: Eigen::MatrixXf        │
│ • rows / cols                   │
│ • size: int (total elements)    │
│ • dtype: float                  │
├─────────────────────────────────┤
│ Arithmetic Operations:          │
│ • operator+, operator-          │
│ • operator*, mul(), div()       │
├─────────────────────────────────┤
│ Shape Operations:               │
│ • reshape(), transpose()        │
│ • sum(), mean(), max()          │
├─────────────────────────────────┤
│ Utility Methods:                │
│ • data(), shape()               │
│ • rows(), cols(), size()        │
│ • ndim(), dtype()               │
└─────────────────────────────────┘
```


# Current Layout

```text
fluxions/
├── core/      Eigen-backed Tensor core
├── docs/      Eigen notes and development references
├── memory/    Memory pool and device metadata
├── ops/       Planned operation implementations
├── optim/     Planned optimizers
├── params/    Planned parameter abstractions
└── tests/     GoogleTest tests
```

# Planned Layout

```text
fluxions/
├── core/
│   ├── tensor.h
│   └── tensor.cc
│
├── graph/
│   ├── computation_graph.h/.cc
│   ├── expression.h/.cc
│   ├── node.h/.cc
│   └── execution.h/.cc
│
├── ops/
│   ├── input.h/.cc
│   ├── parameter.h/.cc
│   ├── arithmetic.h/.cc
│   ├── matmul.h/.cc
│   ├── activations.h/.cc
│   ├── reductions.h/.cc
│   └── losses.h/.cc
│
├── params/
│   ├── parameter.h/.cc
│   └── parameter_collection.h/.cc
│
├── optim/
│   ├── optimizer.h/.cc
│   ├── sgd.h/.cc
│   └── adam.h/.cc
│
├── data/
│   ├── mnist_loader.h/.cc
│   └── batch.h/.cc
│
├── examples/
│   ├── xor
│   └── mnist_mlp
│
├── benchmarks/
│   ├── mnist_cpp
│   └── mnist_pytorch_baseline.py
│
└── tests/
    ├── test_tensor.cpp
    ├── test_gradcheck.cpp
    ├── test_graph.cpp
    └── test_optimizer.cpp
```
