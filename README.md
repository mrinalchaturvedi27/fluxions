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

Fluxions currently has the first pieces of the core training stack:

- `Dim`: lightweight shape metadata with a separate batch count
- `Tensor`: Eigen-backed numerical storage and basic math
- `ParameterStorage`: owns trainable values and gradients
- `Parameter`: lightweight handle to shared parameter storage
- `ParameterCollection`: owns all model parameters

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

## Dim Class Architecture

`Dim` stores tensor shape metadata separately from tensor values.

```text
Dim Structure:
┌─────────────────────────────────┐
│ Core Attributes:                │
│ • rows_: unsigned int           │
│ • cols_: unsigned int           │
│ • batch_: unsigned int          │
├─────────────────────────────────┤
│ Shape Helpers:                  │
│ • rows(), cols()                │
│ • batch_elems()                 │
│ • batch_size(), size()          │
│ • ndims()                       │
├─────────────────────────────────┤
│ Transformations:                │
│ • transpose()                   │
│ • single_batch()                │
└─────────────────────────────────┘
```

Batch is tracked separately, following DyNet's shape design:

```text
Dim(3, 4)      -> {3,4}
Dim(3, 4, 32)  -> {3,4X32}
```

## Parameter Architecture

Parameters are split into storage, handles, and collections.

```text
ParameterStorage:
  owns value tensor, gradient tensor, name, trainable flag

Parameter:
  lightweight shared handle to ParameterStorage

ParameterCollection:
  owns all ParameterStorage objects for a model
```

This keeps parameter copies cheap while preserving one real copy of each trainable
tensor.


# Current Layout

```text
fluxions/
├── core/      Dim metadata and Eigen-backed Tensor core
├── docs/      Eigen notes and development references
├── memory/    Memory pool and device metadata
├── ops/       Planned operation implementations
├── optim/     Planned optimizers
├── params/    Parameter storage, handles, and collections
└── tests/     GoogleTest tests
```

# Planned Layout

```text
fluxions/
├── core/
│   ├── dim.h
│   ├── dim.cc
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
│   └── parameter.h/.cc
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
