#pragma once
#include "tensor.hpp"
template <typename T> class Linear {
private:
  Tensor<T> weight;
  Tensor<T> bias;

public:
  Linear(int rows, int cols) : weight({rows, cols}), bias({cols}) {}

  Tensor<T> forward(const Tensor<T> &x) const {
    return x.matmul(weight) + bias;
  }
};
