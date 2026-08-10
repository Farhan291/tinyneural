#pragma once
#include "tensor.hpp"
template <typename T> class Linear {
private:
  Tensor<T> weight;
  Tensor<T> bias;
  Tensor<T> input;
  Tensor<T> weightgrad;
  Tensor<T> biasgrad;

public:
  Linear(int rows, int cols) : weight({rows, cols}), bias({cols}) {}

  Tensor<T> forward(const Tensor<T> &x) {
    input = x;
    return x.matmul(weight) + bias;
  }

  Tensor<T> backward(const Tensor<T> &dY) {
    // dW = X^T@dY
    weightgrad = input.transpose().matmul(dY);
    // db = sum(dY,axis)
    biasgrad = dY.sum(0);
    // dX = dY@ W^T
    Tensor<T> dX = dY.matmul(weight.transpose());
    return dX;
  }
};
