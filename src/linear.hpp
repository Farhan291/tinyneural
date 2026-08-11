#pragma once
#include "optimizer.hpp"
#include "tensor.hpp"
#include <cmath>
template <typename T> class Linear {
private:
  Tensor<T> weight;
  Tensor<T> bias;
  Tensor<T> input;
  Tensor<T> weightgrad;
  Tensor<T> biasgrad;

public:
  Linear(int rows, int cols) : weight({rows, cols}), bias({cols}), input() {
    T bound = std::sqrt(6.0 / rows);
    weight.random(-bound, bound);
    bias.fill(0.1);
  }

  Tensor<T> forward(const Tensor<T> &x) {
    input = x;
    return x.matmul2D(weight) + bias;
  }

  Tensor<T> backward(const Tensor<T> &dY) {
    // dW = X^T@dY
    weightgrad = input.transpose().matmul2D(dY);
    // db = sum(dY,axis)
    biasgrad = dY.sum(0);
    // dX = dY@ W^T
    Tensor<T> dX = dY.matmul2D(weight.transpose());
    return dX;
  }

  void update(SGD<T> &optimizer) {
    optimizer.update(weight, weightgrad);
    optimizer.update(bias, biasgrad);
  }
  void save(std::ofstream &out) const {
    weight.save(out);
    bias.save(out);
  }

  void load(std::ifstream &in) {
    weight.load(in);
    bias.load(in);
  }
};
