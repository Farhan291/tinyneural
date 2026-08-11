#pragma once
#include "tensor.hpp"

template <typename T> class SGD {
private:
  T learningRate;

public:
  SGD(const T lr) : learningRate(lr) {}

  void update(Tensor<T> &param, const Tensor<T> &grad) {
    param = param - learningRate * grad;
  }
};
