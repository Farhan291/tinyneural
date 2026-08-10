#pragma once
#include "tensor.hpp"

template <typename T> class Losses {
public:
  Tensor<T> mse(const Tensor<T> &prediciton, const Tensor<T> &target) {
    Tensor<T> diff = prediciton - target;
    Tensor<T> squared = diff * diff;
    return squared.mean();
  }

  Tensor<T> mseBackward(const Tensor<T> &prediction, const Tensor<T> &target) {
    return 2 * (prediction - target) / target.size();
  }
};
