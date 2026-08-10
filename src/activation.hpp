#pragma once

#include "tensor.hpp"

template <typename T> class Activation {
public:
  Tensor<T> sigmoid(const Tensor<T> &x) const { return 1 / (1 + (-x).exp()); }

  Tensor<T> relu(const Tensor<T> &x) const {
    Tensor<T> res(x.shape());
    for (int i = 0; i < x.size(); i++) {
      res[i] = std::max(T(0), x[i]);
    }
    return res;
  }

  Tensor<T> tanh(const Tensor<T> &x) const {
    Tensor<T> ePos = x.exp();
    Tensor<T> eNeg = (-x).exp();
    return (ePos - eNeg) / (ePos + eNeg);
  }
};
