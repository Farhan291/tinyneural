#pragma once

#include "tensor.hpp"

template <typename T> class Activation {
public:
  Tensor<T> sigmoid(const Tensor<T> &x) const { return 1 / (1 + (-x).exp()); }

  Tensor<T> relu(const Tensor<T> &x) const {
    Tensor<T> res(x.shape());
    for (int i = 0; i < x.size(); i++) {
      res[i] = x[i] > 0 ? x[i] : 0;
    }
    return res;
  }

  Tensor<T> tanh(const Tensor<T> &x) const {
    Tensor<T> ePos = x.exp();
    Tensor<T> eNeg = (-x).exp();
    return (ePos - eNeg) / (ePos + eNeg);
  }

  Tensor<T> sigmoidBackward(const Tensor<T> &x, const Tensor<T> &dY) {
    Tensor<T> s = sigmoid(x);
    return dY * s * (1 - s);
  }

  Tensor<T> reluBackward(const Tensor<T> &x, const Tensor<T> &dY) {
    Tensor<T> res(x.shape());
    for (int i = 0; i < x.size(); i++) {
      if (x[i] > 0)
        res[i] = 1;
      else
        res[i] = 0;
    }
    return dY * res;
  }

  Tensor<T> tanhBackward(const Tensor<T> &x, const Tensor<T> &dY) {
    return dY * (1 - tanh(x) * tanh(x));
  }

  Tensor<T> leakyRelu(const Tensor<T> &x, T alpha = T(0.01)) const {
    Tensor<T> res(x.shape());

    for (int i = 0; i < x.size(); i++) {
      res[i] = x[i] > T(0) ? x[i] : alpha * x[i];
    }

    return res;
  }

  Tensor<T> leakyReluBackward(const Tensor<T> &x, const Tensor<T> &dY,
                              T alpha = T(0.01)) const {
    Tensor<T> grad(x.shape());

    for (int i = 0; i < x.size(); i++) {
      grad[i] = x[i] > T(0) ? T(1) : alpha;
    }

    return dY * grad;
  }
};
