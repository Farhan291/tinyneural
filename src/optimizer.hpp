#pragma once
#include "tensor.hpp"
#include <unordered_map>

template <typename T> class SGD {
private:
  T learningRate;
  T momentum;
  std::unordered_map<const Tensor<T> *, Tensor<T>> velocities;

public:
  SGD(T lr, T momentum = 0.9) : learningRate(lr), momentum(momentum) {}

  void update(Tensor<T> &param, const Tensor<T> &grad) {
    auto &v = velocities[&param];
    if (v.size() != param.size()) {
      v = Tensor<T>(param.shape());
      v.fill(0);
    }
    for (int i = 0; i < param.size(); i++) {
      v[i] = momentum * v[i] + grad[i];
      param[i] -= learningRate * v[i];
    }
  }

  void decayLearningRate(T factor) { learningRate *= factor; }
};
