#pragma once
#include "tensor.hpp"

template <typename T> class Losses {
public:
  T mse(const Tensor<T> &prediciton, const Tensor<T> &target) {
    Tensor<T> diff = prediciton - target;
    Tensor<T> squared = diff * diff;
    return squared.mean();
  }

  Tensor<T> mseBackward(const Tensor<T> &prediction, const Tensor<T> &target) {
    return 2 * (prediction - target) / target.size();
  }

  T softmaxCrossEntropy(const Tensor<T> &logits, const Tensor<T> &target) {
    Tensor<T> prediction = logits.softmax(1);
    Tensor<T> logPrediction = prediction.log();
    Tensor<T> loss = -(target * logPrediction);
    return loss.sum(1).mean();
  }
  Tensor<T> softmaxCrossEntropyBackward(const Tensor<T> &logits,
                                        const Tensor<T> &target) {
    Tensor<T> prediction = logits.softmax(1);
    T batchSize = logits.shape()[0];
    return (prediction - target) / batchSize;
  }
};
