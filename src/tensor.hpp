#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>
template <typename T> class Tensor {
private:
  std::vector<T> data;
  std::vector<int> _shape;
  std::vector<int> stride;

public:
  Tensor(const std::vector<int> &shape) : _shape(shape) {
    int dimensions = 1;
    int sz = shape.size();
    stride.resize(sz);
    for (int i = sz - 1; i >= 0; i--) {
      if (shape[i] <= 0) {
        throw std::runtime_error("invalid dimension");
      }
      stride[i] = dimensions;
      dimensions *= shape[i];
    }
    data.resize(dimensions);
  }
  int offset(const std::vector<int> &indices) const {
    int n = indices.size();
    if (n != _shape.size()) {
      throw std::runtime_error("wrong no. of indices");
    }
    int off = 0;
    for (int i = 0; i < n; i++) {
      if (indices[i] < 0 || indices[i] >= _shape[i]) {
        throw std::runtime_error("tensor out of index");
      }
      off += indices[i] * stride[i];
    }
    return off;
  }

  T &at(const std::vector<int> &indices) { return data[offset(indices)]; }

  const T &at(const std::vector<int> &indices) const {
    return data[offset(indices)];
  }

  int size() const { return data.size(); }
  int rank() const { return _shape.size(); }
  const std::vector<int> &shape() const { return _shape; }

  void reshape(const std::vector<int> &newshape) {
    int dimensions = 1;
    for (int x : newshape) {
      if (x <= 0) {
        throw std::runtime_error("invalid dimension");
      }
      dimensions *= x;
    }
    if (dimensions != data.size()) {
      throw std::runtime_error("reshape changes tensor size");
    }
    _shape = newshape;
    int sz = _shape.size();
    stride.resize(sz);
    int cur = 1;
    for (int i = sz - 1; i >= 0; i--) {
      stride[i] = cur;
      cur *= _shape[i];
    }
  }
  void fill(const T &value) { std::fill(data.begin(), data.end(), value); }

  Tensor<T> transpose() const {
    std::vector<int> newshape = _shape;
    std::reverse(newshape.begin(), newshape.end());

    Tensor<T> result(newshape);

    for (int off = 0; off < size(); off++) {
      std::vector<int> indices = indicesFromOffset(off);

      std::reverse(indices.begin(), indices.end());

      result.at(indices) = data[off];
    }

    return result;
  }

  template <typename... Args> T &operator()(Args... args) {
    std::vector<int> indices{args...};
    return at(indices);
  }

  template <typename... Args> const T &operator()(Args... args) const {
    std::vector<int> indices{args...};
    return at(indices);
  }

  std::vector<int> indicesFromOffset(int offset) const {
    std::vector<int> indices(rank());
    for (int i = 0; i < rank(); i++) {
      indices[i] = offset / stride[i];
      offset %= stride[i];
    }
    return indices;
  }
};
