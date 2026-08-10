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

  Tensor<T> matmul(const Tensor<T> &other) {
    if (this->rank() < 2 || other.rank() < 2) {
      throw std::runtime_error("matmul requires tensor rank >=2 ");
    }
    // shape = [Batch1, batch2 ....., m,n]
    int r1 = this->rank();
    int r2 = other.rank();
    int m = this->_shape[r1 - 2];
    int k = this->_shape[r1 - 1];
    int k2 = other._shape[r1 - 2];
    int n = other._shape[r1 - 1];

    if (k != k2) {
      throw std::runtime_error("invalid shape");
    }
    if (r1 != r2) {
      throw std::runtime_error("batches size must match");
    }

    for (int i = 0; i < r1 - 2; i++) {
      if ((this->_shape[i] != other._shape[i])) {
        throw std::runtime_error("batches size must match");
      }
    }

    std::vector<int> reshape;
    for (int i = 0; i < r1 - 2; i++) {
      reshape.push_back(other._shape[i]);
    }
    reshape.push_back(m);
    reshape.push_back(n);

    Tensor<T> res(reshape);
    int batchSize = 1;
    for (int i = 0; i < r1 - 2; i++) {
      batchSize *= other._shape[i];
    }

    for (int batch = 0; batch < batchSize; batch++) {
      std::vector<int> batchIndices(r1 - 2);
      int temp = batch;
      for (int i = (r1 - 2) - 1; i >= 0; i--) {
        batchIndices[i] = temp % _shape[i];
        temp /= _shape[i];
      }

      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          T sum = 0;
          for (int x = 0; x < k; x++) {
            std::vector<int> aIndex = batchIndices;
            aIndex.push_back(i);
            aIndex.push_back(x);

            std::vector<int> bIndex = batchIndices;
            bIndex.push_back(x);
            bIndex.push_back(j);

            sum += this->at(aIndex) * other.at(bIndex);
          }
          std::vector<int> cIndex = batchIndices;
          cIndex.push_back(i);
          cIndex.push_back(j);

          res.at(cIndex) = sum;
        }
      }
    }
    return res;
  }

  // element wise operation
  Tensor<T> operator+(const Tensor<T> &other) const {
    if (_shape != other._shape) {
      throw std::runtime_error("incompartible dimensions");
    }
    Tensor<T> res(this->_shape);
    for (int i = 0; i < shape(); i++) {
      res[i] = data[i] + other.data[i];
    }
    return res;
  }
  Tensor<T> operator-(const Tensor<T> &other) const {
    if (_shape != other._shape) {
      throw std::runtime_error("incompartible dimensions");
    }
    Tensor<T> res(this->_shape);
    for (int i = 0; i < shape(); i++) {
      res[i] = data[i] - other.data[i];
    }
    return res;
  }
  Tensor<T> operator*(const Tensor<T> &other) const {
    if (_shape != other._shape) {
      throw std::runtime_error("incompartible dimensions");
    }
    Tensor<T> res(this->_shape);
    for (int i = 0; i < shape(); i++) {
      res[i] = data[i] * other.data[i];
    }
    return res;
  }
  Tensor<T> operator/(const Tensor<T> &other) const {
    if (_shape != other._shape) {
      throw std::runtime_error("incompartible dimensions");
    }
    Tensor<T> res(this->_shape);
    for (int i = 0; i < shape(); i++) {
      res[i] = data[i] / other.data[i];
    }
    return res;
  }
  // scalar operation
  Tensor<T> operator+(T scalar) const {
    Tensor<T> res(_shape);
    for (int i = 0; i < size(); i++) {
      res[i] = data[i] + scalar;
    }
    return res;
  }
  Tensor<T> operator-(T scalar) const {
    Tensor<T> res(_shape);
    for (int i = 0; i < size(); i++) {
      res[i] = data[i] - scalar;
    }
    return res;
  }
  Tensor<T> operator*(T scalar) const {
    Tensor<T> res(_shape);
    for (int i = 0; i < size(); i++) {
      res[i] = data[i] * scalar;
    }
    return res;
  }
  Tensor<T> operator/(T scalar) const {
    Tensor<T> res(_shape);
    for (int i = 0; i < size(); i++) {
      res[i] = data[i] / scalar;
    }
    return res;
  }
};
