#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
template <typename T> class Tensor {
private:
  std::vector<T> data;
  std::vector<int> _shape;
  std::vector<int> stride;

public:
  Tensor() = default;

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
    if (n != (int)_shape.size()) {
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

  T &operator[](int i) { return data[i]; }

  const T &operator[](int i) const { return data[i]; }

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
    if (dimensions != (int)data.size()) {
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
    if (rank() < 2) {
      throw std::runtime_error("transpose requires rank >= 2");
    }
    std::vector<int> newShape = _shape;
    std::swap(newShape[rank() - 2], newShape[rank() - 1]);

    Tensor<T> result(newShape);

    for (int off = 0; off < size(); off++) {
      std::vector<int> indices = indicesFromOffset(off);
      std::swap(indices[rank() - 2], indices[rank() - 1]);
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

  Tensor<T> matmul(const Tensor<T> &other) const {
    int r1 = this->rank();
    int r2 = other.rank();
    if (r1 < 2 || r2 < 2) {
      throw std::runtime_error("matmul requires tensor rank >= 2");
    }
    // shape = [Batch1, batch2 ....., m,n]
    int m = this->_shape[r1 - 2];
    int k = this->_shape[r1 - 1];
    int k2 = other._shape[r2 - 2];
    int n = other._shape[r2 - 1];

    if (k != k2) {
      throw std::runtime_error("invalid shape");
    }
    std::vector<int> aBatchShape(_shape.begin(), _shape.end() - 2);
    std::vector<int> bBatchShape(other._shape.begin(), other._shape.end() - 2);
    std::vector<int> batchShape = broadcastShape(aBatchShape, bBatchShape);
    std::vector<int> reshape = batchShape;
    reshape.push_back(m);
    reshape.push_back(n);

    Tensor<T> res(reshape);
    int batchSize = 1;
    for (auto d : batchShape) {
      batchSize *= d;
    }

    for (int batch = 0; batch < batchSize; batch++) {
      std::vector<int> batchIndices(batchShape.size());
      int temp = batch;
      for (int i = (batchShape.size()) - 1; i >= 0; i--) {
        batchIndices[i] = temp % batchShape[i];
        temp /= _shape[i];
      }

      std::vector<int> aBatchIndex = broadcastIndex(batchIndices, aBatchShape);
      std::vector<int> bBatchIndex = broadcastIndex(batchIndices, bBatchShape);
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          T sum = 0;
          for (int x = 0; x < k; x++) {
            std::vector<int> aIndex = aBatchIndex;
            aIndex.push_back(i);
            aIndex.push_back(x);

            std::vector<int> bIndex = bBatchIndex;
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
  // broadcast
  std::vector<int> broadcastShape(const std::vector<int> &ashape,
                                  const std::vector<int> &bshape) const {
    int n = std::max(ashape.size(), bshape.size());
    std::vector<int> res(n);
    for (int k = 0; k < n; k++) {
      int i = ashape.size() - k - 1;
      int j = bshape.size() - k - 1;

      int adim = (i >= 0) ? ashape[i] : 1;
      int bdim = (j >= 0) ? bshape[j] : 1;
      if (adim != bdim && adim != 1 && bdim != 1) {
        throw std::runtime_error("incompatible broadcast shapes");
      }
      res[n - k - 1] = std::max(adim, bdim);
    }
    return res;
  }
  std::vector<int> broadcastIndex(const std::vector<int> &resultIndex,
                                  const std::vector<int> &originalShape) const {
    int offset = (int)resultIndex.size() - (int)originalShape.size();
    std::vector<int> index(originalShape.size());
    for (int i = 0; i < (int)originalShape.size(); i++) {
      int resultDim = i + offset;
      if (originalShape[i] == 1) {
        index[i] = 0;
      } else {
        index[i] = resultIndex[resultDim];
      }
    }
    return index;
  }

  // element wise operation
  // unary
  Tensor<T> operator-() const {
    Tensor<T> res(_shape);

    for (int i = 0; i < size(); i++) {
      res[i] = -data[i];
    }

    return res;
  }
  Tensor<T> exp() const {
    Tensor<T> res(_shape);

    for (int i = 0; i < size(); i++) {
      res[i] = std::exp(data[i]);
    }

    return res;
  }
  // binary
  Tensor<T> operator+(const Tensor<T> &other) const {
    /*if (_shape != other._shape) {
      throw std::runtime_error("incompartible dimensions");
    }*/
    std::vector<int> newShape = broadcastShape(_shape, other._shape);
    Tensor<T> res(newShape);
    for (int off = 0; off < res.size(); off++) {
      std::vector<int> resultIndex = res.indicesFromOffset(off);
      std::vector<int> aIndex = broadcastIndex(resultIndex, _shape);
      std::vector<int> bIndex = broadcastIndex(resultIndex, other._shape);
      res.at(resultIndex) = this->at(aIndex) + other.at(bIndex);
    }
    return res;
  }
  Tensor<T> operator-(const Tensor<T> &other) const {
    /*if (_shape != other._shape) {
      throw std::runtime_error("incompartible dimensions");
    }*/
    std::vector<int> newShape = broadcastShape(_shape, other._shape);
    Tensor<T> res(newShape);
    for (int off = 0; off < res.size(); off++) {
      std::vector<int> resultIndex = res.indicesFromOffset(off);
      std::vector<int> aIndex = broadcastIndex(resultIndex, _shape);
      std::vector<int> bIndex = broadcastIndex(resultIndex, other._shape);
      res.at(resultIndex) = this->at(aIndex) - other.at(bIndex);
    }
    return res;
  }
  Tensor<T> operator*(const Tensor<T> &other) const {
    /*if (_shape != other._shape) {
      throw std::runtime_error("incompartible dimensions");
    }*/
    std::vector<int> newShape = broadcastShape(_shape, other._shape);
    Tensor<T> res(newShape);
    for (int off = 0; off < res.size(); off++) {
      std::vector<int> resultIndex = res.indicesFromOffset(off);
      std::vector<int> aIndex = broadcastIndex(resultIndex, _shape);
      std::vector<int> bIndex = broadcastIndex(resultIndex, other._shape);
      res.at(resultIndex) = this->at(aIndex) * other.at(bIndex);
    }
    return res;
  }
  Tensor<T> operator/(const Tensor<T> &other) const {
    /*if (_shape != other._shape) {
      throw std::runtime_error("incompartible dimensions");
    }*/
    std::vector<int> newShape = broadcastShape(_shape, other._shape);
    Tensor<T> res(newShape);
    for (int off = 0; off < res.size(); off++) {
      std::vector<int> resultIndex = res.indicesFromOffset(off);
      std::vector<int> aIndex = broadcastIndex(resultIndex, _shape);
      std::vector<int> bIndex = broadcastIndex(resultIndex, other._shape);
      res.at(resultIndex) = this->at(aIndex) / other.at(bIndex);
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
  // reductions {sum,mean,max,min,argmax,argmin}
  Tensor<T> sum(int axis, bool keepdim = false) const {
    if (axis < 0 || axis >= rank()) {
      throw std::runtime_error("invalid axis");
    }
    std::vector<int> newShape;
    for (int i = 0; i < rank(); i++) {
      if (i != axis) {
        newShape.push_back(_shape[i]);
      } else {
        if (keepdim)
          newShape.push_back(1);
      }
    }
    Tensor<T> res(newShape);
    res.fill(0);
    for (int off = 0; off < res.size(); off++) {
      std::vector<int> resIndex = res.indicesFromOffset(off);
      T sum = 0;
      for (int x = 0; x < _shape[axis]; x++) {
        std::vector<int> inputIndex;
        int j = 0;
        for (int i = 0; i < rank(); i++) {
          if (i == axis) {
            inputIndex.push_back(x);
          } else {
            inputIndex.push_back(keepdim ? resIndex[i] : resIndex[j++]);
          }
        }
        sum += at(inputIndex);
      }
      res.at(resIndex) = sum;
    }
    return res;
  }
  Tensor<T> mean(int axis, bool keepdim = false) const {
    Tensor<T> res = sum(axis, keepdim);
    T count = _shape[axis];
    return res / count;
  }
  T mean() const {
    if (size() == 0) {
      throw std::runtime_error("mean of empty tensor");
    }
    T sum = 0;
    for (int i = 0; i < size(); i++) {
      sum += data[i];
    }
    return sum / static_cast<T>(size());
  }
  Tensor<T> max(int axis, bool keepdim = false) const {
    if (axis < 0 || axis >= rank()) {
      throw std::runtime_error("invalid axis");
    }
    std::vector<int> newShape;
    for (int i = 0; i < rank(); i++) {
      if (i != axis)
        newShape.push_back(_shape[i]);
      else {
        if (keepdim)
          newShape.push_back(1);
      }
    }
    Tensor<T> res(newShape);
    for (int off = 0; off < res.size(); off++) {
      T maxi = std::numeric_limits<T>::lowest();
      std::vector<int> resIndex = res.indicesFromOffset(off);
      for (int x = 0; x < _shape[axis]; x++) {
        std::vector<int> inputIndex;
        int j = 0;
        for (int i = 0; i < rank(); i++) {
          if (i == axis) {
            inputIndex.push_back(x);
          } else
            inputIndex.push_back(keepdim ? resIndex[i] : resIndex[j++]);
        }
        maxi = std::max(maxi, at(inputIndex));
      }
      res.at(resIndex) = maxi;
    }
    return res;
  }
  Tensor<T> min(int axis, bool keepdim = false) const {
    if (axis < 0 || axis >= rank()) {
      throw std::runtime_error("invalid axis");
    }
    std::vector<int> newShape;
    for (int i = 0; i < rank(); i++) {
      if (i != axis)
        newShape.push_back(_shape[i]);
      else {
        if (keepdim)
          newShape.push_back(1);
      }
    }
    Tensor<T> res(newShape);
    for (int off = 0; off < res.size(); off++) {
      T mini = std::numeric_limits<T>::max();
      std::vector<int> resIndex = res.indicesFromOffset(off);
      for (int x = 0; x < _shape[axis]; x++) {
        std::vector<int> inputIndex;
        int j = 0;
        for (int i = 0; i < rank(); i++) {
          if (i == axis) {
            inputIndex.push_back(x);
          } else
            inputIndex.push_back(keepdim ? resIndex[i] : resIndex[j++]);
        }
        mini = std::min(mini, at(inputIndex));
      }
      res.at(resIndex) = mini;
    }
    return res;
  }
  Tensor<T> maxIndex(int axis, bool keepdim = false) const {
    if (axis < 0 || axis >= rank()) {
      throw std::runtime_error("invalid axis");
    }
    std::vector<int> newShape;
    for (int i = 0; i < rank(); i++) {
      if (i != axis)
        newShape.push_back(_shape[i]);
      else {
        if (keepdim)
          newShape.push_back(1);
      }
    }
    Tensor<T> res(newShape);
    for (int off = 0; off < res.size(); off++) {
      T maxi = std::numeric_limits<T>::lowest();
      std::vector<int> resIndex = res.indicesFromOffset(off);
      int maxIndex = 0;
      for (int x = 0; x < _shape[axis]; x++) {
        std::vector<int> inputIndex;
        int j = 0;
        for (int i = 0; i < rank(); i++) {
          if (i == axis) {
            inputIndex.push_back(x);
          } else
            inputIndex.push_back(keepdim ? resIndex[i] : resIndex[j++]);
        }
        if (maxi < at(inputIndex)) {
          maxi = at(inputIndex);
          maxIndex = x;
        }
      }
      res.at(resIndex) = maxIndex;
    }
    return res;
  }
  Tensor<T> minIndex(int axis, bool keepdim = false) const {
    if (axis < 0 || axis >= rank()) {
      throw std::runtime_error("invalid axis");
    }
    std::vector<int> newShape;
    for (int i = 0; i < rank(); i++) {
      if (i != axis)
        newShape.push_back(_shape[i]);
      else {
        if (keepdim)
          newShape.push_back(1);
      }
    }
    Tensor<T> res(newShape);
    for (int off = 0; off < res.size(); off++) {
      T mini = std::numeric_limits<T>::max();
      std::vector<int> resIndex = res.indicesFromOffset(off);
      int minIndex = 0;
      for (int x = 0; x < _shape[axis]; x++) {
        std::vector<int> inputIndex;
        int j = 0;
        for (int i = 0; i < rank(); i++) {
          if (i == axis) {
            inputIndex.push_back(x);
          } else
            inputIndex.push_back(keepdim ? resIndex[i] : resIndex[j++]);
        }
        if (mini > at(inputIndex)) {
          mini = at(inputIndex);
          minIndex = x;
        }
      }
      res.at(resIndex) = minIndex;
    }
    return res;
  }

  // softmax
  Tensor<T> softmax(int axis) const {
    Tensor<T> maxVal = max(axis, true);
    Tensor shifted = *this - maxVal;
    Tensor<T> expVal = shifted.exp();
    Tensor<T> sumExp = expVal.sum(axis, true);
    return expVal / sumExp;
  }

  // random generator
  void random(T min, T max) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(min, max);

    for (int i = 0; i < size(); i++) {
      data[i] = dist(gen);
    }
  }
};

template <typename S, typename T>
Tensor<T> operator/(S scalar, const Tensor<T> &tensor) {
  Tensor<T> res(tensor.shape());
  for (int i = 0; i < tensor.size(); i++) {
    res[i] = static_cast<T>(scalar) / tensor[i];
  }
  return res;
}

template <typename S, typename T>
Tensor<T> operator*(S scalar, const Tensor<T> &tensor) {
  Tensor<T> res(tensor.shape());
  for (int i = 0; i < tensor.size(); i++) {
    res[i] = static_cast<T>(scalar) * tensor[i];
  }
  return res;
}

template <typename S, typename T>
Tensor<T> operator+(S scalar, const Tensor<T> &tensor) {
  Tensor<T> res(tensor.shape());
  for (int i = 0; i < tensor.size(); i++) {
    res[i] = static_cast<T>(scalar) + tensor[i];
  }
  return res;
}

template <typename S, typename T>
Tensor<T> operator-(S scalar, const Tensor<T> &tensor) {
  Tensor<T> res(tensor.shape());
  for (int i = 0; i < tensor.size(); i++) {
    res[i] = static_cast<T>(scalar) - tensor[i];
  }
  return res;
}
