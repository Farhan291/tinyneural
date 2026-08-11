// main.cpp
#include "activation.hpp"
#include "linear.hpp"
#include "losses.hpp"
#include "optimizer.hpp"
#include "tensor.hpp"

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using T = float;

// ---- load dataset ----
std::string loadText(const std::string &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("could not open " + path);
  std::stringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

// ---- one-hot encode a window of `context` chars into a flat row ----
void encodeWindow(Tensor<T> &X, int row, const std::string &text, int startIdx,
                  int context, const std::unordered_map<char, int> &stoi,
                  int vocabSize) {
  for (int c = 0; c < context; c++) {
    char ch = text[startIdx + c];
    int id = stoi.at(ch);
    X.at({row, c * vocabSize + id}) = T(1);
  }
}

int main() {
  std::string text = loadText("shakespeare.txt");

  // build vocab (must be deterministic across runs -> always built from
  // the same shakespeare.txt so ids line up with any saved model.bin)
  std::vector<char> chars;
  {
    bool seen[256] = {false};
    for (char c : text)
      if (!seen[(unsigned char)c]) {
        seen[(unsigned char)c] = true;
        chars.push_back(c);
      }
  }
  std::unordered_map<char, int> stoi;
  std::unordered_map<int, char> itos;
  for (int i = 0; i < (int)chars.size(); i++) {
    stoi[chars[i]] = i;
    itos[i] = chars[i];
  }
  int vocabSize = chars.size();
  std::cout << "vocab size: " << vocabSize << "\n";

  int context = 8;
  int hidden = 128;
  int batchSize = 64;
  int steps = 100000;
  T lr = 0.02;

  Linear<T> l1(context * vocabSize, hidden);
  Linear<T> l2(hidden, vocabSize);
  Activation<T> act;
  Losses<T> loss;
  SGD<T> opt(lr, 0.9);

  std::mt19937 rng(42);

  const std::string modelPath = "model.bin";
  std::ifstream check(modelPath, std::ios::binary);
  bool haveModel = check.good();
  check.close();

  if (haveModel) {
    std::cout << "loading saved model from " << modelPath << " ...\n";
    std::ifstream in(modelPath, std::ios::binary);
    l1.load(in);
    l2.load(in);
    std::cout << "loaded.\n";
  } else {
    std::cout << "no saved model found, training from scratch...\n";
    std::uniform_int_distribution<int> pickStart(0, (int)text.size() - context -
                                                        2);

    for (int step = 0; step < steps; step++) {
      Tensor<T> X({batchSize, context * vocabSize});
      Tensor<T> Y({batchSize, vocabSize});
      X.fill(0);
      Y.fill(0);

      for (int b = 0; b < batchSize; b++) {
        int idx = pickStart(rng);
        encodeWindow(X, b, text, idx, context, stoi, vocabSize);
        char target = text[idx + context];
        Y.at({b, stoi[target]}) = T(1);
      }

      // forward
      Tensor<T> h1 = l1.forward(X);
      Tensor<T> a1 = act.tanh(h1);
      Tensor<T> logits = l2.forward(a1);

      T lossVal = loss.softmaxCrossEntropy(logits, Y);

      // backward
      Tensor<T> dLogits = loss.softmaxCrossEntropyBackward(logits, Y);
      Tensor<T> dA1 = l2.backward(dLogits);
      Tensor<T> dH1 = act.tanhBackward(h1, dA1);
      l1.backward(dH1);

      l1.update(opt);
      l2.update(opt);

      if (step > 0 && step % 20000 == 0) {
        opt.decayLearningRate(0.5);
      }

      if (step % 200 == 0) {
        std::cout << "step " << step << " loss " << lossVal << "\n";
      }
    }

    std::ofstream out(modelPath, std::ios::binary);
    l1.save(out);
    l2.save(out);
    std::cout << "saved model to " << modelPath << "\n";
  }

  // ---- CLI generation loop ----
  std::cout << "\nType a seed string (>= " << context << " chars) or 'quit':\n";
  std::string seed;
  while (true) {
    std::cout << "> ";
    if (!std::getline(std::cin, seed))
      break;
    if (seed == "quit")
      break;
    if ((int)seed.size() < context) {
      std::cout << "seed too short, need >= " << context << " chars\n";
      continue;
    }

    std::string generated = seed;
    int genLen = 200;

    bool badSeed = false;
    for (int i = 0; i < genLen; i++) {
      std::string window =
          generated.substr(generated.size() - context, context);

      Tensor<T> X({1, context * vocabSize});
      X.fill(0);
      bool ok = true;
      for (int c = 0; c < context; c++) {
        auto it = stoi.find(window[c]);
        if (it == stoi.end()) {
          ok = false;
          break;
        }
        X.at({0, c * vocabSize + it->second}) = T(1);
      }
      if (!ok) {
        std::cout << "seed has unknown characters\n";
        badSeed = true;
        break;
      }

      Tensor<T> h1 = l1.forward(X);
      Tensor<T> a1 = act.tanh(h1);
      Tensor<T> logits = l2.forward(a1);
      Tensor<T> probs = logits.softmax(1);

      // temperature: lower = sharper/less random
      const T temperature = 0.8;
      Tensor<T> scaled(probs.shape());
      for (int j = 0; j < probs.size(); j++) {
        scaled[j] = std::pow(probs[j], T(1) / temperature);
      }
      T sum = 0;
      for (int j = 0; j < scaled.size(); j++) {
        sum += scaled[j];
      }
      for (int j = 0; j < scaled.size(); j++) {
        scaled[j] /= sum;
      }

      std::discrete_distribution<int> dist(&scaled[0], &scaled[0] + vocabSize);
      int nextId = dist(rng);
      generated += itos[nextId];
    }

    if (!badSeed) {
      std::cout << generated << "\n";
    }
  }

  return 0;
}
