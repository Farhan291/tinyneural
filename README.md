# tinyneural

A tiny neural network implementation in C++ from scratch (no external ML
libraries), trained on Shakespeare's text. Includes a header-only `Tensor`
class, linear layers, activations, losses, and SGD with momentum.

## Features

- **Tensor** : multidimensional array with broadcasting, matmul (batched),
  transpose, reshape, reductions (`sum`, `mean`, `max`, `min`, argmax/argmin),
  `softmax`, element-wise and scalar operators
- **Linear** : fully connected layer with He initialization and momentum-SGD
  updates
- **Activations** : `sigmoid`, `tanh`, `relu`, `leakyRelu` (+ backward passes)
- **Losses** : MSE, softmax cross-entropy (+ backward passes)
- **Example** : character-level language model on Shakespeare: trains a
  2-layer MLP with tanh to predict the next character from 8 previous ones

## Build

Requires a C++17 compiler.

```bash
make all
```

Build in debug mode (much slower, not recommended for training):

```bash
g++ -std=c++17 -O0 -I src src/main.cpp -o app
```

## Run

```bash
make run
# or
./app
```

The program first trains the model (if `model.bin` does not exist), then
starts an interactive prompt. Type a seed string of at least 8 characters to
generate 200 characters of text:

```
> hello world
hello world make for my tor.
```

Type `quit` to exit.

**Note:** training takes ~20 minutes at `-O2` (100k steps). To retrain from
scratch, delete `model.bin` first — it is loaded automatically if present.

## Training configuration

Tunable parameters in `main.cpp`:

| Parameter   | Default | Description                              |
| ----------- | ------- | ---------------------------------------- |
| `context`   | 8       | characters of context fed to the network |
| `hidden`    | 128     | hidden layer size                        |
| `batchSize` | 64      | samples per step                         |
| `steps`     | 100000  | training steps (~110 epochs)             |
| `lr`        | 0.02    | learning rate, halved every 20k steps    |
| momentum    | 0.9     | SGD momentum                             |
| temperature | 0.8     | sampling temperature (lower = sharper)   |

## Tensor operations

- `matmul()` — batched matrix multiplication with broadcasting
- `matmul2D()` — fast path for rank-2 tensors
- `transpose()`, `reshape()`, `fill()`, `random()`
- `exp()`, `log()`
- `sum/mean/max/min/maxIndex/minIndex(axis, keepdim)`
- `softmax(axis)`
- `+ - * /` — tensor-tensor (broadcast) and tensor-scalar (both orders)

## Result

Trained on an AMD Ryzen 7 7735HS with `-O2`:

- loss ~4.17 (random) → ~1.85 at 20k steps → ~1.6 after 100k steps
- generated text at 20k steps already shows word-like structure

For a quick sanity check, reduce `steps` to 2000.

## References

- https://github.com/oktonion/neural-network-from-scratch-in-cpp#Activation-Function-Implementation
- https://github.com/Mear-MRK/Tensor_cpp

## Acknowledgement

- Shakespeare dataset from https://github.com/karpathy/char-rnn
