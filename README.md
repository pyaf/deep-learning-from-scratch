# Deep learning from scratch

Implementing deep learning algorithms in python with numpy.

#### Dependencies
  * Numpy
  * Matplotlib (for Loss plots only)

## Logistic Regression

#### Implementation -> [logistic-regression.ipynb](logistic-regression.ipynb)

#### Architecture
  * `ReLU` in hidden layers and `Sigmoid` in the output layer.
  * Number of layers and number of units in each layer can be set using `layers_dims` hyper-parameter.
  * Uses sigmoid cross entropy for loss computation.
  * Vectorized implementation.

## Softmax Classification

#### Implementation -> [softmax-classification.ipynb](softmax-classification.ipynb)

#### Architecture
  * `ReLU` in hidden layers and `Softmax` in the output layer.
  * Number of layers and number of units in each layer can be set using `layers_dims` hyper-parameter.
  * Uses softmax cross entropy for loss computation.
  * Vectorized implementation

## Vanilla char level RNN 

#### Implementation -> [vanilla-char-rnn.ipynb](vanilla-char-rnn.ipynb)

#### Architecture
  * `tanh` as activation for hidden state, `Softmax` at output.
  * `Adagrad` optimization.
  * Uses softmax cross entropy for loss computation.
