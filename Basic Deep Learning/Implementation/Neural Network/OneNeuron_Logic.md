
# OneNeuron: Single Neuron Neural Network

## Initialization
- The `OneNeuron` class initializes with `input_size` and `output_size`.
- The input size is unpacked into number of rows and columns.
- The weights `w1` and `w2` are initialized to 1, and the bias `b` is initialized to 0.

## Activation Function
- A sigmoid activation function is defined:
  ```
  sigmoid(z) = 1 / (1 + exp(-z))
  ```
- It squashes the linear combination output into a value between 0 and 1.

## Forward Propagation
- The `forward(x)` method computes:
  ```
  z = w1 * x[:,0] + w2 * x[:,1] + b
  p = sigmoid(z)
  ```
- `p` is the prediction output of the neuron.

## Loss Calculation
- The `calcuLoss(y_true, y_pred)` method calculates the binary cross-entropy loss:
  ```
  Loss = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
  ```
- `y_pred` is clipped between a small epsilon and 1-epsilon to prevent issues with log(0).

## Derivatives Calculation
- The `derivative(x, y_true, y_pred)` method calculates gradients for weight and bias updates:
  ```
  error = y_pred - y_true
  dw1 = mean(error * x[:,0])
  dw2 = mean(error * x[:,1])
  db = mean(error)
  ```
- These derivatives are necessary for updating parameters through gradient descent.

## Weights Update
- The `update_weights(learning_rate)` method updates weights and bias:
  ```
  w1 = w1 - learning_rate * dw1
  w2 = w2 - learning_rate * dw2
  b = b - learning_rate * db
  ```

## Accuracy Calculation
- The `calcuAccuracy(y_true, y_pred)` method:
  - Converts probabilities to class labels (0 or 1) using 0.5 as a threshold.
  - Computes the percentage of correct predictions:
    ```
    accuracy = mean(y_true == y_pred_binary)
    ```

## Prediction
- The `predict(x)` method uses `forward(x)` to generate predictions.

## Training (Single Epoch)
- The `train(x, y_true, learning_rate)` method:
  - Performs a forward pass to compute predictions.
  - Calculates loss and accuracy.
  - Computes gradients.
  - Updates weights and bias using gradient descent.
  - Returns the loss and accuracy for the epoch.

## Fitting (Multiple Epochs)
- The `fit(x, y_true, epochs, learning_rate)` method:
  - Repeats the training process for a number of epochs.
  - Prints loss and accuracy every 100 epochs.
  - Early stops training if loss drops below 0.5298.
  - Prints the final loss and accuracy after training ends.

## Getting Parameters
- The `getParameters()` method returns the final learned weights `w1`, `w2`, and bias `b`.
