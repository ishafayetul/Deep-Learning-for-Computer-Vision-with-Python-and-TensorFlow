### 1. Neuron
- Neuron is combination of **y=wx+b | sigmoid(y)** 
  - **w:** is weight
  - **b:** bias
  - **sigmoid** is applied as activation function to convert the value of *y* between *[0 to 1]*

### 2. Activation Function
- Why Important?
  - Helps to reduce value as final output in output layer
  - Without activation function, equation is linear, then hidden layers become useless. Complex problems can not been solved using linear equation.
- Different Activation Function
  - Step Function: Binary output *(0 or 1)*
  - Sigmoid : Output range *[0 to 1]*. Popular for output layer.
  - tanh: output range *[-1 , 1]*
  - Sigmoid & tanh has **Vanishing Gradient** problem since if derivative is close to ZERO learning becomes slow
  - ReLU(z): max(0,x)- if *x < 0* value is 0 otherwise value is x. Popular for hidden layers
  -  Leaky ReLU: max(0.1x, x)
- Graph:
![Graph](<src/Screenshot from 2025-04-29 15-04-52.png>)

### 3. Dervatives
- Dervatives and Partial Derivatives are used to minimize error by tunning the correct values of weights - Backpropagation Error

### 4. Loss and Cost Function
- **Loss** : error of output for individual set of inputs (neuron)
- **Cost Function**: Total error of all the outputs of the network.
![alt text](<src/Screenshot from 2025-04-29 15-27-59.png>)
- **Different Errors**
![alt text](<src/Screenshot from 2025-04-29 15-28-56.png>)
- **Why log loss is used in Logistic Regression** [Link](https://medium.com/@ThiyaneshwaranG/why-cant-we-use-mse-in-logistic-regression-c5ec5723e1c5#:~:text=As%20you%20know%2C%20Logistics%20regression,predicted%20value%20in%20the%20formula.&text=The%20MSE%20is%20just%201%20and%20is%20very%20minimum.&text=as%20there%20is%20a%20mismatch,and%20leave%20your%20valuable%20comments.)

### 5. Gradient Descent for NN
- Core of Supervised Machine Learning
- Used to calculate **weight(w) & Bias(b)**
- *Forward Pass*: guessing value of *w, b*
![Tunning w & b](<src/Screenshot from 2025-04-29 15-46-00.png>)
- **`coef, intercept = model.get_weights()`** to find the weights and bias
- The term **"gradient descent"** comes from two words:
  - **Gradient**: In mathematics, a gradient is a vector that points in the direction of the greatest rate of  increase of a function. It also indicates how steep the function is in different directions.

  - **Descent**: This means "going downward" or "decreasing."

  - So, gradient descent literally means: **"Going downward in the direction of the steepest slope."**

- **In machine learning:**
  - Gradient descent is an optimization algorithm used to minimize a function (usually a loss function) by iteratively moving in the direction of the negative gradient (i.e., the direction where the function decreases the fastest).
- **In short, gradient descent in machine learning is an algorithm used to minimize the loss function by updating model parameters step by step in the direction that reduces error the most.**