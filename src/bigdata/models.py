import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# suppress unnecessary warning
import warnings
warnings.filterwarnings("ignore", message=r"`rcond` parameter", category=FutureWarning)

class Regression():
    def __init__(self, n_iter=1000, eta=1e-3):
        self.n_iter = n_iter 
        self.eta = eta

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        # Add x_0 = 1 as the bias term.
        X = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Store number of samples and features in variables.
        n_samples, n_features = np.shape(X)
        self.costs = []
        # Initialize weights randomly from normal distribution.
        self.coef_ = np.random.randn(n_features, 1)
        # Batch gradient descent for number iterations = n_iter.
        for _ in range(self.n_iter):
            print(f"coef {self.coef_}")
            y_pred = X.dot(self.coef_)
            # Penalty term if regularized (don't include bias term).
            regularization = self.regularization(self.coef_[1:])
            # Calculate mse + penalty term if regularized.
            cost_function = mean_squared_error(y, y_pred, squared=False) + regularization
            self.costs.append(cost_function)
            # Regularization term of gradients (don't include bias term).
            gradient_reg = self.regularization.grad(self.coef_[1:])
            print(f"reg: {regularization}, deriv: {gradient_reg}")
            # Gradients of loss function.
            gradients = (2/n_samples) * np.dot(X.T, y_pred - y)
            gradients = gradients + gradient_reg
            # Update the weights.
            self.coef_ -= self.eta * gradients
        return self 

    def predict(self, X):
        X = np.array(X)
        # Add x_0 = 1 to each instance for the bias term.
        X = np.c_[np.ones((X.shape[0], 1)), X]

        return X.dot(self.coef_)

class LinearRegression(Regression):
    def __init__(self, n_iter=2000, eta=1e-4, solver='LSTSQ'):
        self.solver = solver 
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iter=n_iter, eta=eta)

    def fit(self, X, y):
        if self.solver == 'LSTSQ':
            X = np.array(X)
            y = np.array(y)
            # Add x_0 = 1 to each instance for the bias term.
            X = np.c_[np.ones((X.shape[0], 1)), X]
            # np implementation of least squares.
            self.coef_, residues, rank, singular = np.linalg.lstsq(X, y)
        elif self.solver == 'GD': 
            super(LinearRegression, self).fit(X, y)

class Ridge(Regression):
    def __init__(self, n_iter=1000, eta=1e-3, lmda=1.0):
        self.lmda = lmda
        self.regularization = l2_regularization(lmda=self.lmda)
        super(Ridge, self).__init__(n_iter=n_iter, eta=eta)
        
class l2_regularization():
    def __init__(self, lmda=1.0):
        self.lmda = lmda 
    def __call__(self, w):
        return self.lmda * 0.5 * np.linalg.norm(w, 2)
    def grad(self, w):
        gradient_penalty = self.lmda * w
        # Insert 0 for bias term.
        return np.insert(gradient_penalty, 0, 0, axis=0)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

relu = lambda x, deriv=False: x * (x > 0) if not deriv else 1 * (x > 0)
leakyrelu = lambda x, deriv=False: np.maximum(0.01 * x, x) if not deriv else (x <= 0).astype(float) * 0.01 + (x > 0).astype(float) * 1
sigmoid = lambda x, deriv=False: 1 / (1 + np.exp(-x)) if not deriv else sigmoid(x) * (1 - sigmoid(x))

"""
    def feedforward(self):
        self.l1 = self.act_f(np.dot(self.input, self.w1))
        self.output = self.act_f(np.dot(self.l1, self.w2))
    
    def backprop(self):
        d_w2 = np.dot(self.l1.T, (2*(self.y - self.output) * self.act_f(self.output, deriv=True)))
        d_w1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * self.act_f(self.output, deriv=True), self.w2.T) * self.act_f(self.l1, deriv=True)))
"""
        
class NeuralNetwork():
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.hiddenSize = 64
        # print(f"setup: {self.x.shape[1]} -> {self.hiddenSize} -> {self.y.shape[1]}")

        self.w1 = np.random.rand(self.x.shape[1], self.hiddenSize)
        self.w2 = np.random.rand(self.hiddenSize, self.y.shape[1])

        self.costs = []
        
        self.act_f = relu

    def feedforward(self):
        self.a1 = np.matmul(self.x, self.w1)
        self.z1 = self.act_f(self.a1)
        self.a2 = np.matmul(self.z1, self.w2)
        self.out = self.act_f(self.a2)

    def backprop(self):
        self.err = self.y - self.out

        d_w1 = np.dot(self.x.T, (np.dot(2 * self.err * self.act_f(self.out, deriv=True), self.w2.T) * self.act_f(self.z1, deriv=True)))
        d_w2 = np.dot(self.z1.T, (2 * self.err * self.act_f(self.out, deriv=True)))
        
        self.w1 += d_w1
        self.w2 += d_w2

    def train(self, epochs=100, suppress_prompt=True):
        for i in range(int(epochs / 10)):
            for _ in range(10):
                self.feedforward()
                self.costs.append(mean_squared_error(self.y, self.out, squared=False))
                self.backprop()
            if not suppress_prompt:
                print(f"{i+1}0th...\r", end="")

    def predict(self, x):
        self.x = x
        self.feedforward()
        return self.out

    def plot_cost(self):
        plt.plot(range(len(self.costs)), self.costs)
        plt.title('Mean Sum Squared Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    