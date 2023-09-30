import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
import tensorflow as tf

import seaborn as sns

# suppress unnecessary warning
import warnings
warnings.filterwarnings("ignore", message=r"`rcond` parameter", category=FutureWarning)

class Regression():
    def __init__(self, n_iter=1000, eta=1e-3):
        self.n_iter = n_iter 
        self.eta = eta

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        # bias term
        X = np.c_[np.ones((X.shape[0], 1)), X]
        
        n_samples, n_features = np.shape(X)
        self.costs = []
        self.coef_ = np.random.randn(n_features, 1)
        for _ in range(self.n_iter):
            y_pred = X.dot(self.coef_)
            regularization = self.regularization(self.coef_[1:])
            cost = mean_squared_error(y.flatten(), y_pred, squared=False) + regularization
            self.costs.append(cost)
            w_reg = self.regularization.grad(self.coef_[1:])
            dw = (2/n_samples) * np.dot(X.T, y_pred - y) + w_reg
            self.coef_ -= self.eta * dw
        return self 

    def predict(self, X):
        X = np.array(X)
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
            X = np.c_[np.ones((X.shape[0], 1)), X]
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
        return np.insert(gradient_penalty, 0, 0, axis=0)

relu = lambda x, deriv=False: x * (x > 0) if not deriv else 1 * (x > 0)
leakyrelu = lambda x, deriv=False: np.maximum(0.01 * x, x) if not deriv else (x <= 0).astype(float) * 0.01 + (x > 0).astype(float) * 1
sigmoid = lambda x, deriv=False: 1 / (1 + np.exp(-x)) if not deriv else sigmoid(x) * (1 - sigmoid(x))
        
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

class Solution():
    COLS = [
        'Customer ID', 'Name', 'Age', 'Profession', 'Location', 'Expense Type 1', 'Expense Type 2',
        'No. of Defaults', 'Property ID', 'Property Type', 'Co-Applicant', 'Property Price'
    ]
    COLS_WITH_NA = [
        'Gender', 'Income (USD)', 'Income Stability', 'Type of Employment', 'Current Loan Expenses (USD)',
        'Dependents', 'Credit Score', 'Has Active Credit Card', 'Property Age', 'Property Location', 'Loan Amount'
    ]
    COLS_ID = ['Customer ID', 'Property ID']
    COLS_INFO = ['Name', 'Gender']
    COLS_IGNORE = ['Property Age', 'Property Location', 'Property Type']
    COLS_TODO = ['Profession', 'Type of Employment']
    
    def __init__(self) -> None:
        self.df = pd.read_csv("train.csv")
        self.df = self.df.drop(columns=self.COLS_IGNORE + self.COLS_INFO)
        self.df = self.df[~self.df['Loan Amount'].isna()]
        self.df['Co-Applicant'] = self.df['Co-Applicant'].map({-999: np.nan, 0: "N", 1: "Y"})
        self.df = pd.get_dummies(self.df, columns=[
            'Location', 'Expense Type 1', 'Expense Type 2', 'Has Active Credit Card', 'Income Stability', 'Co-Applicant'
            ])
        self.df = self.df.dropna()
        # self.df = self.df[self.df['Income (USD)'] < 10**6]
    
    def get_data(self, on_cols=None):
        dataset = self.df.loc[:, ~self.df.columns.isin(self.COLS_ID + self.COLS_TODO)]
        if not on_cols:
            cols = dataset.columns.tolist()
            cols = cols[:4] + cols[6:7] + cols[4:6] + cols[7:]
        else:
            cols = on_cols
        self.dataset = dataset[cols]
        
        # ['Age', 'Loan Amount Request (USD)', 'Current Loan Expenses (USD)', 'Credit Score', 'Property Price', 'No. of Defaults', 'Co-Applicant', 'Loan Amount', 'Location_Rural', 'Location_Semi-Urban', 'Location_Urban', 'Expense Type 1_N', 'Expense Type 1_Y', 'Expense Type 2_N', 'Expense Type 2_Y', 'Has Active Credit Card_Active', 'Has Active Credit Card_Inactive', 'Has Active Credit Card_Unpossessed', 'Income Stability_High', 'Income Stability_Low']
        
        train_dataset = dataset.sample(frac=.8)
        test_dataset = dataset.drop(train_dataset.index)
        
        self.X_train = train_dataset.copy()
        self.X_test = test_dataset.copy()

        self.y_train = self.X_train.pop('Loan Amount').to_numpy().reshape(-1, 1)
        self.y_test = self.X_test.pop('Loan Amount').to_numpy().reshape(-1, 1)

    def inspect(self):
        sns.pairplot(self.df)

    def xgb(self):
        self.get_data()
        model = XGBRegressor()
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        return mean_squared_error(self.y_test.flatten(), y_pred, squared=False)

    def lr(self):
        self.get_data(on_cols=['Income (USD)', 'Loan Amount Request (USD)', 'Current Loan Expenses (USD)', 'Credit Score', 'Co-Applicant_Y', 'Co-Applicant_N', 'Property Price', 'Loan Amount'])
        model = ElasticNet()
        
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        return mean_squared_error(self.y_test.flatten(), y_pred, squared=False)
        
    def nn(self):
        self.get_data(on_cols=['Credit Score', 'Loan Amount Request (USD)', 'Co-Applicant_Y', 'Co-Applicant_N', 'Property Price', 'Current Loan Expenses (USD)', 'Loan Amount'])
        
        nn = NeuralNetwork(self.X_train, self.y_train)
        nn.train()
        y_pred = nn.predict(self.X_test)
        
        # nn.plot_cost()
        return mean_squared_error(self.y_test.flatten(), y_pred, squared=False)

    def peek(self):
        self.get_data()
        pd.set_option("display.max_columns", 100)
        print(self.X_train.head())
    
    def plot_col(self, col):
        self.get_data()
        plt.scatter(self.X_train[col], self.y_train, label='Data')
        plt.xlabel(col)
        plt.ylabel('Loan Amount')
        plt.legend()
        plt.show()

    def tfs(self):
        self.get_data()
        

        x = self.dataset.copy()
        y = x.pop('Loan Amount').to_numpy().reshape(-1, 1)

        norm = tf.keras.layers.Normalization(input_shape=[x.shape[1],], axis=None)
        norm.adapt(np.array(x))

        model = tf.keras.Sequential([
            norm,
            tf.keras.layers.Dense(128, activation='sigmoid'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        history = model.fit(
            x,
            y,
            validation_split=0.2,
            verbose=0,
            epochs=100
        )
        hist = pd.DataFrame(history.history)
        return hist.iloc[-1, -1]
        
    def __iter_score__(self, f, n_iter=10):
        for i in range(n_iter):
            print(f'scoring: {i+1}...\r', end='')
            yield f()
    def score(self, f, n_iter=10):
        scores = [score for score in self.__iter_score__(f, n_iter=n_iter)]
        return np.mean(scores)
        
        

if __name__ == "__main__":
    solution = Solution()
    lr_score = solution.score(solution.lr)
    nn_score = solution.score(solution.nn)
    xgb_score = solution.score(solution.xgb)
    print(f"""
{"Linear Regression": >30}: {lr_score}
{"Neural Network": >30}: {nn_score}
{"XGBoost": >30}: {xgb_score}
          """)