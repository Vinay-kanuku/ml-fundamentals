"""
regresion.py
-------------

A modular implementation of linear regression using three different approaches:
- Normal Equation
- Gradient Descent
- Scikit-learn Wrapper

Includes:
- Base class for linear regression models
- Model training, prediction, and evaluation (MSE, RMSE, MAE, R2, Adjusted R2)
- Feature standardization utility
- Convergence plotting for gradient descent
- Example usage with California Housing dataset

Classes:
--------
BaseLinearRegression (ABC):
    Abstract base class for linear regression models. Defines interface for training, prediction, and evaluation.

NormalEquationLinearRegression(BaseLinearRegression):
    Implements closed-form solution using the normal equation.

GradientDescentLinearRegression(BaseLinearRegression):
    Implements linear regression using batch gradient descent. Tracks loss history for convergence plotting.

SklearnLinearRegressionWrapper(BaseLinearRegression):
    Wraps scikit-learn's LinearRegression for interface consistency.

Functions:
----------
standardize_data(X_train, X_test):
    Standardizes features using sklearn's StandardScaler.

Usage:
------
- Loads California Housing dataset
- Splits into train/test sets
- Standardizes features
- Instantiates and trains a selected regression model
- Evaluates and prints metrics
- Plots convergence (if using gradient descent)

"""

import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


class BaseLinearRegression(ABC):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.weights = None

    @abstractmethod
    def train(self):
        pass

    def predict(self, X=None):
        if X is None:
            X = self.X_test
        X_bias = np.hstack([np.ones((X.shape[0], 1)), X])
        return X_bias @ self.weights

    def evaluate(self):
        y_true = self.y_test
        y_pred = self.predict()

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        n, p = self.X_test.shape
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "Adjusted R2": adj_r2
        }


class NormalEquationLinearRegression(BaseLinearRegression):
    def train(self):
        X_bias = np.hstack([np.ones((self.X_train.shape[0], 1)), self.X_train])
        self.weights = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ self.y_train


class GradientDescentLinearRegression(BaseLinearRegression):
    def __init__(self, X_train, y_train, X_test, y_test,
                 lr=0.001, n_iters=5000):
        super().__init__(X_train, y_train, X_test, y_test)
        self.lr = lr
        self.n_iters = n_iters
        self.loss_history = []

    def train(self):
        X_bias = np.hstack([np.ones((self.X_train.shape[0], 1)), self.X_train])
        y = self.y_train.reshape(-1, 1)
        m, n = X_bias.shape
        self.weights = np.zeros((n, 1))

        for i in range(self.n_iters):
            preds = X_bias @ self.weights
            error = preds - y
            gradient = (2 / m) * X_bias.T @ error
            self.weights -= self.lr * gradient

            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

            if i % 500 == 0:
                print(f"Iteration {i}, Loss: {loss:.6f}")

        self.weights = self.weights.flatten()

    def plot_convergence(self):
        plt.plot(self.loss_history)
        plt.title("Gradient Descent Convergence")
        plt.xlabel("Iteration")
        plt.ylabel("MSE Loss")
        plt.grid(True)
        plt.show()


class SklearnLinearRegressionWrapper(BaseLinearRegression):
    def train(self):
        model = SklearnLR()
        model.fit(self.X_train, self.y_train)
        intercept = model.intercept_
        coef = model.coef_
        self.weights = np.concatenate(([intercept], coef))


def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


if __name__ == "__main__":
    # Load and split data
    data = fetch_california_housing()
    X, y = data['data'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Standardize features (important for GD)
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

    # ---- Choose Solver ----
    # model = NormalEquationLinearRegression(X_train, y_train, X_test, y_test)
    model = GradientDescentLinearRegression(
        X_train_scaled, y_train, X_test_scaled, y_test,
        lr=0.001, n_iters=900
    )
    # model = SklearnLinearRegressionWrapper(X_train, y_train, X_test, y_test)
 

    model.train()
    results = model.evaluate()

 
    print("\nFinal Evaluation Metrics:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")



    model.plot_convergence()

