import numpy as np
# This is a simple implementation of a Perceptron algorithm
# The Perceptron is a type of linear classifier, and it is the simplest form of a neural network.
# It is used for binary classification tasks.
# The Perceptron algorithm is a supervised learning algorithm that learns a linear decision boundary
# to separate two classes of data points.
# The algorithm works by iteratively adjusting the weights and bias based on the prediction error
# The Perceptron algorithm is guaranteed to converge if the data is linearly separable.
# The Perceptron algorithm is not guaranteed to converge if the data is not linearly separable.
# The Perceptron algorithm is a simple and efficient algorithm for binary classification tasks.
# The Perceptron algorithm is a supervised learning algorithm that learns a linear decision boundary
# to separate two classes of data points.
class perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Training the perceptron
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                # Update weights and bias
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def activation_function(self, x): # Step function
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted