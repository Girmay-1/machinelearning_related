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
    def __init__(self, learning_rate=0.01, n_iter=1000, random_state=1):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        #initialize weights and bias
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0, scale=0.01, size=X.shape[1])
        self.bias = np.float(0.0)
        self.errors_ = []
        


        # Training the perceptron
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights += update * xi
                self.bias += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    # Activation function
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)
    
    
    