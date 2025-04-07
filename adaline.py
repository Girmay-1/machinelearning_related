import numpy as np
# This is a simple implementation of an Adaline algorithm
# The Adaline (Adaptive Linear Neuron) is a type of linear classifier, and it is the simplest form of a neural network.
# It is used for binary classification tasks.
# The Adaline algorithm is a supervised learning algorithm that learns a linear decision boundary
# the difference between Adaline and Perceptron is that Adaline uses a linear activation function.
# advantages of using linear activation function is that it is easier to optimize and converges faster.(as it is differentiable and convex)

class adaline:
    def __init__(self, learning_rate=0.01, n_iter=1000, random_state=1):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        #initialize weights and bias
        rgen = np.random.RandomState(self.random_state)
        self.weights = rgen.normal(loc=0, scale=0.01, size=X.shape[1])
        self.bias = np.float(0.0)
        self.losses_ = []
        
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.weights += self.learning_rate * 2 * X.T.dot(errors) / X.shape[0]
            # update bias
            self.bias += self.learning_rate *2.0 * errors.mean()
            loss = (errors ** 2).mean()
            self.losses_.append(loss)
        return self
    
    
    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def activation(self, X):
        return X
    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, 0)