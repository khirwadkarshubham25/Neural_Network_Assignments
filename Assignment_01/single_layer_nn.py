# Khirwadkar, Shubham
# 1001_980_371
# 2022_09_25
# Assignment_01_01

import numpy as np


class SingleLayerNN(object):
    def __init__(self, input_dimensions=2, number_of_nodes=4):
        """
        Initialize SingleLayerNN model and set all the weights and biases to random numbers.
        :param input_dimensions: The number of dimensions of the input data
        :param number_of_nodes: Note that number of neurons in the model is equal to the number of classes.
        """
        self.weights = None
        self.input_dimensions = input_dimensions
        self.number_of_nodes = number_of_nodes
        self.initialize_weights()

    def initialize_weights(self, seed=None):
        """
        Initialize the weights, initalize using random numbers.
        If seed is given, then this function should
        use the seed to initialize the weights to random numbers.
        :param seed: Random number generator seed.
        :return: None
        """
        if seed is not None:
            np.random.seed(seed)
        self.weights = np.random.randn(self.number_of_nodes, self.input_dimensions + 1)

    def set_weights(self, W):
        """
        This function sets the weight matrix (Bias is included in the weight matrix).
        :param W: weight matrix
        :return: None if the input matrix, w, has the correct shape.
        If the weight matrix does not have the correct shape, this function
        should not change the weight matrix and it should return -1.
        """
        if self.weights.shape != W.shape:
            return -1
        self.weights = W

    def get_weights(self):
        """
        This function should return the weight matrix(Bias is included in the weight matrix).
        :return: Weight matrix
        """
        return self.weights

    def predict(self, X):
        """
        Make a prediction on a batach of inputs.
        :param X: Array of input [input_dimensions,n_samples]
        :return: Array of model [number_of_nodes ,n_samples]
        Note that the activation function of all the nodes is hard limit.
        """
        biases = np.insert(X, 0, np.ones(X.shape[1]), axis=0)
        hard_limit_act = np.dot(self.weights, biases)
        hard_limit_act[hard_limit_act > 0], hard_limit_act[hard_limit_act < 0] = 1, 0
        return hard_limit_act

    def train(self, X, Y, num_epochs=10, alpha=0.1):
        """
        Given a batch of input and desired outputs, and the necessary hyperparameters (num_epochs and alpha),
        this function adjusts the weights using Perceptron learning rule.
        Training should be repeated num_epochs times.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        X = np.insert(X, 0, np.ones(X.shape[1]), axis=0)
        for i in range(num_epochs):
            for j in range(len(X[0])):
                hard_limit_act = np.dot(self.weights, X[:, j])
                hard_limit_act[hard_limit_act > 0], hard_limit_act[hard_limit_act < 0] = 1, 0
                err = Y[:, j] - hard_limit_act
                self.weights = self.weights + alpha * (np.dot(np.asmatrix(err).T, np.asmatrix(X[:, j])))

    def calculate_percent_error(self, X, Y):
        """
        Given a batch of input and desired outputs, this function calculates percent error.
        For each input sample, if the output is not the same as the desired output, Y,
        then it is considered one error. Percent error is 100*(number_of_errors/ number_of_samples).
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_nodes ,n_samples]
        :return percent_error
        """
        actual = self.predict(X)
        count = 0
        expected = np.asmatrix(Y)
        for i in range(len(X[0])):
            if (actual[:, i] == expected[:, i]).all():
                count += 1
        return ((len(X[0]) - count) * 100) / len(X[0])


if __name__ == "__main__":
    input_dimensions = 2
    number_of_nodes = 2

    model = SingleLayerNN(input_dimensions=input_dimensions, number_of_nodes=number_of_nodes)
    model.initialize_weights(seed=2)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                        [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    print("****** Model weights ******\n", model.get_weights())
    print("****** Input samples ******\n", X_train)
    print("****** Desired Output ******\n", Y_train)
    percent_error = []
    for k in range(20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.1)
        percent_error.append(model.calculate_percent_error(X_train, Y_train))
    print("******  Percent Error ******\n", percent_error)
    print("****** Model weights ******\n", model.get_weights())
