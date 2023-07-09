# Khirwadkar, Shubham
# 1001_980_371
# 2022_10_30
# Assignment_03_01

# %tensorflow_version 2.x
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each input data sample
        """
        self.input_dimensions = input_dimension
        self.weights = []
        self.biases = []
        self.transfer_functions = []

    def add_layer(self, num_nodes, transfer_function="Linear"):
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param transfer_function: Activation function for the layer. Possible values are:
        "Linear", "Relu","Sigmoid".
         :return: None
         """
        if not self.weights:
            w = tf.Variable(np.random.randn(self.input_dimensions, num_nodes), trainable=True)

        else:
            w = tf.Variable(np.random.randn(self.weights[-1].shape[1], num_nodes), trainable=True)

        self.weights.append(w)

        b = tf.Variable(np.random.randn(num_nodes, ), trainable=True)
        self.biases.append(b)

        self.transfer_functions.append(transfer_function)

    def get_weights_without_biases(self, layer_number):
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
         :param layer_number: Layer number starting from layer 0. This means that the first layer with
          activation function is layer zero
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """
        return self.weights[layer_number]

    def get_biases(self, layer_number):
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases).
         Note that the biases shape should be [1][number_of_nodes]
         """
        return self.biases[layer_number]

    def set_weights_without_biases(self, weights, layer_number):
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """
        self.weights[layer_number] = weights

    def set_biases(self, biases, layer_number):
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """
        self.biases[layer_number] = biases

    def calculate_loss(self, y, y_hat):
        """
        This function calculates the sparse softmax cross entropy loss.
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual output values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    def predict(self, X):
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """
        out = tf.Variable(X)

        for _l in range(len(self.weights)):
            w = tf.matmul(out, self.get_weights_without_biases(_l))
            b = tf.add(w, self.get_biases(_l))

            if self.transfer_functions[_l].lower() == "sigmoid":
                out = tf.nn.sigmoid(b)

            elif self.transfer_functions[_l].lower() == "linear":
                out = b

            elif self.transfer_functions[_l].lower() == "relu":
                out = tf.nn.relu(b)

        return out

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8):
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :return: None
         """
        inp = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        inp = inp.batch(batch_size)
        for i in range(num_epochs):
            for j, (x, y) in enumerate(inp):
                with tf.GradientTape(persistent=True) as gr:
                    out = self.predict(x)
                    loss = self.calculate_loss(y, out)
                    for _l in range(len(self.weights)):
                        w, b = gr.gradient(loss, [self.weights[_l], self.biases[_l]])
                        self.weights[_l].assign_sub(alpha * w)
                        self.biases[_l].assign_sub(alpha * b)

    def calculate_percent_error(self, X, y):
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        Note that the predicted class is the index of the node with maximum output.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """
        predict = self.predict(X).numpy()
        out = [np.argmax(v) for v in predict]
        return round(1 - accuracy_score(list(y), out), 3)

    def calculate_confusion_matrix(self, X, y):
        """
        Given input samples and corresponding desired (true) outputs as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m.
        """
        predict = self.predict(X).numpy()
        out = [np.argmax(v) for v in predict]
        return tf.math.confusion_matrix(y, out)
